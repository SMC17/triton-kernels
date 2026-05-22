"""
04 — Flash-Attention v2 BACKWARD pass (fp32, causal optional) on Triton.

Computes (dQ, dK, dV) given (Q, K, V, O, dO, L) where L = log-sum-exp
saved by the forward (`03_flash_attention.py` with `save_lse=True`).

Algorithm (Dao 2023, "Flash-Attention 2," Section 3):

  Preprocess:
    D_i = row-wise dot of (dO * O), shape [B, H, M]

  dK / dV kernel (outer loop over K, V blocks along N):
    For each K, V block j:
      dK_j, dV_j ← 0
      For each Q block i:
        Recompute S_ij = Q_i @ K_j^T * sm_scale
        P_ij = exp(S_ij - L_i)                 # stable with saved L
        dV_j += P_ij^T @ dO_i
        dP_ij = dO_i @ V_j^T
        dS_ij = P_ij * (dP_ij - D_i)           # softmax-Jacobian product
        dK_j += dS_ij^T @ Q_i * sm_scale
      Write dK_j, dV_j

  dQ kernel (outer loop over Q blocks along M):
    For each Q block i:
      dQ_i ← 0
      For each K, V block j:
        Recompute S_ij = Q_i @ K_j^T * sm_scale
        P_ij = exp(S_ij - L_i)
        dP_ij = dO_i @ V_j^T
        dS_ij = P_ij * (dP_ij - D_i)
        dQ_i += dS_ij @ K_j * sm_scale
      Write dQ_i

The two-kernel split lands cleaner than a fused backward on Turing
because the access patterns for dK/dV and dQ are transposed: dK/dV
parallelizes over K/V tiles and sequentially reduces over Q tiles,
while dQ parallelizes over Q tiles and reduces over K/V tiles. A
fused kernel would force one of these to spill or to issue an
extra atomic-accumulation pass.

Memory: the backward never materializes the N×M attention matrix.
P_ij is recomputed inside the inner loop from Q_i, K_j and the
saved L_i (one fp32 scalar per query row), exactly as in Dao 2023.

Scope:
- fp32 only. fp16 backward = v0.4.1. The recomputation of P inside
  the backward has tighter numerical-stability constraints than the
  forward (a small fp16 error in S_ij propagates through exp into P,
  through P^T @ dO into dV, and the cumulative error can blow past
  the atol=1e-2 tolerance that fp16 forward operated under).
- Causal mask supported. Same triangular pattern as the forward.
- Multi-batch + multi-head. Each (B, H) pair is a separate grid
  slot, same as the forward.
- Variable-length sequences (attention masks beyond strict causal)
  NOT supported. Would need additional masking infrastructure that
  doesn't transfer from the forward's `n_offs < N` pattern.
- Dropout NOT in backward. Standard FA-2 supports dropout in both
  forward and backward with a shared RNG state; v0.4 ships without
  to keep the diff focused.
- No gradient checkpointing — the backward recomputes the forward
  state (S_ij, P_ij) but does not save activations.

Correctness check: PyTorch autograd with
`torch.nn.functional.scaled_dot_product_attention` + `.backward()`.
Tolerance `atol=1e-3` for each of dQ, dK, dV; the recomputation
pattern introduces visible fp32 rounding drift relative to torch's
fused-kernel reference.

Hardware: GTX 1650 Max-Q (Turing, compute capability 7.5).
"""

from __future__ import annotations
import argparse
import json
import time
import torch
import triton
import triton.language as tl

# Forward + LSE save lives in 03_flash_attention.py. Import the public
# entry point so the backward harness can drive the full forward+
# backward round-trip from one script.
import importlib.util
import os
_FWD_PATH = os.path.join(os.path.dirname(__file__), "03_flash_attention.py")
_spec = importlib.util.spec_from_file_location("fa_fwd", _FWD_PATH)
_fa_fwd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fa_fwd)
triton_flash_attn_with_lse = _fa_fwd.triton_flash_attn_with_lse


# Same Turing SMEM budget pattern as the forward. The backward kernels
# load (Q tile + K tile + V tile + dO tile) into the inner loop and
# also need register room for accumulators. Empirically the largest
# safe tile pair on this GPU is BLOCK_M=BLOCK_N=64; larger tiles
# spill to global memory and lose more than they gain on a 16-SM
# Turing part with 64KB/SM.
TURING_SMEM_BUDGET = 64 * 1024


def _bwd_smem_bytes(BLOCK_M, BLOCK_N, D_HEAD, num_stages, dtype_bytes):
    # The backward inner loop holds more live tiles than the forward
    # (Q, K, V, dO at minimum, plus the running dK/dV/dQ accumulators).
    # Use a more conservative multiplier than the forward's 2x.
    return 4 * BLOCK_N * D_HEAD * dtype_bytes * num_stages


def _bwd_autotune_configs(dtype_bytes: int):
    # Smaller grid than the forward — the backward pass is bigger per
    # iteration and the autotune sweep is correspondingly more
    # expensive in wall-clock. The 9-config grid below covers the
    # interesting region; the SMEM filter prunes the unfit corners.
    configs = []
    skipped = 0
    for BLOCK_M in (32, 64):
        for BLOCK_N in (32, 64):
            for num_warps in (4, 8):
                for num_stages in (2,):
                    smem = _bwd_smem_bytes(
                        BLOCK_M, BLOCK_N, 64, num_stages, dtype_bytes
                    )
                    if smem > TURING_SMEM_BUDGET:
                        skipped += 1
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs, skipped


_FP32_BWD_CONFIGS, _FP32_BWD_SMEM_SKIPPED = _bwd_autotune_configs(dtype_bytes=4)


# ---------------------------------------------------------------------------
# Preprocess kernel: D_i = sum(dO_i * O_i, axis=-1). One fp32 scalar per
# query row. Cheap, bandwidth-bound; no need for autotune.
# ---------------------------------------------------------------------------
@triton.jit
def flash_attn_bwd_preprocess_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_db, stride_dh, stride_dm,
    H, M,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < M

    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    do_base = dO_ptr + pid_b * stride_dob + pid_h * stride_doh
    d_base = D_ptr + pid_b * stride_db + pid_h * stride_dh

    O_blk = o_base + row_offs[:, None] * stride_om + tl.arange(0, D_HEAD)[None, :] * stride_od
    dO_blk = do_base + row_offs[:, None] * stride_dom + tl.arange(0, D_HEAD)[None, :] * stride_dod

    o_tile = tl.load(O_blk, mask=row_mask[:, None], other=0.0)
    do_tile = tl.load(dO_blk, mask=row_mask[:, None], other=0.0)
    d = tl.sum(o_tile * do_tile, axis=1)

    tl.store(d_base + row_offs * stride_dm, d, mask=row_mask)


# ---------------------------------------------------------------------------
# dK / dV kernel. Outer loop over K, V blocks (n direction). Inner loop
# over Q blocks (m direction) accumulates into dK_j, dV_j registers,
# then writes once at the end.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_FP32_BWD_CONFIGS, key=["M", "N", "D_HEAD"])
@triton.jit
def flash_attn_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    H, M, N,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)        # which K, V tile along N
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    n_row_base = pid_n * BLOCK_N
    n_offs = n_row_base + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Per-(B,H) base pointers.
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    do_base = dO_ptr + pid_b * stride_dob + pid_h * stride_doh
    l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh
    d_base = D_ptr + pid_b * stride_db + pid_h * stride_dh
    dk_base = dK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = dV_ptr + pid_b * stride_dvb + pid_h * stride_dvh

    # Load K_j, V_j into registers; they are constant across the Q loop.
    K_blk = k_base + n_offs[:, None] * stride_kn + tl.arange(0, D_HEAD)[None, :] * stride_kd
    V_blk = v_base + n_offs[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
    k = tl.load(K_blk, mask=n_mask[:, None], other=0.0)
    v = tl.load(V_blk, mask=n_mask[:, None], other=0.0)

    # Accumulators in fp32.
    dk_acc = tl.zeros((BLOCK_N, D_HEAD), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D_HEAD), dtype=tl.float32)

    # Causal short-circuit: only Q tiles with q_row >= n_row_base contribute.
    if IS_CAUSAL:
        m_start = (n_row_base // BLOCK_M) * BLOCK_M
    else:
        m_start = 0
    m_block_start = m_start // BLOCK_M
    m_block_end = (M + BLOCK_M - 1) // BLOCK_M

    for m_block in tl.range(m_block_start, m_block_end):
        m_offs = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M

        Q_blk = q_base + m_offs[:, None] * stride_qm + tl.arange(0, D_HEAD)[None, :] * stride_qd
        dO_blk = do_base + m_offs[:, None] * stride_dom + tl.arange(0, D_HEAD)[None, :] * stride_dod
        q = tl.load(Q_blk, mask=m_mask[:, None], other=0.0)
        do = tl.load(dO_blk, mask=m_mask[:, None], other=0.0)

        # S_ij = Q_i @ K_j^T * sm_scale  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # Reconstruct P_ij = exp(S_ij - L_i). Apply causal + bounds mask
        # before the exp to force masked-out entries to zero.
        l_i = tl.load(l_base + m_offs * stride_lm, mask=m_mask, other=0.0)
        if IS_CAUSAL:
            causal_ok = m_offs[:, None] >= n_offs[None, :]
            valid = causal_ok & m_mask[:, None] & n_mask[None, :]
        else:
            valid = m_mask[:, None] & n_mask[None, :]
        p = tl.where(valid, tl.exp(s - l_i[:, None]), 0.0)

        # dV_j += P_ij^T @ dO_i
        dv_acc += tl.dot(tl.trans(p), do)

        # dP_ij = dO_i @ V_j^T   [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v))

        # D_i (precomputed): one fp32 per query row.
        D_i = tl.load(d_base + m_offs * stride_dm, mask=m_mask, other=0.0)

        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - D_i[:, None])
        # Re-apply the mask so dropped-out entries don't pollute the
        # downstream matmuls. (P was already zero on masked entries,
        # but dp - D_i is nonzero, so P * (dp - D_i) is also zero;
        # the explicit mask is belt-and-suspenders for very small fp32
        # underflow accumulating across many tiles.)
        ds = tl.where(valid, ds, 0.0)

        # dK_j += dS_ij^T @ Q_i * sm_scale
        dk_acc += tl.dot(tl.trans(ds), q) * sm_scale

    # Write dK_j, dV_j.
    dK_blk = dk_base + n_offs[:, None] * stride_dkn + tl.arange(0, D_HEAD)[None, :] * stride_dkd
    dV_blk = dv_base + n_offs[:, None] * stride_dvn + tl.arange(0, D_HEAD)[None, :] * stride_dvd
    tl.store(dK_blk, dk_acc, mask=n_mask[:, None])
    tl.store(dV_blk, dv_acc, mask=n_mask[:, None])


# ---------------------------------------------------------------------------
# dQ kernel. Outer loop over Q blocks (m direction), inner over K, V blocks.
# Mirror image of the forward, with the saved L and precomputed D giving
# the stable exp.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_FP32_BWD_CONFIGS, key=["M", "N", "D_HEAD"])
@triton.jit
def flash_attn_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    H, M, N,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    m_row_base = pid_m * BLOCK_M
    m_offs = m_row_base + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    do_base = dO_ptr + pid_b * stride_dob + pid_h * stride_doh
    l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh
    d_base = D_ptr + pid_b * stride_db + pid_h * stride_dh
    dq_base = dQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh

    Q_blk = q_base + m_offs[:, None] * stride_qm + tl.arange(0, D_HEAD)[None, :] * stride_qd
    dO_blk = do_base + m_offs[:, None] * stride_dom + tl.arange(0, D_HEAD)[None, :] * stride_dod
    q = tl.load(Q_blk, mask=m_mask[:, None], other=0.0)
    do = tl.load(dO_blk, mask=m_mask[:, None], other=0.0)
    l_i = tl.load(l_base + m_offs * stride_lm, mask=m_mask, other=0.0)
    D_i = tl.load(d_base + m_offs * stride_dm, mask=m_mask, other=0.0)

    dq_acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)

    if IS_CAUSAL:
        n_end = tl.minimum(N, m_row_base + BLOCK_M)
    else:
        n_end = N
    n_block_end = (n_end + BLOCK_N - 1) // BLOCK_N

    for n_block in tl.range(0, n_block_end):
        n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        K_blk = k_base + n_offs[:, None] * stride_kn + tl.arange(0, D_HEAD)[None, :] * stride_kd
        V_blk = v_base + n_offs[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
        k = tl.load(K_blk, mask=n_mask[:, None], other=0.0)
        v = tl.load(V_blk, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            causal_ok = m_offs[:, None] >= n_offs[None, :]
            valid = causal_ok & m_mask[:, None] & n_mask[None, :]
        else:
            valid = m_mask[:, None] & n_mask[None, :]
        p = tl.where(valid, tl.exp(s - l_i[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - D_i[:, None])
        ds = tl.where(valid, ds, 0.0)

        # dQ_i += dS_ij @ K_j * sm_scale
        dq_acc += tl.dot(ds, k) * sm_scale

    dQ_blk = dq_base + m_offs[:, None] * stride_dqm + tl.arange(0, D_HEAD)[None, :] * stride_dqd
    tl.store(dQ_blk, dq_acc, mask=m_mask[:, None])


# ---------------------------------------------------------------------------
# Public entry point. Given (q, k, v, o, do, lse) returns (dq, dk, dv).
# `o` and `lse` come from `triton_flash_attn_with_lse(q, k, v, ...)`.
# ---------------------------------------------------------------------------
def triton_flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
):
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert o.shape == q.shape and do.shape == q.shape
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    assert lse.shape == (B, H, M) and lse.dtype == torch.float32
    assert D in (16, 32, 64, 128), f"D_HEAD={D} not in supported set"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # D_i = sum(dO * O, axis=-1)
    D_vec = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

    BLOCK_M_PRE = 64  # preprocess kernel block; not autotuned (cheap)
    pre_grid = ((M + BLOCK_M_PRE - 1) // BLOCK_M_PRE, B * H)
    flash_attn_bwd_preprocess_kernel[pre_grid](
        o, do, D_vec,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        D_vec.stride(0), D_vec.stride(1), D_vec.stride(2),
        H, M,
        D_HEAD=D,
        BLOCK_M=BLOCK_M_PRE,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    dkdv_grid = lambda META: ((N + META["BLOCK_N"] - 1) // META["BLOCK_N"], B * H)
    flash_attn_bwd_dkdv_kernel[dkdv_grid](
        q, k, v, do, lse, D_vec,
        dk, dv,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        D_vec.stride(0), D_vec.stride(1), D_vec.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        H, M, N,
        sm_scale,
        IS_CAUSAL=is_causal,
        D_HEAD=D,
    )

    dq_grid = lambda META: ((M + META["BLOCK_M"] - 1) // META["BLOCK_M"], B * H)
    flash_attn_bwd_dq_kernel[dq_grid](
        q, k, v, do, lse, D_vec,
        dq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        D_vec.stride(0), D_vec.stride(1), D_vec.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        H, M, N,
        sm_scale,
        IS_CAUSAL=is_causal,
        D_HEAD=D,
    )

    return dq, dk, dv


# ---------------------------------------------------------------------------
# Correctness: compare against PyTorch autograd.
# ---------------------------------------------------------------------------
def correctness_check() -> bool:
    torch.manual_seed(0)
    cases = [
        # (B, H, M=N, D, is_causal, label)
        (1, 1, 64, 64, False, "small noncausal"),
        (1, 1, 64, 64, True,  "small causal"),
        (1, 4, 128, 32, True, "small D=32 causal"),
        (1, 4, 256, 64, True, "medium causal"),
        (2, 4, 256, 64, True, "medium multi-batch causal"),
        (1, 8, 512, 64, True, "large causal"),
        (1, 4, 1024, 64, True, "M=1024 causal"),
    ]
    all_ok = True
    rows = []
    for B, H, M, D, causal, label in cases:
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        sm_scale = 1.0 / (D ** 0.5)

        # Reference: torch autograd via plain softmax(QK^T*scale) @ V.
        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        v_ref = v.clone().requires_grad_(True)
        s_ref = (q_ref @ k_ref.transpose(-2, -1)) * sm_scale
        if causal:
            mask = torch.tril(torch.ones((M, M), device=s_ref.device, dtype=torch.bool))
            s_ref = s_ref.masked_fill(~mask, float("-inf"))
        p_ref = torch.softmax(s_ref, dim=-1)
        o_ref = p_ref @ v_ref
        do = torch.ones_like(o_ref)
        o_ref.backward(do)

        # Triton forward + backward.
        o_t, lse_t = triton_flash_attn_with_lse(q, k, v, sm_scale=sm_scale, is_causal=causal)
        dq_t, dk_t, dv_t = triton_flash_attn_backward(
            q, k, v, o_t, do, lse_t, sm_scale=sm_scale, is_causal=causal,
        )

        max_err_o  = (o_t  - o_ref.detach()).abs().max().item()
        max_err_dq = (dq_t - q_ref.grad).abs().max().item()
        max_err_dk = (dk_t - k_ref.grad).abs().max().item()
        max_err_dv = (dv_t - v_ref.grad).abs().max().item()

        ok_o  = torch.allclose(o_t,  o_ref.detach(), atol=1e-3, rtol=1e-3)
        ok_dq = torch.allclose(dq_t, q_ref.grad,     atol=1e-3, rtol=1e-3)
        ok_dk = torch.allclose(dk_t, k_ref.grad,     atol=1e-3, rtol=1e-3)
        ok_dv = torch.allclose(dv_t, v_ref.grad,     atol=1e-3, rtol=1e-3)
        ok = ok_o and ok_dq and ok_dk and ok_dv

        flag = "OK" if ok else "FAIL"
        print(f"  B={B} H={H} M={M} D={D} causal={causal!s:5} ({label})")
        print(f"    o={max_err_o:.2e}  dq={max_err_dq:.2e}  "
              f"dk={max_err_dk:.2e}  dv={max_err_dv:.2e}  {flag}")
        rows.append({
            "B": B, "H": H, "M": M, "D": D, "causal": causal, "label": label,
            "max_err_o": max_err_o,
            "max_err_dq": max_err_dq,
            "max_err_dk": max_err_dk,
            "max_err_dv": max_err_dv,
            "ok": ok,
        })
        all_ok = all_ok and ok
    return all_ok, rows


# ---------------------------------------------------------------------------
# Benchmark vs torch.nn.functional.scaled_dot_product_attention + autograd.
# ---------------------------------------------------------------------------
def benchmark(out_path: str | None = None) -> dict:
    torch.manual_seed(0)
    B, H, D = 1, 8, 64
    results = []
    for M in [128, 256, 512, 1024, 2048]:
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        sm_scale = 1.0 / (D ** 0.5)

        # Warm-up — first invocations trigger autotune sweep.
        for _ in range(3):
            o_t, lse_t = triton_flash_attn_with_lse(q, k, v, sm_scale=sm_scale, is_causal=True)
            do = torch.ones_like(o_t)
            _ = triton_flash_attn_backward(q, k, v, o_t, do, lse_t,
                                           sm_scale=sm_scale, is_causal=True)
        torch.cuda.synchronize()

        n_iter = 50 if M <= 1024 else 20

        # Triton backward only (the forward+lse is timed separately via 03).
        o_t, lse_t = triton_flash_attn_with_lse(q, k, v, sm_scale=sm_scale, is_causal=True)
        do = torch.ones_like(o_t)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = triton_flash_attn_backward(q, k, v, o_t, do, lse_t,
                                           sm_scale=sm_scale, is_causal=True)
        torch.cuda.synchronize()
        t_triton_bwd = (time.perf_counter() - t0) / n_iter * 1e6

        # Torch autograd backward through scaled_dot_product_attention.
        q_a = q.clone().requires_grad_(True)
        k_a = k.clone().requires_grad_(True)
        v_a = v.clone().requires_grad_(True)
        # Warm up torch.
        for _ in range(3):
            if q_a.grad is not None:
                q_a.grad = None
                k_a.grad = None
                v_a.grad = None
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_a, k_a, v_a, is_causal=True,
            )
            o_ref.sum().backward()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            q_a.grad = None
            k_a.grad = None
            v_a.grad = None
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_a, k_a, v_a, is_causal=True,
            )
            o_ref.sum().backward()
        torch.cuda.synchronize()
        t_torch_full = (time.perf_counter() - t0) / n_iter * 1e6

        # ----- apples-to-apples torch backward-only timing (v0.4.3) -----
        # torch.autograd.grad with retain_graph=True lets us time the
        # backward in isolation: do ONE forward (untimed), then issue
        # repeated backward calls against the saved graph. This is the
        # SIMT-vs-SIMT comparison the prior `torch_fwd_bwd_us` column
        # could not give (no public bwd-only entry point in SDPA). The
        # forward is computed once outside the timing loop; only the
        # `autograd.grad` call is timed.
        q_b = q.clone().requires_grad_(True)
        k_b = k.clone().requires_grad_(True)
        v_b = v.clone().requires_grad_(True)
        o_ref_b = torch.nn.functional.scaled_dot_product_attention(
            q_b, k_b, v_b, is_causal=True,
        )
        grad_out = torch.ones_like(o_ref_b)
        # Warm up torch's backward-only path against the saved graph.
        for _ in range(3):
            torch.autograd.grad(
                outputs=o_ref_b, inputs=[q_b, k_b, v_b],
                grad_outputs=grad_out, retain_graph=True,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            torch.autograd.grad(
                outputs=o_ref_b, inputs=[q_b, k_b, v_b],
                grad_outputs=grad_out, retain_graph=True,
            )
        torch.cuda.synchronize()
        t_torch_bwd_only = (time.perf_counter() - t0) / n_iter * 1e6

        # Backward FLOPs: ~5 N×M×D matmuls in causal backward (Q@K^T,
        # P^T@dO, dO@V^T, dS^T@Q, dS@K). At 2 flops/multiply-add each,
        # that's 10*B*H*M*M*D total. Halve for the causal mask.
        flops = 10 * B * H * M * M * D * 0.5
        tflops_triton = flops / (t_triton_bwd * 1e-6) / 1e12

        results.append({
            "B": B, "H": H, "M": M, "D": D,
            "triton_bwd_us": round(t_triton_bwd, 2),
            "torch_fwd_bwd_us": round(t_torch_full, 2),
            # v0.4.3: apples-to-apples bwd-vs-bwd column. Uses
            # autograd.grad(retain_graph=True) to time the backward
            # in isolation after a single untimed forward pass.
            "torch_bwd_only_us": round(t_torch_bwd_only, 2),
            "triton_bwd_tflops": round(tflops_triton, 3),
            # Honest reading: torch_fwd_bwd_us includes the forward,
            # ours doesn't. torch_bwd_only_us is apples-to-apples.
            "torch_full_vs_triton_bwd_ratio": round(t_torch_full / t_triton_bwd, 3),
            "torch_bwd_only_vs_triton_bwd_ratio": round(t_torch_bwd_only / t_triton_bwd, 3),
        })
        print(f"  B={B} H={H} M={M:5d} D={D}  triton_bwd={t_triton_bwd:9.2f}us  "
              f"torch_fwd+bwd={t_torch_full:9.2f}us  "
              f"torch_bwd_only={t_torch_bwd_only:9.2f}us  "
              f"triton_bwd TFLOPS={tflops_triton:.3f}")
    out = {
        "kernel": "flash-attention-v2-backward-causal-fp32",
        "autotuned": True,
        "smem_pruned_configs": _FP32_BWD_SMEM_SKIPPED,
        "live_configs": len(_FP32_BWD_CONFIGS),
        "note": (
            "Triton timing is the backward-only call. "
            "`torch_fwd_bwd_us` times the full forward+backward through "
            "scaled_dot_product_attention + .backward() (kept from "
            "v0.4.0 for back-compat). `torch_bwd_only_us` (v0.4.3) "
            "times torch's backward in isolation via "
            "torch.autograd.grad(retain_graph=True) against a "
            "pre-computed forward graph — apples-to-apples bwd-vs-bwd. "
            "Both columns are SIMT-vs-SIMT on sm_75 per v0.4.2 PTX "
            "diagnosis; the SIMT-path perf bar has not moved."
        ),
        "results": results,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if not args.correctness and not args.bench:
        args.correctness = True
        args.bench = True
    if args.correctness:
        print("=== correctness (forward + backward vs torch autograd) ===")
        ok, _rows = correctness_check()
        if not ok:
            print("FAIL")
            raise SystemExit(1)
        print("OK")
        print()
    if args.bench:
        print(f"=== benchmark backward (B=1, H=8, D=64, causal=True, autotuned)  "
              f"[{len(_FP32_BWD_CONFIGS)} live configs, "
              f"{_FP32_BWD_SMEM_SKIPPED} SMEM-pruned] ===")
        benchmark(out_path=args.out)


if __name__ == "__main__":
    main()
