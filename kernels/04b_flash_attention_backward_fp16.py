"""
04b — Flash-Attention v2 BACKWARD pass (fp16, causal optional) on Triton.

fp16 input variant of `04_flash_attention_backward.py`. Q, K, V, O, dO
and the output dQ, dK, dV are fp16. The saved log-sum-exp (`L`) and
the row-statistic (`D_i = sum(dO * O, axis=-1)`) stay in fp32; the
running dQ, dK, dV accumulators inside each kernel stay in fp32 until
the final cast-and-store.

Algorithm (Dao 2023 Section 3, same as the fp32 backward):

  Preprocess:
    D_i = sum(dO_i * O_i, axis=-1)                  # fp32 [B, H, M]

  dK / dV kernel (outer loop over K, V blocks along N):
    For each Q block i:
      S_ij  = Q_i @ K_j^T * sm_scale                # fp16 inputs, fp32 acc
      P_ij  = exp(S_ij - L_i)                       # fp32 from saved L
      dV_j += P_ij^T @ dO_i                         # P cast to fp16 in
      dP_ij = dO_i @ V_j^T                          # fp16 inputs, fp32 acc
      dS_ij = P_ij * (dP_ij - D_i)
      dK_j += dS_ij^T @ Q_i * sm_scale              # dS cast to fp16 in

  dQ kernel (outer loop over Q blocks along M):
    For each K, V block j:
      S_ij  = Q_i @ K_j^T * sm_scale
      P_ij  = exp(S_ij - L_i)
      dP_ij = dO_i @ V_j^T
      dS_ij = P_ij * (dP_ij - D_i)
      dQ_i += dS_ij @ K_j * sm_scale

Five matmul steps total across the two kernels:

  1. Q @ K^T     (per inner iteration; both kernels recompute S)
  2. P^T @ dO    (dK/dV kernel; accumulates dV_j across the Q loop)
  3. dO @ V^T    (per inner iteration; both kernels recompute dP)
  4. dS^T @ Q    (dK/dV kernel; accumulates dK_j across the Q loop)
  5. dS @ K      (dQ kernel; accumulates dQ_i across the K, V loop)

v0.4.1 doc claimed all five route through Turing m16n8k16 fp16 mma
via `tl.dot`. **v0.4.2 PTX inspection (`docs/PTX_INSPECTION.md`)
disproved that claim on two counts:**

  1. m16n8k16 with `.f32.f16.f16.f32` requires sm_80+ (Ampere). On
     sm_75 (Turing) the fp16 mma flavor is m16n8k8.
  2. Triton 3.7.0's `TritonGPUAccelerateMatmul` pass declines the
     MMA layout promotion on sm_75 for these kernels, leaving the
     dot in a `#blocked` layout that lowers to SIMT `fma.rn.f32` —
     no tensor cores engaged. Identical fma counts in v0.4.1 fp16
     PTX vs v0.4.0 fp32 PTX (2072 in dK/dV, 1560 in dQ).

The fp16 path still saves bandwidth (halved global I/O on Q, K, V,
dO, dQ, dK, dV), which is why the v0.4.1 vs v0.4.0 perf ratio closes
at large M, but it pays cast overhead (~128 cvt.* per dK/dV
iteration) without tensor-core compute speedup. Closing the gap to
torch via tensor cores requires a Triton backend that emits
`mma.sync.aligned.m16n8k8` on sm_75 — out of scope for v0.4.2.

Numerical-stability concern (preserved from v0.4.0):
- P recomputation under fp16 input multiplies (Q@K^T at fp16 inputs)
  introduces ~1e-3 round-off into S_ij before the `exp`. After the
  exp this becomes a multiplicative ~1e-3 perturbation on P. That
  perturbation propagates through five matmul steps; the dV
  reduction chain is the longest (P^T @ dO summed over every Q tile
  per K tile), so dV is where the worst-case error lands.
- LSE stays fp32 (forward already wrote it that way). We use it
  directly; no precision is lost in the exp's stabilizer.
- dS = P * (dP - D_i): both factors are fp32 in-register; the cast
  to fp16 happens *only* at the matmul input, so the subtraction
  `dP - D_i` does not lose precision to fp16 round-off.
- Accumulators dV, dK, dQ stay in fp32 across the entire reduction;
  only the final write to global memory casts to fp16.
- v0.4.0 fp32 backward hit `1.91e-06` worst-case dV at M=512 causal;
  we expect fp16 to land ~3 orders of magnitude higher, comfortably
  inside the `atol=1e-2` bar that the fp16 forward already operates
  under. (Empirical results below.)

Scope:
- fp16 only. fp32 backward lives in `04_flash_attention_backward.py`.
- Causal mask supported. Same triangular pattern as the forward.
- Multi-batch + multi-head. Each (B, H) is a grid slot.
- Variable-length sequences (attention masks beyond strict causal)
  NOT supported. → v0.5.
- Dropout NOT in backward. → v0.5+.

Hardware: GTX 1650 Max-Q (Turing, compute capability 7.5).
"""

from __future__ import annotations
import argparse
import json
import time
import torch
import triton
import triton.language as tl

# Forward + LSE save lives in 03b_flash_attention_fp16.py. Import the
# public entry point so the backward harness can drive the full
# forward + backward round-trip from one script.
import importlib.util
import os
_FWD_PATH = os.path.join(os.path.dirname(__file__), "03b_flash_attention_fp16.py")
_spec = importlib.util.spec_from_file_location("fa_fwd_fp16", _FWD_PATH)
_fa_fwd_fp16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fa_fwd_fp16)
triton_flash_attn_fp16 = _fa_fwd_fp16.triton_flash_attn_fp16


# Same Turing SMEM budget as the fp32 backward. fp16 halves the
# per-element SMEM cost so more aggressive tile sizes survive the
# pre-filter; we keep the 4× multiplier from the fp32 backward because
# the backward holds more live tiles than the forward.
TURING_SMEM_BUDGET = 64 * 1024


def _bwd_smem_bytes(BLOCK_M, BLOCK_N, D_HEAD, num_stages, dtype_bytes):
    # Mirror the fp32 backward's SMEM estimator. The 4× multiplier
    # accounts for Q, K, V, dO live in the inner loop plus the
    # dK/dV/dQ accumulators in registers. fp16 dtype_bytes=2 halves
    # the per-element footprint vs fp32.
    return 4 * BLOCK_N * D_HEAD * dtype_bytes * num_stages


def _bwd_autotune_configs_fp16():
    # Mirror the v0.4.0 fp32 backward's grid exactly:
    # (BLOCK_M, BLOCK_N) ∈ {32, 64}² × num_warps ∈ {4, 8} × num_stages=2.
    # The wider grid that also probed num_stages=3 was tried in an
    # earlier draft but the autotune wall-clock cost (12 live configs
    # × 2 backward kernels × 7 correctness shapes × autotune-on-first-
    # call) blew the harness budget; the previous agent timed out
    # during the sweep. Keep parity with v0.4.0 — winners on Turing
    # cluster at BLOCK_N ∈ {32, 64} and num_warps ∈ {4, 8} regardless.
    # If perf later flags as the bottleneck, widen in a follow-on and
    # record the new winners in CHANGELOG.
    configs = []
    skipped = 0
    for BLOCK_M in (32, 64):
        for BLOCK_N in (32, 64):
            for num_warps in (4, 8):
                for num_stages in (2,):
                    smem = _bwd_smem_bytes(
                        BLOCK_M, BLOCK_N, 64, num_stages, dtype_bytes=2
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


_FP16_BWD_CONFIGS, _FP16_BWD_SMEM_SKIPPED = _bwd_autotune_configs_fp16()


# ---------------------------------------------------------------------------
# Preprocess kernel: D_i = sum(dO_i * O_i, axis=-1). One fp32 scalar per
# query row. dO and O loaded as fp16, multiplied in fp32, summed in fp32.
# Bandwidth-bound; no autotune.
# ---------------------------------------------------------------------------
@triton.jit
def flash_attn_bwd_preprocess_kernel_fp16(
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

    # Load fp16, cast to fp32 for the reduction.
    o_tile = tl.load(O_blk, mask=row_mask[:, None], other=0.0).to(tl.float32)
    do_tile = tl.load(dO_blk, mask=row_mask[:, None], other=0.0).to(tl.float32)
    d = tl.sum(o_tile * do_tile, axis=1)

    tl.store(d_base + row_offs * stride_dm, d, mask=row_mask)


# ---------------------------------------------------------------------------
# dK / dV kernel (fp16). Outer loop over K, V tiles along N. Inner loop
# over Q tiles along M accumulates fp32 dK_j, dV_j; cast to fp16 on store.
# All five matmul steps in the backward use tl.dot with fp16 inputs and
# fp32 accumulator. On sm_75 + Triton 3.7.0 these lower to SIMT fma (no
# tensor cores) — see docs/PTX_INSPECTION.md for the v0.4.2 diagnosis.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_FP16_BWD_CONFIGS, key=["M", "N", "D_HEAD"])
@triton.jit
def flash_attn_bwd_dkdv_kernel_fp16(
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
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    n_row_base = pid_n * BLOCK_N
    n_offs = n_row_base + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    do_base = dO_ptr + pid_b * stride_dob + pid_h * stride_doh
    l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh
    d_base = D_ptr + pid_b * stride_db + pid_h * stride_dh
    dk_base = dK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = dV_ptr + pid_b * stride_dvb + pid_h * stride_dvh

    # K_j, V_j stay fp16 in registers across the Q loop (used as
    # tl.dot inputs, no precision lost vs reload).
    K_blk = k_base + n_offs[:, None] * stride_kn + tl.arange(0, D_HEAD)[None, :] * stride_kd
    V_blk = v_base + n_offs[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
    k = tl.load(K_blk, mask=n_mask[:, None], other=0.0)
    v = tl.load(V_blk, mask=n_mask[:, None], other=0.0)

    # fp32 accumulators across the entire Q loop.
    dk_acc = tl.zeros((BLOCK_N, D_HEAD), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D_HEAD), dtype=tl.float32)

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
        q = tl.load(Q_blk, mask=m_mask[:, None], other=0.0)        # fp16
        do = tl.load(dO_blk, mask=m_mask[:, None], other=0.0)      # fp16

        # S_ij = Q_i @ K_j^T * sm_scale; fp16 inputs, fp32 acc.
        # On sm_75 + Triton 3.7.0 this lowers to SIMT fma.rn.f32 —
        # not Turing mma.sync.aligned.m16n8k8 (see docs/PTX_INSPECTION.md).
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale  # fp32

        # Reconstruct P_ij = exp(S_ij - L_i). LSE was saved in fp32
        # by the forward; use it directly.
        l_i = tl.load(l_base + m_offs * stride_lm, mask=m_mask, other=0.0)  # fp32
        if IS_CAUSAL:
            causal_ok = m_offs[:, None] >= n_offs[None, :]
            valid = causal_ok & m_mask[:, None] & n_mask[None, :]
        else:
            valid = m_mask[:, None] & n_mask[None, :]
        p = tl.where(valid, tl.exp(s - l_i[:, None]), 0.0)            # fp32

        # dV_j += P_ij^T @ dO_i. Cast p to fp16 at the matmul input
        # so the dot uses tensor cores; accumulate in fp32.
        p_fp16 = p.to(tl.float16)
        dv_acc += tl.dot(tl.trans(p_fp16), do, out_dtype=tl.float32)

        # dP_ij = dO_i @ V_j^T. fp16 inputs, fp32 acc.
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)            # fp32

        # D_i precomputed (fp32).
        D_i = tl.load(d_base + m_offs * stride_dm, mask=m_mask, other=0.0)

        # dS_ij = P_ij * (dP_ij - D_i). All fp32 in-register; cast to
        # fp16 only for the matmul inputs below.
        ds = p * (dp - D_i[:, None])
        # Re-apply the validity mask (mirrors the fp32 backward).
        ds = tl.where(valid, ds, 0.0)
        ds_fp16 = ds.to(tl.float16)

        # dK_j += dS_ij^T @ Q_i * sm_scale.
        dk_acc += tl.dot(tl.trans(ds_fp16), q, out_dtype=tl.float32) * sm_scale

    # Store dK_j, dV_j as fp16.
    dK_blk = dk_base + n_offs[:, None] * stride_dkn + tl.arange(0, D_HEAD)[None, :] * stride_dkd
    dV_blk = dv_base + n_offs[:, None] * stride_dvn + tl.arange(0, D_HEAD)[None, :] * stride_dvd
    tl.store(dK_blk, dk_acc.to(tl.float16), mask=n_mask[:, None])
    tl.store(dV_blk, dv_acc.to(tl.float16), mask=n_mask[:, None])


# ---------------------------------------------------------------------------
# dQ kernel (fp16). Outer loop over Q tiles, inner over K, V tiles.
# Mirror of the forward iteration pattern. Two matmuls per inner step
# (Q @ K^T and dO @ V^T) plus the dS @ K reduction.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_FP16_BWD_CONFIGS, key=["M", "N", "D_HEAD"])
@triton.jit
def flash_attn_bwd_dq_kernel_fp16(
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
    q = tl.load(Q_blk, mask=m_mask[:, None], other=0.0)               # fp16
    do = tl.load(dO_blk, mask=m_mask[:, None], other=0.0)             # fp16
    l_i = tl.load(l_base + m_offs * stride_lm, mask=m_mask, other=0.0)  # fp32
    D_i = tl.load(d_base + m_offs * stride_dm, mask=m_mask, other=0.0)  # fp32

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

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        if IS_CAUSAL:
            causal_ok = m_offs[:, None] >= n_offs[None, :]
            valid = causal_ok & m_mask[:, None] & n_mask[None, :]
        else:
            valid = m_mask[:, None] & n_mask[None, :]
        p = tl.where(valid, tl.exp(s - l_i[:, None]), 0.0)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - D_i[:, None])
        ds = tl.where(valid, ds, 0.0)
        ds_fp16 = ds.to(tl.float16)

        # dQ_i += dS_ij @ K_j * sm_scale.
        dq_acc += tl.dot(ds_fp16, k, out_dtype=tl.float32) * sm_scale

    dQ_blk = dq_base + m_offs[:, None] * stride_dqm + tl.arange(0, D_HEAD)[None, :] * stride_dqd
    tl.store(dQ_blk, dq_acc.to(tl.float16), mask=m_mask[:, None])


# ---------------------------------------------------------------------------
# Public entry point. Given (q, k, v, o, do, lse) all fp16 except lse
# (fp32), returns (dq, dk, dv) all fp16.
# `o` and `lse` come from `triton_flash_attn_fp16(q, k, v, save_lse=True)`.
# ---------------------------------------------------------------------------
def triton_flash_attn_backward_fp16(
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
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16, \
        "expected fp16 q, k, v"
    assert o.dtype == torch.float16 and do.dtype == torch.float16, \
        "expected fp16 o, do"
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    assert lse.shape == (B, H, M) and lse.dtype == torch.float32, \
        "expected fp32 lse [B, H, M] from forward with save_lse=True"
    assert D in (16, 32, 64, 128), f"D_HEAD={D} not in supported set"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    # D_i = sum(dO * O, axis=-1) (fp32).
    D_vec = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

    BLOCK_M_PRE = 64
    pre_grid = ((M + BLOCK_M_PRE - 1) // BLOCK_M_PRE, B * H)
    flash_attn_bwd_preprocess_kernel_fp16[pre_grid](
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
    flash_attn_bwd_dkdv_kernel_fp16[dkdv_grid](
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
    flash_attn_bwd_dq_kernel_fp16[dq_grid](
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
# Correctness vs PyTorch autograd (fp16 reference). Mirrors the v0.4.0
# 7-shape suite with atol=1e-2 / rtol=1e-2 to match the fp16 forward.
# ---------------------------------------------------------------------------
def correctness_check():
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
        q = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        k = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        v = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        sm_scale = 1.0 / (D ** 0.5)

        # fp16 reference via torch autograd: plain softmax(QK^T*scale)@V
        # path. We use the same algebraic form as the v0.4.0 fp32
        # backward harness; torch's SDPA backward also has its own
        # internal precision choices, but the plain-softmax path keeps
        # the reference structurally close to what the Triton kernel
        # is implementing.
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

        # Triton fp16 forward + backward.
        o_t, lse_t = triton_flash_attn_fp16(
            q, k, v, sm_scale=sm_scale, is_causal=causal, save_lse=True,
        )
        dq_t, dk_t, dv_t = triton_flash_attn_backward_fp16(
            q, k, v, o_t, do, lse_t, sm_scale=sm_scale, is_causal=causal,
        )

        # Promote to fp32 for the error comparison so the tolerance
        # arithmetic itself doesn't lose precision.
        max_err_o  = (o_t.float()  - o_ref.detach().float()).abs().max().item()
        max_err_dq = (dq_t.float() - q_ref.grad.float()).abs().max().item()
        max_err_dk = (dk_t.float() - k_ref.grad.float()).abs().max().item()
        max_err_dv = (dv_t.float() - v_ref.grad.float()).abs().max().item()

        atol, rtol = 1e-2, 1e-2
        ok_o  = torch.allclose(o_t,  o_ref.detach(), atol=atol, rtol=rtol)
        ok_dq = torch.allclose(dq_t, q_ref.grad,     atol=atol, rtol=rtol)
        ok_dk = torch.allclose(dk_t, k_ref.grad,     atol=atol, rtol=rtol)
        ok_dv = torch.allclose(dv_t, v_ref.grad,     atol=atol, rtol=rtol)
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
# Benchmark vs torch.nn.functional.scaled_dot_product_attention + autograd
# (fp16). Mirrors the v0.4.0 fp32 backward harness structure exactly.
#
# Honest non-claim, preserved from v0.4.0: torch's autograd path does
# not expose a backward-only entry comparable to ours, so torch's
# column times the full forward+backward through `.backward()` while
# Triton's column is the backward-only call (forward already done).
# The ratio column is informative but is fwd+bwd vs bwd, not bwd vs
# bwd. True backward-vs-backward gap at large M is *larger* than the
# table shows.
# ---------------------------------------------------------------------------
def _winning_config_for(kernel, M: int, N: int, D: int) -> dict:
    cache = kernel.cache
    if not cache:
        return {}
    for key, cfg in cache.items():
        if isinstance(key, tuple) and len(key) >= 3:
            if key[0] == M and key[1] == N and key[2] == D:
                return {
                    "BLOCK_M": cfg.kwargs.get("BLOCK_M"),
                    "BLOCK_N": cfg.kwargs.get("BLOCK_N"),
                    "num_warps": cfg.num_warps,
                    "num_stages": cfg.num_stages,
                }
    return {}


def benchmark(out_path: str | None = None) -> dict:
    torch.manual_seed(0)
    B, H, D = 1, 8, 64
    results = []
    for M in [128, 256, 512, 1024, 2048]:
        q = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        k = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        v = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        sm_scale = 1.0 / (D ** 0.5)

        # Warm-up — first invocations trigger the autotune sweep on
        # both backward kernels.
        for _ in range(3):
            o_t, lse_t = triton_flash_attn_fp16(
                q, k, v, sm_scale=sm_scale, is_causal=True, save_lse=True,
            )
            do = torch.ones_like(o_t)
            _ = triton_flash_attn_backward_fp16(
                q, k, v, o_t, do, lse_t, sm_scale=sm_scale, is_causal=True,
            )
        torch.cuda.synchronize()

        n_iter = 50 if M <= 1024 else 20

        # Triton backward only (the forward+lse is timed separately
        # via 03b_flash_attention_fp16.py).
        o_t, lse_t = triton_flash_attn_fp16(
            q, k, v, sm_scale=sm_scale, is_causal=True, save_lse=True,
        )
        do = torch.ones_like(o_t)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = triton_flash_attn_backward_fp16(
                q, k, v, o_t, do, lse_t, sm_scale=sm_scale, is_causal=True,
            )
        torch.cuda.synchronize()
        t_triton_bwd = (time.perf_counter() - t0) / n_iter * 1e6

        # Torch autograd backward through scaled_dot_product_attention
        # (fp16; same path the fp16 forward benchmark uses as
        # reference).
        q_a = q.clone().requires_grad_(True)
        k_a = k.clone().requires_grad_(True)
        v_a = v.clone().requires_grad_(True)
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

        # Backward FLOPs: 5 N×M×D matmuls in the causal backward,
        # halved by the causal mask. Same accounting as v0.4.0.
        flops = 10 * B * H * M * M * D * 0.5
        tflops_triton = flops / (t_triton_bwd * 1e-6) / 1e12

        dkdv_cfg = _winning_config_for(flash_attn_bwd_dkdv_kernel_fp16, M, M, D)
        dq_cfg = _winning_config_for(flash_attn_bwd_dq_kernel_fp16, M, M, D)
        results.append({
            "B": B, "H": H, "M": M, "D": D,
            "triton_bwd_us": round(t_triton_bwd, 2),
            "torch_fwd_bwd_us": round(t_torch_full, 2),
            "triton_bwd_tflops": round(tflops_triton, 3),
            "torch_full_vs_triton_bwd_ratio": round(t_torch_full / t_triton_bwd, 3),
            "autotune_dkdv": dkdv_cfg,
            "autotune_dq": dq_cfg,
        })
        print(f"  B={B} H={H} M={M:5d} D={D}  triton_bwd={t_triton_bwd:9.2f}us  "
              f"torch_fwd+bwd={t_torch_full:9.2f}us  "
              f"triton_bwd TFLOPS={tflops_triton:.3f}  "
              f"dkdv={dkdv_cfg}  dq={dq_cfg}")
    out = {
        "kernel": "flash-attention-v2-backward-causal-fp16",
        "dtype": "float16",
        "autotuned": True,
        "smem_pruned_configs": _FP16_BWD_SMEM_SKIPPED,
        "live_configs": len(_FP16_BWD_CONFIGS),
        "note": (
            "Triton timing is the backward-only call. Torch timing is "
            "the full forward+backward pass through "
            "scaled_dot_product_attention + .backward(); torch's "
            "autograd-backward path does not expose a backward-only "
            "entry point comparable to ours. The ratio column is "
            "informative but is fwd+bwd vs bwd, not bwd vs bwd."
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
        print("=== correctness fp16 (forward + backward vs torch autograd) ===")
        ok, _rows = correctness_check()
        if not ok:
            print("FAIL")
            raise SystemExit(1)
        print("OK")
        print()
    if args.bench:
        print(f"=== benchmark fp16 backward (B=1, H=8, D=64, causal=True, autotuned)  "
              f"[{len(_FP16_BWD_CONFIGS)} live configs, "
              f"{_FP16_BWD_SMEM_SKIPPED} SMEM-pruned] ===")
        benchmark(out_path=args.out)


if __name__ == "__main__":
    main()
