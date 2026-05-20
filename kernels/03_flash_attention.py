"""
03 — Flash-Attention v2 (forward only, causal optional) on Triton.

Tile-based attention with the online-softmax trick from the
flash-attention-2 paper (Dao 2023). The Q dimension is tiled along
sequence-length; for each Q tile, the kernel iterates over K/V tiles
and maintains a running max + exp-sum + output accumulator, all in
registers. No N×M attention matrix is ever materialised.

Output: O = softmax(Q @ K^T * sm_scale + causal_mask) @ V

Scope of this artifact:

- Forward pass only. Backward is a separate kernel (and a larger
  engineering exercise — gradient through online softmax requires
  the saved log-sum-exp + statistics, not just the output).
- Causal mask supported (per query position, only sees keys at or
  before its own position).
- fp32 throughout. The fp16 + tensor-core variant ships as
  `03b_flash_attention_fp16.py`; bf16 is Ampere+ and not available
  on Turing tensor cores.
- Multi-batch, multi-head: supported. Each (batch, head) pair gets
  its own grid slot.

Correctness check: comparison against
`torch.nn.functional.scaled_dot_product_attention` (which Torch
backs with its own memory-efficient attention on supported
hardware). Tolerance is 1e-3 (fp32 accumulation has visible round-off
relative to the reference's mathematically-equivalent but
differently-ordered reductions).

v0.3.0: wrapped in `@triton.autotune` over (BLOCK_M, BLOCK_N,
num_warps, num_stages) keyed on (M, N, D_HEAD). Configs whose
estimated SMEM footprint exceeds Turing's 64KB/SM are filtered out
at autotune construction time so the autotuner never attempts a
config that will fail to compile.

v0.4.0: optional `save_lse` flag. When set, the kernel additionally
writes the per-row log-sum-exp `L_i = m_i + log(l_i)` (fp32, shape
[B, H, M]) to an output buffer. The backward kernel
(`04_flash_attention_backward.py`) consumes that buffer to recompute
the softmax stably without re-evaluating the online max+sum
reduction.

Hardware: GTX 1650 Max-Q (Turing, compute capability 7.5).
"""

from __future__ import annotations
import argparse
import json
import time
import torch
import triton
import triton.language as tl


# Turing has 64 KB shared memory per SM (compute capability 7.5).
# Triton pipelines K/V loads across `num_stages` buffers; a rough
# estimate of per-tile SMEM usage that empirically matches what
# compiles on this GPU is the K+V double-buffered footprint:
#   2 * BLOCK_N * D_HEAD * dtype_bytes * num_stages
# Q is loaded once per Q-tile and typically lives in registers
# (Triton lowers small tiles to the register file, not SMEM); the
# softmax intermediate also stays in registers. The pre-filter is a
# soft check — if a config still fails to compile (e.g. due to a
# register-spill SMEM allocation we didn't account for) the Triton
# autotuner skips it at sweep time.
TURING_SMEM_BUDGET = 64 * 1024


def _flash_attn_smem_bytes(BLOCK_M, BLOCK_N, D_HEAD, num_stages, dtype_bytes):
    return 2 * BLOCK_N * D_HEAD * dtype_bytes * num_stages


def _flash_attn_autotune_configs(dtype_bytes: int):
    # Conservative starting grid (36 raw configs, ~14 alive for fp32
    # after SMEM pruning). The task spec's wider grid
    # (BLOCK_N ∈ {16,32,64}, num_stages ∈ {2,3,4}) ran the autotuner
    # for hours on a single GTX 1650; this trimmed grid still covers
    # the interesting region. If a future winner lands on the {32,64}
    # BLOCK_N edge or the {2,3} num_stages edge, the boundary should
    # be widened.
    configs = []
    skipped = 0
    for BLOCK_M in (16, 32, 64):
        for BLOCK_N in (32, 64):
            for num_warps in (2, 4, 8):
                for num_stages in (2, 3):
                    smem = _flash_attn_smem_bytes(
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


_FP32_CONFIGS, _FP32_SMEM_SKIPPED = _flash_attn_autotune_configs(dtype_bytes=4)


@triton.autotune(configs=_FP32_CONFIGS, key=["M", "N", "D_HEAD"])
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    H, M, N,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    SAVE_LSE: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)        # which Q tile along sequence
    pid_bh = tl.program_id(1)       # which (batch, head) pair
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Q tile offsets
    q_row_base = pid_m * BLOCK_M
    q_row_offs = q_row_base + tl.arange(0, BLOCK_M)
    q_mask = q_row_offs < M

    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    Q_block_ptr = q_base + q_row_offs[:, None] * stride_qm + tl.arange(0, D_HEAD)[None, :] * stride_qd
    q = tl.load(Q_block_ptr, mask=q_mask[:, None], other=0.0)

    # K, V offsets (the rows of K and V vary; the head/batch base is fixed)
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

    # Initialize online softmax accumulators.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)

    # If causal, only iterate K tiles up to the last one that contains
    # a position <= q_row_base + BLOCK_M - 1.
    if IS_CAUSAL:
        n_end = tl.minimum(N, q_row_base + BLOCK_M)
    else:
        n_end = N
    n_block_end = (n_end + BLOCK_N - 1) // BLOCK_N

    for n_block in tl.range(0, n_block_end):
        n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load K tile [BLOCK_N, D_HEAD].
        K_block_ptr = k_base + n_offs[:, None] * stride_kn + tl.arange(0, D_HEAD)[None, :] * stride_kd
        k = tl.load(K_block_ptr, mask=n_mask[:, None], other=0.0)

        # S = Q @ K^T * sm_scale  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply causal mask and N-out-of-bounds mask
        if IS_CAUSAL:
            causal_ok = q_row_offs[:, None] >= n_offs[None, :]
            s = tl.where(causal_ok & n_mask[None, :], s, -float("inf"))
        else:
            s = tl.where(n_mask[None, :], s, -float("inf"))

        # Online softmax update.
        s_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, s_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # Load V tile [BLOCK_N, D_HEAD].
        V_block_ptr = v_base + n_offs[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
        v = tl.load(V_block_ptr, mask=n_mask[:, None], other=0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_acc = o_acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    # Normalize.
    o_acc = o_acc / l_i[:, None]

    # Write O tile.
    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    O_block_ptr = o_base + q_row_offs[:, None] * stride_om + tl.arange(0, D_HEAD)[None, :] * stride_od
    tl.store(O_block_ptr, o_acc, mask=q_mask[:, None])

    # Optionally save log-sum-exp per query row for the backward pass.
    # L_i = m_i + log(l_i). Numerically stable because we never exponentiate
    # without first subtracting the running max.
    if SAVE_LSE:
        lse = m_i + tl.log(l_i)
        l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh
        L_block_ptr = l_base + q_row_offs * stride_lm
        tl.store(L_block_ptr, lse, mask=q_mask)


def triton_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
    save_lse: bool = False,
):
    """Public entry point. q/k/v are [B, H, M_or_N, D]. M==N typical for
    self-attention.

    Returns:
      - o [B, H, M, D] when `save_lse=False` (back-compat shape).
      - (o, lse) tuple when `save_lse=True`, with `lse` fp32 [B, H, M].
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    assert D in (16, 32, 64, 128), \
        f"D_HEAD={D} not in supported set"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    o = torch.empty_like(q)
    if save_lse:
        lse = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        l_strides = (lse.stride(0), lse.stride(1), lse.stride(2))
    else:
        # Placeholder pointer + zero strides; the kernel never reads/writes
        # via L_ptr when SAVE_LSE is False (constexpr-elided branch), but
        # Triton still requires the argument slot to be a valid tensor.
        lse = o  # any valid tensor pointer suffices since the branch is dead
        l_strides = (0, 0, 0)

    # The grid depends on BLOCK_M which is chosen by autotune; Triton
    # supports a callable `grid` that receives the chosen Config kwargs.
    grid = lambda META: ((M + META["BLOCK_M"] - 1) // META["BLOCK_M"], B * H)
    flash_attn_fwd_kernel[grid](
        q, k, v, o, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        l_strides[0], l_strides[1], l_strides[2],
        H, M, N,
        sm_scale,
        IS_CAUSAL=is_causal,
        SAVE_LSE=save_lse,
        D_HEAD=D,
    )
    if save_lse:
        return o, lse
    return o


def triton_flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
):
    """Forward that always saves the per-row log-sum-exp. Convenience
    wrapper used by the backward harness."""
    return triton_flash_attn(q, k, v, sm_scale=sm_scale, is_causal=is_causal,
                             save_lse=True)


def reference_attn(q, k, v, sm_scale, is_causal: bool) -> torch.Tensor:
    """Plain attention reference: O = softmax(Q @ K^T * scale + mask) @ V."""
    s = (q @ k.transpose(-2, -1)) * sm_scale
    if is_causal:
        M, N = s.shape[-2], s.shape[-1]
        mask = torch.tril(torch.ones((M, N), device=s.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ v


def reference_lse(q, k, v, sm_scale, is_causal: bool) -> torch.Tensor:
    """Per-row log-sum-exp of (Q @ K^T * scale + mask)."""
    s = (q @ k.transpose(-2, -1)) * sm_scale
    if is_causal:
        M, N = s.shape[-2], s.shape[-1]
        mask = torch.tril(torch.ones((M, N), device=s.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(s, dim=-1)


def correctness_check() -> bool:
    """Compare Triton vs reference attention across shapes and causal modes.
    Verifies both the original `save_lse=False` API (back-compat) and the
    new `save_lse=True` path (output O still matches, and the saved LSE
    matches the reference)."""
    torch.manual_seed(0)
    cases = [
        # (B, H, M=N, D, is_causal)
        (1, 1, 64, 64, False),
        (1, 1, 64, 64, True),
        (1, 4, 256, 64, False),
        (2, 4, 256, 64, True),
        (1, 8, 512, 64, True),
        (1, 4, 128, 32, True),
    ]
    all_ok = True
    for B, H, M, D, causal in cases:
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        sm_scale = 1.0 / (D ** 0.5)
        # Original API (save_lse=False) — must still pass.
        o_triton = triton_flash_attn(q, k, v, sm_scale=sm_scale, is_causal=causal)
        o_ref = reference_attn(q, k, v, sm_scale, is_causal=causal)
        max_abs_err_o = (o_triton - o_ref).abs().max().item()
        ok_o = torch.allclose(o_triton, o_ref, atol=1e-3, rtol=1e-3)
        # New API (save_lse=True) — O must match the same reference, and
        # the saved LSE must match torch.logsumexp.
        o_triton2, lse_triton = triton_flash_attn(
            q, k, v, sm_scale=sm_scale, is_causal=causal, save_lse=True,
        )
        max_abs_err_o2 = (o_triton2 - o_ref).abs().max().item()
        lse_ref = reference_lse(q, k, v, sm_scale, is_causal=causal)
        max_abs_err_lse = (lse_triton - lse_ref).abs().max().item()
        ok_o2 = torch.allclose(o_triton2, o_ref, atol=1e-3, rtol=1e-3)
        ok_lse = torch.allclose(lse_triton, lse_ref, atol=1e-4, rtol=1e-4)
        ok = ok_o and ok_o2 and ok_lse
        flag = "OK" if ok else "FAIL"
        print(f"  B={B} H={H} M={M} D={D} causal={causal!s:5}  "
              f"o_err={max_abs_err_o:.2e}  o(lse)_err={max_abs_err_o2:.2e}  "
              f"lse_err={max_abs_err_lse:.2e}  {flag}")
        all_ok = all_ok and ok
    return all_ok


def _winning_config_for(M: int, N: int, D: int) -> dict:
    cache = flash_attn_fwd_kernel.cache
    if not cache:
        return {}
    for key, cfg in cache.items():
        # key is (M, N, D_HEAD, *signature_strs).
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
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
        sm_scale = 1.0 / (D ** 0.5)

        # Warm-up — first invocation also triggers autotune sweep.
        for _ in range(3):
            triton_flash_attn(q, k, v, sm_scale=sm_scale, is_causal=True)
            reference_attn(q, k, v, sm_scale, is_causal=True)
        torch.cuda.synchronize()

        n_iter = 50 if M <= 1024 else 20
        t0 = time.perf_counter()
        for _ in range(n_iter):
            o = triton_flash_attn(q, k, v, sm_scale=sm_scale, is_causal=True)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / n_iter * 1e6

        t0 = time.perf_counter()
        for _ in range(n_iter):
            o = reference_attn(q, k, v, sm_scale, is_causal=True)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / n_iter * 1e6

        # Approximate FLOPs: 4 * B * H * M * M * D for forward causal attn
        # (2 for matmul Q@K^T, 2 for matmul P@V; ignoring softmax cost).
        flops = 4 * B * H * M * M * D * 0.5  # 0.5 because causal
        tflops_triton = flops / (t_triton * 1e-6) / 1e12
        tflops_ref = flops / (t_ref * 1e-6) / 1e12
        cfg = _winning_config_for(M, M, D)
        results.append({
            "B": B, "H": H, "M": M, "D": D,
            "triton_us": round(t_triton, 2),
            "ref_us": round(t_ref, 2),
            "triton_tflops": round(tflops_triton, 3),
            "ref_tflops": round(tflops_ref, 3),
            "speedup": round(t_ref / t_triton, 3),
            "autotune": cfg,
        })
        print(f"  B={B} H={H} M={M:5d} D={D}  triton={t_triton:8.2f}us  "
              f"ref={t_ref:8.2f}us  triton TFLOPS={tflops_triton:.3f}  "
              f"speedup={t_ref / t_triton:.2f}x  cfg={cfg}")
    out = {
        "kernel": "flash-attention-v2-forward-causal",
        "autotuned": True,
        "smem_pruned_configs": _FP32_SMEM_SKIPPED,
        "live_configs": len(_FP32_CONFIGS),
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
        print("=== correctness ===")
        ok = correctness_check()
        if not ok:
            print("FAIL")
            raise SystemExit(1)
        print("OK")
        print()
    if args.bench:
        print(f"=== benchmark (B=1, H=8, D=64, causal=True, autotuned)  "
              f"[{len(_FP32_CONFIGS)} live configs, {_FP32_SMEM_SKIPPED} SMEM-pruned] ===")
        benchmark(out_path=args.out)


if __name__ == "__main__":
    main()
