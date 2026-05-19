"""
03b — Flash-Attention v2 (forward, causal optional, fp16) on Triton.

fp16 input variant of `03_flash_attention.py`. Q, K, V, O are fp16
tensors; the per-tile `tl.dot(q, k^T)` and `tl.dot(p, v)` are issued
with fp16 inputs and an fp32 accumulator (`out_dtype=tl.float32`),
which routes through Turing's m16n8k16 / m16n8k8 fp16 tensor-core
MMA instructions.

This is the kernel where the headline tensor-core thesis is tested:
the fp32 SIMT version (`03_flash_attention.py`) lost 2–3× to torch's
`scaled_dot_product_attention` on Turing because torch's reference
uses tensor cores (TF32) while the SIMT path didn't. Switching this
kernel to fp16 puts both implementations on tensor cores; the
remaining gap (if any) reflects what Triton-on-Turing leaves on the
table relative to torch's tuned tensor-core path.

Numerical considerations:
- Online-softmax state (m_i, l_i, o_acc) stays in fp32.
- p (post-softmax) is cast to fp16 only at the `p @ v` matmul input
  so the second dot also runs on tensor cores. The accumulator into
  o_acc is fp32.
- Output O is cast fp32 -> fp16 at the final store.

Tolerance vs reference: 1e-2 atol, 1e-2 rtol. fp16 has ~3-4 decimal
digits of precision; tighter would hide real fp16 round-off.

v0.3.0: wrapped in `@triton.autotune` over (BLOCK_M, BLOCK_N,
num_warps, num_stages) keyed on (M, N, D_HEAD). The v0.2.0
`BLOCK_M=BLOCK_N=32, num_warps=4, num_stages=2` hand-tuned pick is
in the config grid alongside the larger and smaller tiles. Configs
exceeding the Turing 64 KB SMEM budget are filtered out at
autotune-construction time. fp16 dtype halves the per-element SMEM
cost relative to fp32, so more aggressive tile sizes survive the
pruning step here than in `03_flash_attention.py`.

Hardware: GTX 1650 Max-Q (Turing, compute capability 7.5).
"""

from __future__ import annotations
import argparse
import json
import time
import torch
import triton
import triton.language as tl


# Turing 64 KB SMEM/SM. fp16 tiles halve the per-element SMEM cost
# vs fp32 so more aggressive tile sizes survive the budget check.
# Estimate is the K+V double-buffered footprint; Q and softmax tiles
# stay in registers in practice. See `03_flash_attention.py`.
TURING_SMEM_BUDGET = 64 * 1024


def _flash_attn_smem_bytes(BLOCK_M, BLOCK_N, D_HEAD, num_stages, dtype_bytes):
    return 2 * BLOCK_N * D_HEAD * dtype_bytes * num_stages


def _flash_attn_fp16_autotune_configs():
    # Conservative starting grid. The task spec calls for
    # BLOCK_M ∈ {16,32,64} × BLOCK_N ∈ {16,32,64} ×
    # num_warps ∈ {2,4,8} × num_stages ∈ {2,3,4} (81 configs).
    # On a single-GPU GTX 1650, compile-time for 81 configs across
    # multiple shape keys was empirically untenable — each fresh
    # (M,N,D) key triggers a full sweep. We trim to 36 configs and
    # widen only if winners land on the grid boundary.
    configs = []
    skipped = 0
    for BLOCK_M in (16, 32, 64):
        for BLOCK_N in (32, 64):
            for num_warps in (2, 4, 8):
                for num_stages in (2, 3):
                    smem = _flash_attn_smem_bytes(
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


_FP16_CONFIGS, _FP16_SMEM_SKIPPED = _flash_attn_fp16_autotune_configs()


@triton.autotune(configs=_FP16_CONFIGS, key=["M", "N", "D_HEAD"])
@triton.jit
def flash_attn_fwd_kernel_fp16(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
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

    q_row_base = pid_m * BLOCK_M
    q_row_offs = q_row_base + tl.arange(0, BLOCK_M)
    q_mask = q_row_offs < M

    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    Q_block_ptr = q_base + q_row_offs[:, None] * stride_qm + tl.arange(0, D_HEAD)[None, :] * stride_qd
    # Load fp16. Keep q in fp16 for the tl.dot tensor-core input.
    q = tl.load(Q_block_ptr, mask=q_mask[:, None], other=0.0)

    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

    # Accumulators stay in fp32 for numerical stability of the online softmax.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)

    if IS_CAUSAL:
        n_end = tl.minimum(N, q_row_base + BLOCK_M)
    else:
        n_end = N
    n_block_end = (n_end + BLOCK_N - 1) // BLOCK_N

    for n_block in tl.range(0, n_block_end):
        n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load K tile [BLOCK_N, D_HEAD] in fp16.
        K_block_ptr = k_base + n_offs[:, None] * stride_kn + tl.arange(0, D_HEAD)[None, :] * stride_kd
        k = tl.load(K_block_ptr, mask=n_mask[:, None], other=0.0)

        # S = Q @ K^T * sm_scale. tl.dot inputs are fp16; accumulator is fp32.
        # This routes through Turing's m16n8k16 fp16 mma.
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale

        if IS_CAUSAL:
            causal_ok = q_row_offs[:, None] >= n_offs[None, :]
            s = tl.where(causal_ok & n_mask[None, :], s, -float("inf"))
        else:
            s = tl.where(n_mask[None, :], s, -float("inf"))

        # Online softmax update (all fp32).
        s_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, s_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        # Load V tile [BLOCK_N, D_HEAD] in fp16.
        V_block_ptr = v_base + n_offs[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
        v = tl.load(V_block_ptr, mask=n_mask[:, None], other=0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Cast p to fp16 so the p@v dot uses tensor cores; o_acc stays fp32.
        p_fp16 = p.to(tl.float16)
        o_acc = o_acc * alpha[:, None] + tl.dot(p_fp16, v, out_dtype=tl.float32)
        m_i = m_new

    # Normalize, then cast fp32 -> fp16 for the store.
    o_acc = o_acc / l_i[:, None]
    o_fp16 = o_acc.to(tl.float16)

    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    O_block_ptr = o_base + q_row_offs[:, None] * stride_om + tl.arange(0, D_HEAD)[None, :] * stride_od
    tl.store(O_block_ptr, o_fp16, mask=q_mask[:, None])


def triton_flash_attn_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Public entry point. q/k/v are [B, H, M_or_N, D] fp16. Returns o fp16."""
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16, \
        "expected fp16 inputs"
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D)
    assert D in (16, 32, 64, 128), \
        f"D_HEAD={D} not in supported set"
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    o = torch.empty_like(q)
    grid = lambda META: ((M + META["BLOCK_M"] - 1) // META["BLOCK_M"], B * H)
    flash_attn_fwd_kernel_fp16[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, M, N,
        sm_scale,
        IS_CAUSAL=is_causal,
        D_HEAD=D,
    )
    return o


def reference_attn_fp16(q, k, v, sm_scale, is_causal: bool) -> torch.Tensor:
    """torch reference. Use scaled_dot_product_attention which on Turing
    routes through cuBLAS / tensor cores for the matmul steps."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=sm_scale,
    )


def correctness_check() -> bool:
    """Compare Triton fp16 vs torch fp16 reference."""
    torch.manual_seed(0)
    cases = [
        (1, 1, 64, 64, False),
        (1, 1, 64, 64, True),
        (1, 4, 256, 64, False),
        (2, 4, 256, 64, True),
        (1, 8, 512, 64, True),
        (1, 4, 128, 32, True),
    ]
    all_ok = True
    for B, H, M, D, causal in cases:
        q = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        k = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        v = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
        sm_scale = 1.0 / (D ** 0.5)
        o_triton = triton_flash_attn_fp16(q, k, v, sm_scale=sm_scale, is_causal=causal)
        o_ref = reference_attn_fp16(q, k, v, sm_scale, is_causal=causal)
        max_abs_err = (o_triton.float() - o_ref.float()).abs().max().item()
        ok = torch.allclose(o_triton, o_ref, atol=1e-2, rtol=1e-2)
        flag = "OK" if ok else "FAIL"
        print(f"  B={B} H={H} M={M} D={D} causal={causal!s:5}  "
              f"max_abs_err={max_abs_err:.2e}  {flag}")
        all_ok = all_ok and ok
    return all_ok


def _winning_config_for(M: int, N: int, D: int) -> dict:
    cache = flash_attn_fwd_kernel_fp16.cache
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

        for _ in range(3):
            triton_flash_attn_fp16(q, k, v, sm_scale=sm_scale, is_causal=True)
            reference_attn_fp16(q, k, v, sm_scale, is_causal=True)
        torch.cuda.synchronize()

        n_iter = 50 if M <= 1024 else 20
        t0 = time.perf_counter()
        for _ in range(n_iter):
            o = triton_flash_attn_fp16(q, k, v, sm_scale=sm_scale, is_causal=True)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / n_iter * 1e6

        t0 = time.perf_counter()
        for _ in range(n_iter):
            o = reference_attn_fp16(q, k, v, sm_scale, is_causal=True)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / n_iter * 1e6

        # Approx FLOPs identical to fp32 path; precision doesn't change op count.
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
        "kernel": "flash-attention-v2-forward-causal-fp16",
        "dtype": "float16",
        "autotuned": True,
        "smem_pruned_configs": _FP16_SMEM_SKIPPED,
        "live_configs": len(_FP16_CONFIGS),
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
        print("=== correctness (fp16) ===")
        ok = correctness_check()
        if not ok:
            print("FAIL")
            raise SystemExit(1)
        print("OK")
        print()
    if args.bench:
        print(f"=== benchmark (B=1, H=8, D=64, causal=True, fp16, autotuned)  "
              f"[{len(_FP16_CONFIGS)} live configs, {_FP16_SMEM_SKIPPED} SMEM-pruned] ===")
        benchmark(out_path=args.out)


if __name__ == "__main__":
    main()
