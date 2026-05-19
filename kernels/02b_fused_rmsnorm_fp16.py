"""
02b — Fused RMSNorm (fp16 inputs/weights) on Triton.

fp16 input/weight variant of `02_fused_rmsnorm.py`. fp16 loads halve
HBM traffic relative to the fp32 path; the per-row variance reduction
still runs in fp32 to preserve numerical headroom on
`mean(x ** 2) + eps`, which can underflow / lose precision badly if
done in fp16 directly.

Like fp16 softmax, this kernel has no matmul, so it doesn't exercise
Turing's fp16 tensor cores. The win available is bandwidth from fp16
loads/stores. The fp32 baseline already saturates ~96 GB/s of the
GTX 1650's ~128 GB/s theoretical, so the headroom is finite.

Hardware: GTX 1650 Max-Q (Turing, compute capability 7.5).
"""

from __future__ import annotations
import argparse
import json
import time
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel_fp16(
    x_ptr,
    y_ptr,
    weight_ptr,
    x_row_stride,
    y_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row_ptr = x_ptr + row_idx * x_row_stride
    x_fp16 = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
    # Promote to fp32 for the reduction.
    x = x_fp16.to(tl.float32)

    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    mean_sq = sum_sq / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # Load per-channel weight in fp16, promote to fp32 for the multiply.
    w_fp16 = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    w = w_fp16.to(tl.float32)
    y_fp32 = x * rrms * w
    y = y_fp32.to(tl.float16)

    y_row_ptr = y_ptr + row_idx * y_row_stride
    tl.store(y_row_ptr + col_offsets, y, mask=mask)


def triton_rmsnorm_fp16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Public entry point. x is 2D [rows, cols] fp16; weight is 1D [cols] fp16."""
    assert x.dim() == 2, "expected 2D input"
    assert x.dtype == torch.float16, "expected fp16 input"
    assert weight.dim() == 1 and weight.shape[0] == x.shape[1], \
        "weight must be 1D and match x's last dim"
    assert weight.dtype == torch.float16, "expected fp16 weight"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    assert BLOCK_SIZE <= 32768, \
        f"K={n_cols} (BLOCK={BLOCK_SIZE}) exceeds single-pass kernel limit"
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)
    rmsnorm_kernel_fp16[(n_rows,)](
        x, y, weight,
        x.stride(0), y.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


def reference_rmsnorm_fp16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch fp16 reference."""
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def correctness_check() -> bool:
    """Compare Triton fp16 vs torch fp16 reference across a few shapes."""
    torch.manual_seed(0)
    shapes = [
        (823, 781),
        (4096, 1024),
        (32, 2048),
        (128, 4096),
        (1024, 8192),
    ]
    all_ok = True
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        w = torch.randn(cols, device="cuda", dtype=torch.float16)
        y_triton = triton_rmsnorm_fp16(x, w)
        y_torch = reference_rmsnorm_fp16(x, w)
        max_abs_err = (y_triton.float() - y_torch.float()).abs().max().item()
        ok = torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2)
        print(f"  shape={(rows, cols)} max_abs_err={max_abs_err:.2e} {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def benchmark(out_path: str | None = None) -> dict:
    torch.manual_seed(0)
    rows = 4096
    results = []
    for cols in [128, 256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        w = torch.randn(cols, device="cuda", dtype=torch.float16)

        for _ in range(3):
            triton_rmsnorm_fp16(x, w)
            reference_rmsnorm_fp16(x, w)
        torch.cuda.synchronize()

        n_iter = 100
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_t = triton_rmsnorm_fp16(x, w)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / n_iter * 1e6

        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_p = reference_rmsnorm_fp16(x, w)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / n_iter * 1e6

        # fp16 = 2 bytes. Read x + write y + read w.
        bytes_moved = rows * cols * 2 * 2 + cols * 2
        gbps_triton = bytes_moved / (t_triton * 1e-6) / 1e9
        gbps_torch = bytes_moved / (t_torch * 1e-6) / 1e9
        results.append({
            "rows": rows,
            "cols": cols,
            "triton_us": round(t_triton, 2),
            "torch_us": round(t_torch, 2),
            "triton_gbps": round(gbps_triton, 2),
            "torch_gbps": round(gbps_torch, 2),
            "speedup": round(t_torch / t_triton, 3),
        })
        print(f"  cols={cols:5d}  triton={t_triton:7.2f}us  torch={t_torch:7.2f}us  "
              f"triton GB/s={gbps_triton:6.1f}  speedup={t_torch / t_triton:.2f}x")
    out = {"kernel": "fused-rmsnorm-fp16", "dtype": "float16", "results": results}
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
        print("=== benchmark (rows=4096, fp16) ===")
        benchmark(out_path=args.out)


if __name__ == "__main__":
    main()
