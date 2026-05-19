"""
02 — Fused RMSNorm on Triton.

A row-wise RMSNorm with learnable scale weight. Fuses the two-pass
PyTorch reference (read for variance, then read again for the scaled
output) into a single Triton kernel that reads each row exactly once.

Output: y[m, k] = x[m, k] / sqrt(mean(x[m, :] ** 2) + eps) * weight[k]

This is a deliberate single-pass single-kernel implementation that
assumes the per-row tensor fits in a single Triton block. For
K > BLOCK_SIZE_MAX, a tiled / online variant is needed (the same
shape Welford-style numeric stability would also apply); both are
out of scope for v0.1.0 and are tracked in ROADMAP.md.

The fused-rmsnorm-linear (RMSNorm + linear projection in one kernel)
is a separate, harder artifact deferred to v0.3 — single-pass fusion
of the norm reduction with a matmul accumulator requires careful
shared-memory budgeting that the GTX 1650 (Turing, 64KB SMEM per SM)
constrains tightly. v0.2.0 ships an fp16 variant of *this* kernel as
`02b_fused_rmsnorm_fp16.py` but not the fused linear projection.

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
def rmsnorm_kernel(
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
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)

    # Mean of squares: sum(x^2) / n_cols. Single-pass.
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    mean_sq = sum_sq / n_cols
    # Reciprocal of RMS norm: 1 / sqrt(mean_sq + eps).
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # Load per-channel weight and scale.
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    y = x * rrms * w

    y_row_ptr = y_ptr + row_idx * y_row_stride
    tl.store(y_row_ptr + col_offsets, y, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Public entry point. x is 2D [rows, cols]; weight is 1D [cols]."""
    assert x.dim() == 2, "expected 2D input"
    assert weight.dim() == 1 and weight.shape[0] == x.shape[1], \
        "weight must be 1D and match x's last dim"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    assert BLOCK_SIZE <= 32768, \
        f"K={n_cols} (BLOCK={BLOCK_SIZE}) exceeds single-pass kernel limit; tiled variant is v0.3 territory"
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)
    rmsnorm_kernel[(n_rows,)](
        x, y, weight,
        x.stride(0), y.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference for correctness comparison."""
    # torch.rms_norm exists in recent versions; fall back to manual if not.
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def correctness_check() -> bool:
    """Compare Triton vs torch reference across a few shapes."""
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
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        w = torch.randn(cols, device="cuda", dtype=torch.float32)
        y_triton = triton_rmsnorm(x, w)
        y_torch = reference_rmsnorm(x, w)
        max_abs_err = (y_triton - y_torch).abs().max().item()
        ok = torch.allclose(y_triton, y_torch, atol=1e-4, rtol=1e-4)
        print(f"  shape={(rows, cols)} max_abs_err={max_abs_err:.2e} {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def benchmark(out_path: str | None = None) -> dict:
    torch.manual_seed(0)
    rows = 4096
    results = []
    for cols in [128, 256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        w = torch.randn(cols, device="cuda", dtype=torch.float32)

        for _ in range(3):
            triton_rmsnorm(x, w)
            reference_rmsnorm(x, w)
        torch.cuda.synchronize()

        n_iter = 100
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_t = triton_rmsnorm(x, w)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / n_iter * 1e6

        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_p = reference_rmsnorm(x, w)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / n_iter * 1e6

        # Bandwidth: read x + read w + write y = 2*M*K*4 + K*4 bytes
        bytes_moved = rows * cols * 4 * 2 + cols * 4
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
    out = {"kernel": "fused-rmsnorm", "results": results}
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
        print("=== benchmark (rows=4096) ===")
        benchmark(out_path=args.out)


if __name__ == "__main__":
    main()
