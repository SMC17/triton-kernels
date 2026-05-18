"""
01 — Fused Softmax (row-wise) on Triton.

A row-wise softmax kernel that fuses the read, max, subtract, exp, sum,
divide, and write passes that a naive PyTorch implementation issues as
separate kernel launches. Single Triton program handles one or more
rows; the per-row work is fully streamed through registers without
intermediate HBM writeback.

Canonical Triton tutorial: this is the 02-fused-softmax.py demo from
the official `triton-lang/triton` tutorials, reimplemented from
specification rather than copy-pasted. Numerical stability follows
the standard `x - max(x)` trick.

Hardware: GTX 1650 (Turing, compute capability 7.5). Standard Triton
kernel; no Hopper-specific (wgmma / TMA) features used.
"""

from __future__ import annotations
import argparse
import json
import time
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # One Triton program handles a slab of rows. The slab is sized by
    # the launcher to keep occupancy reasonable on a small GPU.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_in_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_in_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Numerically stable softmax: subtract max before exp.
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        row_out_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = row_out_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Public entry point. Row-wise softmax over the last dim of a 2D tensor."""
    assert x.dim() == 2, "expected 2D input (rows, cols)"
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    num_stages = 4 if BLOCK_SIZE >= 2048 else 2
    y = torch.empty_like(x)
    # Cap the number of programs at 1024 so each program handles
    # multiple rows when n_rows is large; this keeps register
    # pressure under control on the GTX 1650.
    num_programs = min(n_rows, 1024)
    softmax_kernel[(num_programs,)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


def correctness_check() -> bool:
    """Compare Triton vs torch.softmax across a few shapes. Returns True
    if all match within atol=1e-5."""
    torch.manual_seed(0)
    shapes = [
        (823, 781),
        (4096, 1024),
        (32, 2048),
        (128, 8192),
    ]
    all_ok = True
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        y_triton = triton_softmax(x)
        y_torch = torch.softmax(x, dim=-1)
        max_abs_err = (y_triton - y_torch).abs().max().item()
        ok = torch.allclose(y_triton, y_torch, atol=1e-5)
        print(f"  shape={(rows, cols)} max_abs_err={max_abs_err:.2e} {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def benchmark(out_path: str | None = None) -> dict:
    """Time triton vs torch.softmax across a column-size sweep. Returns
    a dict suitable for JSON serialization."""
    torch.manual_seed(0)
    rows = 4096
    results = []
    for cols in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        # Warm-up
        for _ in range(3):
            triton_softmax(x)
            torch.softmax(x, dim=-1)
        torch.cuda.synchronize()

        # Time triton
        n_iter = 100
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_t = triton_softmax(x)
        torch.cuda.synchronize()
        t_triton = (time.perf_counter() - t0) / n_iter * 1e6  # microseconds

        # Time torch
        t0 = time.perf_counter()
        for _ in range(n_iter):
            y_p = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
        t_torch = (time.perf_counter() - t0) / n_iter * 1e6

        gbps_triton = (rows * cols * 4 * 2) / (t_triton * 1e-6) / 1e9
        gbps_torch = (rows * cols * 4 * 2) / (t_torch * 1e-6) / 1e9
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
    out = {"kernel": "fused-softmax", "results": results}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness", action="store_true", help="run correctness check only")
    parser.add_argument("--bench", action="store_true", help="run benchmark only")
    parser.add_argument("--out", type=str, default=None, help="write benchmark JSON here")
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
