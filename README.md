# triton-kernels

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)

Three Triton kernels with end-to-end benchmark + correctness harnesses
against PyTorch references. fp32 implementations targeting Turing
(GTX 1650, compute capability 7.5). Each kernel ships its own
correctness check, benchmark sweep, and JSON results.

> Pre-registered honest non-claim: this is a v0.1 set on a small
> consumer Turing GPU. Numbers below are reproducible on that hardware
> and not representative of what a tuned fp16+tensor-core version
> would do on Hopper. The discipline is the harness; performance
> across hardware classes is a v0.2 concern.

## What's in v0.1.0

| Kernel | Status | Best speedup vs torch | Bandwidth saturated |
|---|---|---|---|
| `01_fused_softmax.py` | row-wise, numerically stable | **1.85×** at cols=16384 | ~96 GB/s |
| `02_fused_rmsnorm.py` | learnable scale, single-pass | **3.5×** at cols ≥ 512 | ~96 GB/s |
| `03_flash_attention.py` | FA-v2 forward, causal | **0.5–0.9×** (cuBLAS wins fp32 on Turing) | — |

All three pass correctness checks against PyTorch references
(`torch.softmax`, `torch.nn.functional.rms_norm`, plain
`Q @ Kᵀ * scale → softmax → @ V`) to:

- Softmax: `max_abs_err ≤ 1.12e-08`
- RMSNorm: `max_abs_err ≤ 1.91e-06`
- Flash-Attention (forward, causal+non-causal): `max_abs_err ≤ 3.35e-08`

## Honest reading of the numbers

The **softmax** result is mostly a "match cuBLAS at large columns,
underperform at small columns" story. The win at cols=16384 (1.85×)
is real and reproducible; below cols=2048 there isn't a meaningful
gap. The bandwidth saturation at ~96 GB/s is the load-bearing
finding — it's close to the GTX 1650's theoretical 128 GB/s, so
this kernel is genuinely memory-bandwidth bound.

The **RMSNorm** result is the clearest performance story. The Triton
kernel reads each row exactly once; the PyTorch reference does two
passes (variance reduction + scaled output). The fused single-pass
kernel saves a full row read and gets a consistent ~3.5× speedup
across the sweep.

The **flash-attention** result is honest non-victory and the most
informative. At fp32 on Turing, PyTorch's reference matmul uses
cuBLAS, which routes through TF32 tensor cores on this GPU. My
Triton kernel does plain fp32 SIMT multiplies, so it loses 2–3×
on the matmul step that flash-attention spends most of its time
on. The flash-attention *algorithm* is correct (max_abs_err ~3e-8
across all 6 cases including B=2, H=8, M=2048 with causal mask),
so the artifact is a working FA-v2 forward — it just doesn't beat
cuBLAS on this hardware class at this precision. The win comes back
when (a) sequence length exceeds memory and the O(N²) attention
matrix can't be materialized at all, or (b) the kernel moves to
fp16+tensor-cores in v0.2.

## What is *not* in v0.1.0

- **fp16 / bf16 / fp8 paths.** All three kernels are fp32. Adding
  fp16 with `tl.dot(..., out_dtype=tl.float32)` for the matmul
  accumulator is the v0.2.0 target.
- **Backward pass for flash-attention.** Forward only. The backward
  pass needs the saved log-sum-exp + the dO statistics for the
  gradient computation; that's a second kernel of similar size.
- **Autotuning.** All three kernels use fixed `BLOCK_M`/`BLOCK_N`/
  `num_warps`/`num_stages` heuristics. `@triton.autotune` over a
  grid of configs is the obvious v0.2.0 win on tail-shape kernels.
- **`fused-rmsnorm-linear`** (RMSNorm + matmul in one kernel).
  Single-kernel norm+gemm is the harder fusion; deferred to v0.2.0
  because the matmul tile sizing on Turing (64KB SMEM/SM) needs
  careful occupancy tuning that I'd rather get right than ship
  half-cooked.
- **TF32 explicit hints.** PyTorch's `matmul` autoselects TF32 on
  Turing+ via `torch.backends.cuda.matmul.allow_tf32`. The comparison
  benchmarks above run with whatever the venv's default policy is;
  v0.2 will report both "TF32 on" and "TF32 off" reference numbers
  so the Triton-vs-cuBLAS picture is unambiguous.

## Build / run

Requires the Triton+PyTorch venv set up per `reference_triton_python_wrapper.md`
(needs `LD_LIBRARY_PATH=/usr/lib` for `libz.so.1`):

```sh
# Correctness + benchmark on every kernel
triton-python kernels/01_fused_softmax.py
triton-python kernels/02_fused_rmsnorm.py
triton-python kernels/03_flash_attention.py

# Save benchmark JSON
triton-python kernels/01_fused_softmax.py --bench --out bench/01_fused_softmax.json
triton-python kernels/02_fused_rmsnorm.py --bench --out bench/02_fused_rmsnorm.json
triton-python kernels/03_flash_attention.py --bench --out bench/03_flash_attention.json
```

The `triton-python` wrapper is a thin shell script at
`~/.local/bin/triton-python` that sets `LD_LIBRARY_PATH` so the
nix-linked Python interpreter finds the system `libz.so.1`. Substitute
`python` if your env doesn't need that.

## Reproducibility

- Hardware: NVIDIA GeForce GTX 1650 with Max-Q Design (Turing,
  compute capability 7.5, 3.9 GB device memory).
- Software: Triton 3.7.0, PyTorch 2.5.1+cu121, CUDA 13.2, Arch Linux
  kernel 7.0.3.
- Bench shapes are seeded with `torch.manual_seed(0)`; the kernel's
  fp32 reductions are deterministic.
- Timings are wall-clock over 100 (or 50, or 20 for very large)
  iterations after 3 warm-up iterations, with `torch.cuda.synchronize()`
  between runs. All JSON outputs live in `bench/`.

## What this work is for

This repo is the Lane D artifact in a four-month research agenda
focused on agentic-safety substrate work. ML systems / RL is the
Green-Beret cross-domain competence layer — knowing the kernel stack
well enough to debug a frontier-model deployment when the bottleneck
turns out to be in the attention path, rather than to compete with
the cuBLAS / cutlass / FlashAttention teams who own that surface.

The discipline above (pre-registered shape sweep, correctness
verification before performance numbers, honest non-claims on
hardware mismatch) is the same discipline that runs through
`evals-agentic-control`, `stax-experiment`, and the rest of the
fleet at `github.com/SMC17`.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
