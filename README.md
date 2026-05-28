# triton-kernels

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![test](https://github.com/SMC17/triton-kernels/actions/workflows/test.yml/badge.svg)](https://github.com/SMC17/triton-kernels/actions/workflows/test.yml)

Triton kernels for inference primitives, paired with correctness +
benchmark harnesses against PyTorch references. fp32 + fp16 forward
plus fp32 + fp16 backward for flash-attention v2 (Dao 2023). All eight
kernels carry `@triton.autotune` over a SMEM-pruned config grid.

Pre-1.0 substrate. No stability claims. Bench numbers were measured
on a single Turing GPU (GTX 1650, sm_75); see [`STATUS.md`](STATUS.md)
for the live state register and [`CHANGELOG.md`](CHANGELOG.md) for the
full release log.

## What ships

| # | kernel | dtype | pass | autotune | source |
|---|--------|-------|------|----------|--------|
| 01 | fused softmax | fp32 | forward | yes | [`kernels/01_fused_softmax.py`](kernels/01_fused_softmax.py) |
| 01b | fused softmax | fp16 | forward | yes | [`kernels/01b_fused_softmax_fp16.py`](kernels/01b_fused_softmax_fp16.py) |
| 02 | fused RMSNorm | fp32 | forward | yes | [`kernels/02_fused_rmsnorm.py`](kernels/02_fused_rmsnorm.py) |
| 02b | fused RMSNorm | fp16 | forward | yes | [`kernels/02b_fused_rmsnorm_fp16.py`](kernels/02b_fused_rmsnorm_fp16.py) |
| 03 | flash-attention v2 | fp32 | forward (+ optional LSE) | yes | [`kernels/03_flash_attention.py`](kernels/03_flash_attention.py) |
| 03b | flash-attention v2 | fp16 | forward (+ optional LSE) | yes | [`kernels/03b_flash_attention_fp16.py`](kernels/03b_flash_attention_fp16.py) |
| 04 | flash-attention v2 | fp32 | backward, causal optional | yes | [`kernels/04_flash_attention_backward.py`](kernels/04_flash_attention_backward.py) |
| 04b | flash-attention v2 | fp16 | backward, causal optional | yes | [`kernels/04b_flash_attention_backward_fp16.py`](kernels/04b_flash_attention_backward_fp16.py) |

Each kernel ships its own correctness check (against a torch reference
at a documented tolerance) and a benchmark sweep with JSON results in
[`bench/`](bench).

## Hardware tier required

Triton + an NVIDIA GPU. The bench numbers in `bench/` were captured
on a single GTX 1650 Max-Q (Turing, compute capability 7.5); the
kernels themselves do not assume Turing-specific layouts, but the
performance characteristics will differ on Ampere+ where fp16 tensor
cores are reachable through Triton's MLIR pipeline (see "What is NOT
shipped" below).

- **Minimum to compile + run correctness checks:** any CUDA-capable
  NVIDIA GPU that Triton 3.x supports.
- **fp16 tensor-core path on Triton 3.7.0:** Ampere+ (sm_80+). On
  sm_75 (Turing), `TritonGPUAccelerateMatmul` declines the MMA layout
  promotion and the fp16 dot lowers to SIMT `fma.rn.f32`. The Turing
  fp16 mma codegen (`m16n8k8`) *is* present in the Triton binary; the
  MLIR layout-assignment pass is the gate. See
  [`docs/PTX_INSPECTION.md`](docs/PTX_INSPECTION.md) for the PTX-level
  diagnosis behind this finding.
- **bf16, fp8:** Ampere+ for bf16, Hopper for fp8 mma. Not in scope.
- **ROCm / AMD:** out of scope for this artifact.

## What is NOT shipped

Substrate-honest non-claims, in the same order the kernels appear:

- **Variable-length / packed sequences and arbitrary attention masks.**
  Strict causal is supported; varlen is parked at v0.5 pending the
  tensor-core MLIR question (see [`STATUS.md`](STATUS.md) Blocked).
- **Dropout in forward and backward.** FA-2's shared-RNG-state
  dropout is documented but not implemented here.
- **Triton-side gradient checkpointing.** The backward recomputes
  the attention scores and softmax from Q, K, V and the saved LSE,
  but does not implement a separate activation-save tier.
- **Fused-RMSNorm-Linear** (RMSNorm + GEMM in one kernel). Turing's
  64 KB SMEM/SM budget constrains tile sizing tightly and the work
  deserves its own artifact.
- **bf16 / fp8 paths.** Out of hardware reach on this bench host.
- **Persistent per-disk autotune cache across processes.** Triton
  caches the winning config in memory per Python process; first
  invocation at a new key on a fresh process still triggers a sweep.
- **A hosted CI run of the kernels themselves.** GitHub free runners
  are CPU-only; the [`test.yml`](.github/workflows/test.yml) workflow
  runs lint + AST-parse + smoke tests but cannot exercise the
  CUDA-dependent code paths. The local
  [`scripts/run_correctness_sweep.sh`](scripts/run_correctness_sweep.sh)
  is the bench-host source of truth for kernel correctness.

## Build / run

Requires a CUDA-capable NVIDIA GPU and a Python environment with
PyTorch (with CUDA) + Triton installed.

```sh
# Per-kernel correctness + benchmark (each script is a CLI entry)
python kernels/01_fused_softmax.py
python kernels/01b_fused_softmax_fp16.py
python kernels/02_fused_rmsnorm.py
python kernels/02b_fused_rmsnorm_fp16.py
python kernels/03_flash_attention.py
python kernels/03b_flash_attention_fp16.py
python kernels/04_flash_attention_backward.py
python kernels/04b_flash_attention_backward_fp16.py

# Save benchmark JSON (autotune winning configs annotated per shape)
python kernels/01_fused_softmax.py --bench --out bench/01_fused_softmax_autotuned.json
# ...one --bench invocation per kernel...

# Local 8-kernel no-regression sweep (set TRITON_PY if your python
# wrapper has a different name; e.g. TRITON_PY=triton-python).
scripts/run_correctness_sweep.sh
```

This repo's bench host uses a thin wrapper `triton-python` that
exports `LD_LIBRARY_PATH=/usr/lib` for `libz.so.1` lookup against the
Triton C extension on an Arch+nix dual-runtime setup. On a standard
distro the wrapper is unnecessary; substitute `python`.

First-invocation cost: with autotune, the first launch at a new shape
key sweeps every config. The flash-attention bench over five shapes
(M = 128…2048) takes ~90s wall-clock on a GTX 1650; softmax and
RMSNorm sweeps run in under 30s.

## Reproducibility

- Hardware: NVIDIA GeForce GTX 1650 with Max-Q Design (Turing, compute
  capability 7.5, 3.9 GB device memory).
- Software: Triton 3.7.0, PyTorch 2.5.1+cu121.
- Bench shapes seeded with `torch.manual_seed(0)`. fp32 reductions are
  bit-deterministic; fp16 reductions have visible round-off vs the
  torch reference, captured by `atol=1e-2, rtol=1e-2` correctness
  tolerance with `max_abs_err` reported per case.
- Timings are wall-clock over 100 (softmax / RMSNorm) or 50/20
  (flash-attention) iterations after 3 warm-up iterations, with
  `torch.cuda.synchronize()` between runs. Autotune cache is populated
  during warmup; timing iterations see the cached winning config.
- Sub-200us kernels on this GTX 1650 vary ±20% per run with
  power/clock-state jitter; comparisons in that regime are within-noise
  unless the delta exceeds 2×.

## Evidence vocabulary

- `compiled` — yes, Triton 3.7.0 / PyTorch 2.5.1+cu121.
- `unit-tested` — per-kernel correctness check passes at the
  tolerances quoted in the in-file docstrings (fp32: `1e-5` / `1e-4` /
  `1e-3`; fp16: `1e-2` across all three).
- `benchmarked` — wall-clock timings with warmup + sync; JSON in
  [`bench/`](bench).
- `hardware-verified` — measured on the GTX 1650 Max-Q listed above.

Words *not* used: "production", "verified" (without tolerance +
reference), "fast" (without a specific reference + shape), "optimized."

## What this work is for

This repo is the kernel-stack-fluency artifact in a research agenda
focused on agentic-safety substrate. ML systems / kernels is the
cross-domain competence layer — knowing the GPU stack well enough to
debug an attention-path bottleneck in a deployment, rather than to
compete with the cuBLAS / cutlass / FlashAttention teams who own that
surface.

The discipline above (pre-registered shape sweep, correctness checks
before performance numbers, honest non-claims where the kernel does
not improve on the reference) is the same discipline that runs
through the rest of the substrate fleet.

## Layout

```
kernels/                Eight kernel modules (01..04 fp32 + 01b..04b fp16).
bench/                  Per-kernel benchmark JSON (raw + autotuned variants).
ptx-inspection/         PTX dumps + dump_ptx.py used in v0.4.2 diagnosis.
docs/PTX_INSPECTION.md  PTX-level writeup behind the sm_75 tensor-core finding.
scripts/run_correctness_sweep.sh
                        Local 8-kernel no-regression sweep (operator script).
tests/test_smoke.py     CPU-only smoke contract — kernel inventory + public-API
                        + autotune-grid presence. Runs in CI without CUDA.
.github/workflows/test.yml
                        Ruff lint + AST parse + smoke tests. CUDA paths not
                        exercised on hosted runners.
STATUS.md               Honest-state register (Released / Blocked / Open).
CHANGELOG.md            Per-release notes incl. PTX-level diagnoses.
```

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
