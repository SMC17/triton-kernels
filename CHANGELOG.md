# Changelog

## v0.1.0 — 2026-05-18

First public release. Three Triton kernels with end-to-end correctness
and benchmark harnesses.

### Added

- `kernels/01_fused_softmax.py` — row-wise numerically-stable softmax.
  Correctness checked against `torch.softmax` to `max_abs_err ≤ 1.12e-8`
  across 4 shapes. Benchmark sweep over cols=128..16384 with rows=4096.
  Best speedup: **1.85×** at cols=16384 vs `torch.softmax`. Bandwidth
  saturates at ~96 GB/s on GTX 1650 (theoretical: 128 GB/s).
- `kernels/02_fused_rmsnorm.py` — single-pass RMSNorm with learnable
  scale. Correctness checked against `torch.nn.functional.rms_norm`
  to `max_abs_err ≤ 1.91e-6` across 5 shapes. Consistent **3.5×
  speedup** across cols=512..8192 vs torch's two-pass reference.
- `kernels/03_flash_attention.py` — FA-v2 forward kernel with optional
  causal mask. Correctness checked against plain
  `softmax(Q@Kᵀ*scale) @ V` reference to `max_abs_err ≤ 3.35e-8`
  across 6 (batch, head, seq_len, head_dim, causal) configurations
  including B=2, H=8, M=2048 causal. Benchmark sweep over M=128..2048.
  fp32 SIMT impl on Turing loses to cuBLAS-TF32 reference at every
  measured size (speedup 0.45–0.91×); documented honestly in README.
- `bench/*.json` — saved benchmark results from the v0.1.0 sweep.

### Evidence vocabulary applied

- `compiled` — yes, Triton 3.7.0 / PyTorch 2.5.1+cu121.
- `unit-tested` — correctness checks per kernel, all pass.
- `benchmarked` — wall-clock timings over 100 (or 50/20) iterations
  with warm-up + `torch.cuda.synchronize()` per measurement.
- `hardware-verified` — measured on GTX 1650 Max-Q Design (Turing,
  compute capability 7.5).

Words *deliberately not used*: "production", "verified", "fast"
(without a specific reference + tolerance), "optimized".

### Non-goals (deliberately deferred to v0.2.0)

- fp16 / bf16 / fp8 paths.
- Flash-attention backward pass.
- `@triton.autotune` over BLOCK_M/BLOCK_N/num_warps/num_stages grid.
- True fused-rmsnorm-linear (norm + matmul in one kernel) — Turing's
  64 KB SMEM/SM constrains tile sizing tightly; deserves its own
  artifact rather than half-cooked here.
- TF32-on / TF32-off reference comparison columns.
- Hopper-specific (wgmma / TMA) variants.
- ROCm / AMD path.
