# Changelog

## v0.3.0 — 2026-05-19

All six kernels (3 fp32 + 3 fp16) wrapped in `@triton.autotune`,
closing the v0.2.0 documented finding that tile-size choices do not
transfer across precisions and require offline hand-sweeps per
hardware target.

### Added

- `@triton.autotune` decorator on all six kernels with per-kernel
  config grids:
  - **Softmax / RMSNorm (fp32 + fp16):** 8 configs of
    `num_warps ∈ {2, 4, 8, 16}` × `num_stages ∈ {2, 4}`. Keyed on
    `n_cols`. `BLOCK_SIZE` is fixed by the launcher at
    `next_power_of_2(n_cols)` because the kernel issues a single
    masked load per row; tuning `BLOCK_SIZE` at the kernel level
    would admit configs that silently drop columns under the mask.
  - **Flash-attention (fp32 + fp16):** 36-config grid of
    `BLOCK_M ∈ {16, 32, 64}` × `BLOCK_N ∈ {32, 64}` ×
    `num_warps ∈ {2, 4, 8}` × `num_stages ∈ {2, 3}`. Keyed on
    `(M, N, D_HEAD)`. Configs whose double-buffered K+V SMEM
    footprint exceeds the Turing 64 KB/SM budget are pruned at
    autotune-construction time (fp32: 9 pruned, 27 live;
    fp16: 0 pruned, 36 live).

- `bench/0*_autotuned.json` — six new benchmark JSON files. Each
  result row includes an `"autotune"` field listing the winning
  config chosen by the autotuner at that shape on a GTX 1650
  Max-Q.

### Performance vs v0.2.0 hand-tuned

**Flash-attention fp32** is the headline result. The v0.1 / v0.2
config (`BLOCK_M=BLOCK_N=64, num_warps=4, num_stages=2`) lost to
torch's `scaled_dot_product_attention` at every measured shape
(speedup 0.45–0.91×). Autotuning inverts this — Triton fp32 now
beats torch at every shape:

| M | v0.2 us | v0.3 us | v0.2 speedup vs ref | v0.3 speedup vs ref |
|---|---|---|---|---|
| 128 | 241.96 | 137.58 | 1.10× | **1.55×** |
| 256 | 754.74 | 154.55 | 0.47× | **1.91×** |
| 512 | 2474.66 | 515.21 | 0.45× | **2.21×** |
| 1024 | 8205.14 | 1891.34 | 0.54× | **2.41×** |
| 2048 | 32146.83 | 7304.04 | 0.58× | **2.63×** |

At M=2048 the autotune-vs-hand-tuned delta is **4.4×**. The
autotuner picks `BLOCK_M=BLOCK_N=64, num_warps=8, num_stages=2`
for M ≥ 1024 (which the v0.1 / v0.2 heuristic with `num_warps=4`
underutilized) and switches to smaller tiles (`BLOCK_M=16` or
`BLOCK_M=32`) at M ∈ {256, 512} where they fit more program slots.

**Flash-attention fp16** at M=128 closes the v0.2.0 loss to torch
SDPA (0.47×) to parity (1.02×) — a 2.26× wall-clock improvement.
For M ≥ 256 the v0.2.0 hand-tuned `(32, 32, 4, 2)` config was
already near-optimal; autotune confirms it for M=128 and lands on
`(BLOCK_M=16, BLOCK_N=32, nw=2, ns=3)` at M=256 worth ~5%, and
`num_stages=3` (instead of 2) at M ≥ 512 worth ~5%.

**Softmax (fp32 + fp16)** and **RMSNorm (fp32 + fp16)** were
already bandwidth-saturated at v0.2.0 (~96 GB/s of the GTX 1650's
~128 GB/s theoretical). Autotuning preserves that saturation at
cols ≥ 512 and is within measurement noise at cols ≤ 256 (the
launch-overhead-bound regime). RMSNorm holds its consistent
3.4–3.7× speedup vs torch's two-pass reference across the column
sweep.

Honest read: autotuning **did not improve** the softmax and RMSNorm
kernels on this GPU — they were already at the right config for
their bandwidth-bound regime. The v0.3 win is concentrated in
flash-attention, particularly the fp32 inversion above.

### Winning-config table (empirical sweet-spot, GTX 1650 Turing)

Per-shape winning configs that v0.4 could promote to hard-coded
defaults. The autotune path remains in place so new shapes still
get a sweep on first invocation.

**softmax fp32 / fp16 (rows=4096):**

| n_cols | fp32 (nw, ns) | fp16 (nw, ns) |
|---|---|---|
| 128 | (2, 4) | (2, 4) |
| 256 | (2, 2) | (2, 2) |
| 512 | (2, 2) | (2, 4) |
| 1024 | (4, 4) | (2, 4) |
| 2048 | (8, 2) | (4, 4) |
| 4096 | (2, 4) | (4, 4) |
| 8192 | (16, 4) | (16, 2) |
| 16384 | (8, 4) | (8, 2) |

**rmsnorm fp32 / fp16 (rows=4096):**

| n_cols | fp32 (nw, ns) | fp16 (nw, ns) |
|---|---|---|
| 128 | (2, 4) | (2, 4) |
| 256 | (2, 2) | (2, 2) |
| 512 | (2, 4) | (2, 4) |
| 1024 | (4, 4) | (2, 2) |
| 2048 | (16, 4) | (4, 4) |
| 4096 | (16, 2) | (8, 2) |
| 8192 | (8, 4) | (16, 2) |

**flash-attention fp32 / fp16 (B=1, H=8, D=64, causal=True):**

| M | fp32 (BLOCK_M, BLOCK_N, nw, ns) | fp16 (BLOCK_M, BLOCK_N, nw, ns) |
|---|---|---|
| 128 | (64, 64, 8, 2) | (32, 32, 4, 2) |
| 256 | (16, 32, 2, 3) | (16, 32, 2, 3) |
| 512 | (32, 32, 2, 2) | (32, 32, 4, 3) |
| 1024 | (64, 64, 8, 2) | (32, 32, 4, 3) |
| 2048 | (64, 64, 8, 2) | (32, 32, 4, 3) |

### Config-space pruning notes

The flash-attention task spec called for an 81-config grid
(BLOCK_N ∈ {16, 32, 64}, num_stages ∈ {2, 3, 4}). On the GTX 1650
with one Python process per kernel, the wider grid ran the
autotuner for hours per kernel — each fresh `(M, N, D_HEAD)` key
triggers a complete sweep, and the correctness check uses different
shapes than the benchmark, so the cache cannot be shared between
stages within a single invocation. The shipped grid trims
`BLOCK_N` to `{32, 64}` and `num_stages` to `{2, 3}` (36 configs).

No configs raised a compile error during the sweep on this
hardware. The SMEM pre-filter caught all the would-be-too-big
configs at construction time.

### Correctness

All 31 correctness cells across the six kernels still pass at the
v0.2.0 tolerances:

- Softmax fp32 (4 shapes): `max_abs_err ≤ 1.12e-08`
- Softmax fp16 (4 shapes): `max_abs_err ≤ 7.63e-06`
- RMSNorm fp32 (5 shapes): `max_abs_err ≤ 1.91e-06`
- RMSNorm fp16 (5 shapes): `max_abs_err ≤ 7.81e-03`
- Flash-attention fp32 (6 cases): `max_abs_err ≤ 3.17e-08`
- Flash-attention fp16 (6 cases): `max_abs_err ≤ 3.05e-05`

### Evidence vocabulary applied

- `compiled` — yes, Triton 3.7.0 / PyTorch 2.5.1+cu121.
- `unit-tested` — correctness per kernel, fp32 at atol=1e-5 / 1e-4 /
  1e-3, fp16 at atol=1e-2.
- `benchmarked` — wall-clock timings with warmup + sync, JSON in
  `bench/0*_autotuned.json`.
- `hardware-verified` — measured on GTX 1650 Max-Q (Turing, cc 7.5).

### Non-goals (deferred to v0.4.0)

- **bf16.** Turing tensor cores lack bf16 mma; waits for Ampere or
  newer hardware.
- **fp8.** Hopper-only.
- **Flash-attention backward pass.** Forward only.
- **Fused-RMSNorm-Linear** (single-kernel norm + matmul). Turing's
  64 KB SMEM/SM still constrains tile sizing tightly.
- **TF32-on / TF32-off reference comparison columns.** The fp16
  comparison column addresses the v0.1 "cuBLAS uses TF32"
  observation by moving both sides to the same tensor-core class.
- **ROCm / AMD path.** NVIDIA Turing only.
- **Persistent on-disk autotune cache across processes.** Triton's
  `Autotuner` exposes `save_path` / `restore_value` hooks that a
  future revision could wire up to skip the cold-start sweep when
  the same shape was tuned in a previous run.
- **Widening the flash-attention grid** to include `BLOCK_N=16`
  and `num_stages=4`. The current 36-config grid finds the
  v0.2.0 hand-tuned pick automatically; if a future winner lands
  on the boundary the grid should be widened.

## v0.2.0 — 2026-05-19

Three fp16 kernel variants added alongside the v0.1.0 fp32 set. The
flash-attention fp16 variant routes both matmul steps through Turing's
fp16 tensor-core MMA (m16n8k16) via `tl.dot` with an fp32 accumulator;
this resolves the v0.1.0 honest non-victory (fp32 SIMT lost 2–3× to
torch's TF32 reference on Turing).

### Added

- `kernels/01b_fused_softmax_fp16.py` — row-wise numerically-stable
  softmax with fp16 input/output and fp32 reduction. Correctness vs
  `torch.softmax(x.half())` at `atol=1e-2, rtol=1e-2`,
  `max_abs_err ≤ 7.63e-06` across 4 shapes. Benchmark sweep over
  cols=128..16384 with rows=4096. Best **2.88×** at cols=2048;
  approximately at parity (1.00–1.05×) at cols ≥ 4096 where both
  Triton and torch saturate bandwidth. Loses at cols ∈ {128, 256}
  (0.39×, 0.43×) — the small-column regime is small enough that
  kernel launch overhead dominates and torch's launcher wins.
  Bandwidth ~96 GB/s on GTX 1650.

- `kernels/02b_fused_rmsnorm_fp16.py` — single-pass RMSNorm with
  fp16 input/weight/output and fp32 reduction. Correctness vs
  `torch.nn.functional.rms_norm` (fp16) at `atol=1e-2, rtol=1e-2`,
  `max_abs_err ≤ 7.81e-03` across 5 shapes. Consistent **3.6–4.2×
  speedup** at cols ≥ 512 vs torch's two-pass fp16 reference.
  Bandwidth ~96 GB/s.

- `kernels/03b_flash_attention_fp16.py` — FA-v2 forward kernel with
  optional causal mask, fp16 Q/K/V/O, fp32 accumulators for the
  online-softmax state (m_i, l_i, o_acc). Both `tl.dot` calls
  (Q@K^T and P@V) use fp16 inputs with `out_dtype=tl.float32`,
  routing through Turing's m16n8k16 mma. Correctness vs
  `torch.nn.functional.scaled_dot_product_attention` (fp16) at
  `atol=1e-2, rtol=1e-2`, `max_abs_err ≤ 3.05e-05` across 6
  configurations.

  Benchmark sweep B=1, H=8, D=64, causal=True over M=128..2048:

  | M | triton us | torch us | vs torch | vs fp32 Triton |
  |---|---|---|---|---|
  | 128 | 225 | 106 | 0.47× | 1.07× |
  | 256 | 309 | 399 | 1.29× | 2.44× |
  | 512 | 894 | 1100 | 1.23× | 2.77× |
  | 1024 | 1854 | 2032 | 1.10× | 4.43× |
  | 2048 | 6904 | 7261 | 1.05× | 4.66× |

  The v0.1.0 pre-registered claim that "fp16 + tensor-cores closes
  the gap" holds at M ≥ 256; v0.2.0 modestly beats torch's reference
  at every M from 256 upward, and runs 2.4–4.7× faster than the
  fp32 Triton kernel in this repo (the closes-the-gap result).
  At M=128, torch SDPA's small-sequence path is still faster; we
  document the loss honestly.

- `bench/01b_fused_softmax_fp16.json`,
  `bench/02b_fused_rmsnorm_fp16.json`,
  `bench/03b_flash_attention_fp16.json` — saved benchmark results
  from the v0.2.0 sweep.

### Tuning note (Turing-specific)

The fp32 kernel's launch configuration (`BLOCK_M=BLOCK_N=64`,
`num_warps=4`) is *catastrophic* on the fp16 path — measured ~6×
slower than necessary at M=512 in an offline sweep. The shipped
fp16 flash-attention uses `BLOCK_M=BLOCK_N=32`, `num_warps=4`,
`num_stages=2`. The fp16 mma instruction has different occupancy /
register-pressure characteristics than the fp32 SIMT FMA path; tile
sizes do not transfer across precisions. This is what real
`@triton.autotune` would fix and is the strongest argument for
shipping that lane next.

### Evidence vocabulary applied (unchanged from v0.1.0)

- `compiled` — yes, Triton 3.7.0 / PyTorch 2.5.1+cu121.
- `unit-tested` — correctness per kernel, fp32 at atol=1e-5/1e-4/1e-3,
  fp16 at atol=1e-2.
- `benchmarked` — wall-clock timings with warmup + sync, JSON in
  `bench/`.
- `hardware-verified` — measured on GTX 1650 Max-Q (Turing, cc 7.5).

### Non-goals (deferred to v0.3.0)

- **bf16.** Turing's tensor cores lack bf16 mma; bf16 path waits for
  Ampere or newer hardware.
- **fp8.** Hopper-only.
- **`@triton.autotune`** over a config grid. The shipped fp16 launch
  params are hard-coded picks from an offline sweep and will not
  transfer to other hardware. Autotune is the right answer.
- **Flash-attention backward pass.**
- **Fused RMSNorm-Linear** (single-kernel norm + matmul).
- **TF32-on / TF32-off reference comparison columns** (deferred from
  v0.1.0; partially addressed by the fp16 comparison which puts both
  implementations on the same tensor-core class).
- **ROCm / AMD path.**

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
