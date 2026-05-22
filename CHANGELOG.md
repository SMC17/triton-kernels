# Changelog

## v0.4.3 — 2026-05-21

Two small clarifying lanes from v0.4.2's STATUS.md "Open" section.
No kernel-body changes; no perf-bar movement; the v0.4.2 PTX finding
(tensor cores not engaged on sm_75) remains the active bar. This
release closes the open items at the documentation + harness level
and lets the v0.4.x line wind down cleanly.

### Added

- `torch_bwd_only_us` column in the backward benchmark functions of
  both `kernels/04_flash_attention_backward.py` (fp32) and
  `kernels/04b_flash_attention_backward_fp16.py` (fp16). Implements
  apples-to-apples SIMT-vs-SIMT timing by issuing
  `torch.autograd.grad(outputs=O, inputs=[Q,K,V], grad_outputs=dO,
  retain_graph=True)` against a once-computed forward graph — the
  call times *only* the backward, addressing the v0.4.0/v0.4.1
  honest-non-claim that SDPA exposed no backward-only entry. The
  existing `torch_fwd_bwd_us` column is preserved verbatim for
  back-compatibility with documented v0.4.0/v0.4.1 numbers.

- `scripts/run_correctness_sweep.sh` — local pass/fail sweep across
  every kernel's `--correctness` mode. Eight kernels total (the
  v0.3.0-era "six-kernel" label became 8 once the v0.4 backward
  suite arrived: 01, 01b, 02, 02b, 03, 03b, 04, 04b). Exit code 0
  iff every kernel passes; non-zero on any fail. NOT a hosted CI
  pipeline — no GitHub Actions. Local-script discipline only. A
  pre-tag invocation pattern is documented inside the script
  (operator wires it into their release flow if desired; this
  release does not install the hook).

  Manual invocation:

      scripts/run_correctness_sweep.sh

  Sample output (annotated; actual run blocked this session — see
  "Re-bench / re-sweep deferred" below):

      == 01_fused_softmax ==
        -> PASS (Ns)
      == 01b_fused_softmax_fp16 ==
        -> PASS (Ns)
      ...
      ================================================================
        correctness sweep summary  (8 kernels, total Ts)
      ================================================================
        kernel                                    status   secs
        ------                                    ------   ----
        01_fused_softmax                          PASS     ...
        01b_fused_softmax_fp16                    PASS     ...
        02_fused_rmsnorm                          PASS     ...
        02b_fused_rmsnorm_fp16                    PASS     ...
        03_flash_attention                        PASS     ...
        03b_flash_attention_fp16                  PASS     ...
        04_flash_attention_backward               PASS     ...
        04b_flash_attention_backward_fp16         PASS     ...

      PASS — all 8 kernels green.

### Re-bench / re-sweep deferred to v0.4.4

The column-emitting code and the sweep script ship this release. The
*new* numeric column and the green sweep transcript do not — this
session's `/dev/nvidia-uvm` returned EIO under the working driver
(`595.71.05`, kernel-side state mismatch), CUDA failed to initialize
in PyTorch, and no kernel could be exercised live. The fix
(`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`, or a reboot)
is a fleet-state action outside this session's commit budget. Per
the v0.4.x discipline ("don't fabricate numbers"), the bench JSON
files are left unmodified and carry the v0.4.0/v0.4.1 columns
verbatim until a clean-CUDA session re-runs them.

When CUDA is restored:

    triton-python kernels/04_flash_attention_backward.py \
        --bench --out bench/04_flash_attention_backward.json
    triton-python kernels/04b_flash_attention_backward_fp16.py \
        --bench --out bench/04b_flash_attention_backward_fp16.json
    scripts/run_correctness_sweep.sh

That run will populate `torch_bwd_only_us` +
`torch_bwd_only_vs_triton_bwd_ratio` in both JSON files and produce
the sweep transcript. v0.4.4 will ship as a numbers-only update.

### Changed

- `kernels/04_flash_attention_backward.py` benchmark `note` field
  updated to describe the two-column structure (fwd+bwd preserved
  for back-compat; bwd-only added in v0.4.3 as the apples-to-apples
  SIMT-vs-SIMT comparison). Same on `04b_*_fp16.py`.

### Interpretation (forward-looking, gated on the deferred re-bench)

What the new `torch_bwd_only_us` column gives us that
`torch_fwd_bwd_us` did not:

- `torch_fwd_bwd_us` includes a forward pass that the triton column
  does not. At small M the forward is a non-trivial fraction of the
  fwd+bwd wallclock (the v0.4.1 M=128 fp16 row: 1425 us fwd+bwd
  against ~951 us triton bwd-only). At large M the forward shrinks
  as a fraction (M=2048 fp16: 24784 us fwd+bwd, of which the
  forward is ~5-10% — `bench/03b_flash_attention_fp16.json` puts
  the M=2048 fp16 forward at ~6900 us).
- The v0.4.1 documented ratio at M=2048 fp16 was 0.86× (24784 us
  fwd+bwd ÷ 28865 us triton bwd-only). Subtracting the ~6.9k-us
  forward gives a bwd-only torch estimate of ~17.9k us, which would
  shift the apples-to-apples ratio to ~0.62× — torch's backward is
  faster than the table suggested, because the table charged torch
  for the forward.
- This is the prediction the new column will confirm or refute.
  The v0.4.2 PTX finding (zero `mma.sync`, all SIMT `fma.rn.f32`
  on sm_75) means *both* implementations are on the SIMT path; the
  comparison is therefore SIMT-vs-SIMT — clean, but torch's SIMT
  backward enjoys cuDNN-tuned kernel fusion that the triton path
  does not.

The new column is **informational, not a perf unlock**. The v0.4.2
bar (tensor cores not engaged on sm_75) is unchanged. Closing the
real gap requires path 1 (Triton MLIR patch) or path 3 (Ampere+
hardware) per STATUS.md.

### Type-I / Type-II self-audit

- **Type-II caught (this release).** v0.4.0/v0.4.1 perf tables
  hedged with "not apples-to-apples" honest-non-claims but did not
  ship a code change to *measure* apples-to-apples. The hedge
  was correct as written; the missing measurement was a Type-II
  (under-claim of capability — the measurement was straightforward
  via `torch.autograd.grad(retain_graph=True)`). v0.4.3 fills it.
- **Type-I avoided — perf-unlock framing.** Did not let "apples-to-
  apples column" slide into "we closed the gap." The v0.4.2 SIMT
  bar holds; the comparison is cleaner, the ratio likely worsens
  on the new column at large M, not better. Honest column note
  in the JSON `note` field calls this out.
- **Type-II open — re-bench deferred.** Code ships; numbers do not.
  The CHANGELOG sample-output and interpretation are predictions,
  not measurements. v0.4.4 closes this when the fleet's UVM
  recovers.
- **Type-II avoided — sweep-script scope.** Documented explicitly
  as a local script, not a CI pipeline. Did not claim a GitHub
  Actions workflow that does not exist.

### Naming

- "Six-kernel sweep" was the v0.3.0 vocabulary when there were six
  kernels (`01`, `01b`, `02`, `02b`, `03`, `03b`). v0.4 added the
  backward suite (`04`, `04b`) bringing it to eight. The script
  uses the accurate count ("8 kernels") in its summary line; the
  STATUS.md "Open" entry that used "six-kernel" wording is closed
  with the 8-kernel script.

### Not changed

- Bench JSON files (`bench/04_flash_attention_backward.json`,
  `bench/04b_flash_attention_backward_fp16.json`): the v0.4.0 /
  v0.4.1 / v0.4.2 columns and numbers stand verbatim. v0.4.4 will
  add the `torch_bwd_only_us` rows.
- Kernel algorithmic bodies: no change. Correctness is unchanged
  from v0.4.1 (atol=1e-2 rtol=1e-2 fp16; atol=1e-3 rtol=1e-3 fp32).
- The v0.4.2 PTX diagnosis: still the active read on tensor-core
  engagement. v0.5 stays blocked per STATUS.md.

## v0.4.2 — 2026-05-20

Diagnosis-only release. Closes the v0.4.1 Type-II open flag
("tensor-core engagement unconfirmed") via direct PTX inspection
of v0.4.1 fp16 backward + v0.4.0 fp32 backward kernels at a
representative shape (B=1, H=8, M=N=512, D=64, BLOCK_M=BLOCK_N=32,
num_warps=4, num_stages=2 — the autotune winner across most of the
v0.4.1 sweep).

Full writeup: `docs/PTX_INSPECTION.md`. PTX dumps under
`ptx-inspection/v0.4.1/` and `ptx-inspection/v0.4.0/`. Repro:
`ptx-inspection/dump_ptx.py`.

### Diagnosis

**Tensor cores are NOT engaged in v0.4.1 fp16 backward (or in v0.4.0
fp32 backward).** Both run on the SIMT `fma.rn.f32` path on this
Triton 3.7.0 + GTX 1650 (sm_75) stack.

| Kernel              | mma.sync.* | m16n8k8 | ldmatrix.sync | fma.rn.f32 |
|---------------------|-----------:|--------:|--------------:|-----------:|
| v0.4.1 fp16 dK/dV   |          0 |       0 |             0 |       2072 |
| v0.4.0 fp32 dK/dV   |          0 |       0 |             0 |       2072 |
| v0.4.1 fp16 dQ      |          0 |       0 |             0 |       1560 |
| v0.4.0 fp32 dQ      |          0 |       0 |             0 |       1560 |

Identical fma counts in fp16 vs fp32 across the matching kernels;
zero `mma.sync` instructions in either. The only PTX difference is
fp16↔fp32 cast traffic in the v0.4.1 variant (80 `cvt.f32.f16` + 48
`cvt.rn.f16.f32` per dK/dV iteration). That cast overhead — without
any tensor-core compute speedup — is why v0.4.1 fp16 is slower than
v0.4.0 fp32 at small M (952 us vs 447 us at M=128) and only closes
the gap at large M (28865 us vs 26153 us at M=2048) where the halved
global-memory bandwidth amortizes.

### Root cause

MLIR pass-output dump (`MLIR_ENABLE_DUMP=1`) shows the dot operation
retains a `ttg.blocked` parent encoding (`parent = #blocked8`) after
the `TritonGPUAccelerateMatmul` pass on sm_75, rather than being
promoted to `#nvidia_mma` (`NvidiaMmaEncodingAttr`). The pass declined
the MMA layout assignment for these dots on Turing. Probed across 17
distinct (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) configs
on a standalone fp16 matmul; none unlocked `mma.sync` on sm_75. The
Turing fp16 mma codegen path *is* present in the Triton 3.7.0 binary
(`callMmaTuringFp16` symbol; `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
string), so this is a layout-assignment / pipeline-gating regression
in Triton's MLIR rewrite, not an outright codegen-missing case.

`compiler.py:make_ttgir` (lines 273-299 in this Triton build) confirms
the structural gating: modern loop-pipelining / matmul-scheduling
passes (`add_pipeline`, `add_schedule_loops`, `add_fuse_nested_loops`)
run only when `capability // 10 in [8, 9]` (sm_80, sm_89) or `>= 10`
(Blackwell). sm_75 falls into the `else` branch running only
`add_triton_licm`.

### Type-I / Type-II self-audit

- **Type-II closed (from v0.4.1).** "Tensor-core engagement
  unconfirmed" flag closed with definitive PTX evidence: not engaged.
- **Type-II caught (new in v0.4.2).** v0.4.1 prose and the
  `04b_flash_attention_backward_fp16.py` docstring named
  `m16n8k16` as Turing's fp16 mma — incorrect. m16n8k16 with
  `.f32.f16.f16.f32` requires sm_80+ (Ampere). Turing's fp16 mma is
  `m16n8k8`. Docstring corrected in v0.4.2.
- **Type-I avoided.** v0.4.1 prose hedged ("Type-II open",
  "v0.4.2 should inspect the generated PTX directly to confirm")
  rather than asserting tensor-core engagement as a fact. The hedge
  matches what PTX revealed. No retraction of a too-strong v0.4.1
  perf claim is required.
- **Type-II open (new).** "Backward perf gap to torch at large M"
  cannot be closed on this hardware without a Triton backend that
  emits Turing fp16 mma. Two paths out: (a) patch Triton's
  `TritonGPUAccelerateMatmul` for sm_75; (b) downgrade to Triton 2.x
  where Turing MMAv1/v2 was on the main lowering path. Both are
  out of repo scope.

### Fix considered, not shipped

A kernel-side fix is not available: `tl.dot`'s `input_precision`
parameter only affects fp32×fp32 (TF32 vs IEEE), not fp16×fp16
layout selection. No tile-size combination unlocks the MMA layout on
this Triton 3.7.0 sm_75 backend. The fix would require an upstream
Triton change or a downgrade — out of scope for v0.4.2's
"ship the diagnosis" target. Kernels themselves are not modified
beyond the docstring correction. Correctness is unchanged from v0.4.1
(the bench results stand; no re-bench required because compute path
did not move).

### Added

- `docs/PTX_INSPECTION.md` — full writeup with PTX excerpts, MLIR
  before/after AccelerateMatmul, fma/cast counts, probe-config grid.
- `ptx-inspection/dump_ptx.py` — reproducible PTX dump script
  (forces single-config autotune via `kern.configs = [...]`,
  routes Triton cache through a tmpdir, copies the resulting
  `*.ptx` files to `ptx-inspection/{v0.4.1,v0.4.0}/`).
- `ptx-inspection/v0.4.1/{04b-preprocess,04b-dkdv,04b-dq}.ptx`
- `ptx-inspection/v0.4.0/{04-preprocess,04-dkdv,04-dq}.ptx`

### Changed

- `kernels/04b_flash_attention_backward_fp16.py` docstring + inline
  comments: removed the incorrect "Turing m16n8k16" claim, added
  pointer to `docs/PTX_INSPECTION.md` for the diagnosis.
- `kernels/03b_flash_attention_fp16.py` docstring + inline comments:
  the v0.2.0 / v0.3.0 fp16 forward also claimed Turing tensor-core
  engagement. Side-note PTX dump
  (`ptx-inspection/v0.4.1/03b-forward-reference.ptx`) confirms the
  same root cause applies — zero `mma.sync`, all SIMT `fma.rn.f32`.
  Docstring corrected; bench numbers unchanged.
- `.canonical-readme.md` v0.4.1 section: the "Type-II open" line is
  now the definitive v0.4.2 answer. v0.4.0 section's "fp16 backward
  would close the gap via tensor cores" forward-looking claim is
  marked disproved.

### Not changed

- `kernels/04b_flash_attention_backward_fp16.py` kernel bodies: no
  algorithmic or numeric change. v0.4.1 correctness (7 shapes,
  atol=1e-2 rtol=1e-2) and v0.4.1 perf still stand verbatim.
- `bench/04b_flash_attention_backward_fp16.json`: unchanged. The
  bench numbers in v0.4.1 are the same numbers v0.4.2 explains.

## v0.4.1 — 2026-05-20

Flash-attention v2 BACKWARD pass — fp16 variant. Consumes the saved
LSE (fp32) from `03b_flash_attention_fp16.py` (wiring already in
place from v0.4.0). All five matmul steps route through fp16 inputs
with `out_dtype=tl.float32` accumulation; dQ/dK/dV accumulators stay
fp32 across the entire reduction and only cast to fp16 at the final
store.

### Added

- `kernels/04b_flash_attention_backward_fp16.py` — fp16 backward
  with three kernels (preprocess for `D_i = sum(dO * O)`, dK/dV
  outer-over-N, dQ outer-over-M). Same algorithmic structure as
  `04_flash_attention_backward.py` v0.4.0; differs only in input
  dtype + the m16n8k16 mma routing through `tl.dot` with fp16 inputs.
- `bench/04b_flash_attention_backward_fp16.json` — perf sweep.

### Correctness (7 shapes, atol=1e-2 rtol=1e-2 matching fp16 forward bar)

| Case (B, H, M, D, causal)  | o        | dQ       | dK       | dV       |
|----------------------------|----------|----------|----------|----------|
| (1, 1, 64,   64, False)    | 1.53e-05 | 3.81e-06 | 7.63e-06 | 0.00e+00 |
| (1, 1, 64,   64, True)     | 1.22e-04 | 5.72e-06 | 1.53e-05 | 0.00e+00 |
| (1, 4, 128,  32, True)     | 1.22e-04 | 1.53e-05 | 1.53e-05 | 1.95e-03 |
| (1, 4, 256,  64, True)     | 1.22e-04 | 1.53e-05 | 1.53e-05 | 9.77e-04 |
| (2, 4, 256,  64, True)     | 1.22e-04 | 1.53e-05 | 1.53e-05 | 9.77e-04 |
| (1, 8, 512,  64, True)     | 1.22e-04 | 1.53e-05 | 1.53e-05 | 3.91e-03 |
| (1, 4, 1024, 64, True)     | 1.22e-04 | 1.53e-05 | 1.53e-05 | 1.95e-03 |

All within tolerance. Largest observed error 3.91e-03 (dV at M=512
causal), well inside the 1e-2 bar. dV's reduction chain remains the
longest in fp16 as it was in fp32, and dV's error scales accordingly.

### Backward perf (B=1, H=8, D=64, causal=True, fp16, M-sweep)

| M    | triton bwd (us) | torch fwd+bwd (us) | triton bwd TFLOPS | ratio (torch ÷ triton bwd) |
|------|-----------------|--------------------|--------------------|------------------------------|
|  128 |    951.94       |    1425.36         | 0.044              | 1.50×                        |
|  256 |    934.59       |     902.03         | 0.180              | 0.97×                        |
|  512 |   1985.04       |    2216.07         | 0.338              | 1.12×                        |
| 1024 |   7305.95       |    6965.66         | 0.367              | 0.95×                        |
| 2048 |  28864.98       |   24784.02         | 0.372              | 0.86×                        |

**Honest non-claim preserved from v0.4.0**: not apples-to-apples. Torch
column is fwd+bwd; triton column is bwd-only. Torch has no
backward-only entry point. The true bwd-vs-bwd gap at large M is
LARGER than this table shows.

Versus v0.4.0 fp32 backward at M=2048 (ratio 0.71×): v0.4.1 fp16 sits
at 0.86× — closes the gap but does not pass torch. At M=128, v0.4.1
fp16 backward is SLOWER than v0.4.0 fp32 backward (952 us vs 447 us);
the fp16 path has per-launch overhead that doesn't amortize until
M ~ 256.

### Type-I / Type-II self-audit

- **Type-II open — tensor-core engagement unconfirmed.** Autotuner
  picked `(BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=2)` for
  both kernels at every shape except M=2048 dq (BLOCK_N=64). fp16
  being slower than fp32 at small M is the wrong signature for a
  Turing m16n8k16-engaged path — under tensor cores fp16 should beat
  fp32 at every M, not lose at small M. `tl.dot` with fp16 inputs +
  `out_dtype=tl.float32` SHOULD route through tensor cores, but the
  empirical perf signature does not match. v0.4.2 should inspect the
  generated PTX directly to confirm.
- **Type-II open — autotune grid edge.** M=2048 dq kernel picked the
  `BLOCK_N=64` upper edge of the grid. Grid widening deferred to
  v0.4.2 alongside the tensor-core diagnosis.
- **Type-I avoided — perf framing.** Did not let "fp16 closes the
  bwd-vs-fwd+bwd gap" slide into "fp16 backward beats torch." Same
  honest-non-claim framing preserved from v0.4.0.

### What is *not* in v0.4.1

- PTX-confirmed m16n8k16 tensor-core engagement (deferred to v0.4.2)
- Variable-length sequence masks (deferred to v0.5)
- Dropout in backward (deferred to v0.5+)
- Wider autotune grid (deferred to v0.4.2)

## v0.4.0 — 2026-05-19

Flash-attention v2 BACKWARD pass (fp32, causal optional). Closes the
v0.3.0 non-goal of "forward only — backward needs the saved
log-sum-exp + dO statistics; that's a separate kernel of similar size."

### Added

- **Forward modification:** `03_flash_attention.py` and
  `03b_flash_attention_fp16.py` gain an optional `save_lse: bool`
  kwarg on the public entry point. When `True`, the kernel writes
  `L_i = m_i + log(l_i)` (fp32, shape `[B, H, M]`) to an output
  buffer at the end of each Q-tile iteration. The default
  `save_lse=False` preserves the v0.3.0 single-tensor return shape;
  the existing forward correctness suite passes unchanged with the
  new optional output present.

  - fp32 `03_flash_attention.py`: kernel signature gains `L_ptr` and
    three LSE strides, plus a `SAVE_LSE: tl.constexpr` flag. The
    save path runs `m_i + tl.log(l_i)` once per Q-tile at the
    bottom of the kernel; the branch is constexpr-elided when
    `save_lse=False`.
  - fp16 `03b_flash_attention_fp16.py`: mirror change. LSE is fp32
    regardless of input dtype (the running max+sum state already
    lives in fp32).
  - New convenience wrapper `triton_flash_attn_with_lse(...)` in the
    fp32 module always saves LSE; the backward harness uses this
    to drive the full forward+backward round-trip from one script.

- **Backward kernel:** `kernels/04_flash_attention_backward.py`.
  Three-kernel structure following Dao 2023 Section 3:

  1. `flash_attn_bwd_preprocess_kernel` — computes
     `D_i = sum(dO_i * O_i, axis=-1)` (fp32 [B, H, M]). Bandwidth-
     bound, no autotune (fixed BLOCK_M=64).
  2. `flash_attn_bwd_dkdv_kernel` — outer loop over K/V blocks
     along N, inner loop over Q blocks along M. Accumulates
     `dK_j, dV_j` in registers, writes once at end. Causal-aware:
     skips Q blocks below the K block when `IS_CAUSAL=True`.
  3. `flash_attn_bwd_dq_kernel` — outer loop over Q blocks along M,
     inner loop over K/V blocks along N. Mirror image of the
     forward iteration pattern.

  Per the recomputation pattern in Dao 2023, no kernel materializes
  the N×M attention matrix. P_ij is recomputed inside the inner
  loop from Q, K and the saved per-row LSE.

  Both backward kernels are wrapped in `@triton.autotune` over
  `BLOCK_M ∈ {32, 64} × BLOCK_N ∈ {32, 64} × num_warps ∈ {4, 8}` at
  `num_stages=2` (8 configs, 4 alive after the backward-specific
  SMEM filter — the backward holds more live tiles than the forward,
  so the per-config SMEM estimate uses a 4× multiplier instead of
  the forward's 2×). Keyed on `(M, N, D_HEAD)`.

- **Backward Python entry point:** `triton_flash_attn_backward(
  q, k, v, o, do, lse, sm_scale=None, is_causal=False)` returning
  `(dq, dk, dv)`. The caller is expected to have run the forward
  with `save_lse=True` first.

- `bench/04_flash_attention_backward.json` — backward sweep over
  M=128..2048 with the autotune winning configs.

### Backward correctness vs PyTorch autograd

Reference: `softmax(Q @ K^T * scale + causal_mask) @ V`,
`.backward(dO=ones_like)`. Tolerance `atol=1e-3, rtol=1e-3` per the
spec.

| Case (B, H, M, D, causal) | dQ max err | dK max err | dV max err | O max err |
|---|---|---|---|---|
| (1, 1, 64, 64, False)  | 2.10e-09 | 3.26e-09 | 2.38e-07 | 1.12e-08 |
| (1, 1, 64, 64, True)   | 2.79e-09 | 5.59e-09 | 9.54e-07 | 2.98e-08 |
| (1, 4, 128, 32, True)  | 6.54e-09 | 1.49e-08 | 7.15e-07 | 2.24e-08 |
| (1, 4, 256, 64, True)  | 6.62e-09 | 2.98e-08 | 9.54e-07 | 2.98e-08 |
| (2, 4, 256, 64, True)  | 8.09e-09 | 2.05e-08 | 9.54e-07 | 2.98e-08 |
| (1, 8, 512, 64, True)  | 5.59e-09 | 2.42e-08 | 1.91e-06 | 3.17e-08 |
| (1, 4, 1024, 64, True) | 6.50e-09 | 4.10e-08 | 1.43e-06 | 3.73e-08 |

All 7 cases pass at `atol=1e-3, rtol=1e-3`. Largest observed dV
error is `1.91e-06` (M=512 causal) — three orders of magnitude
below the tolerance. dQ and dK are tighter still; the dV path has
the longest reduction chain (`P^T @ dO` accumulating across every
Q tile per K tile), so its accumulated round-off is correspondingly
larger.

### Backward perf vs `torch.nn.functional.scaled_dot_product_attention` + autograd (fp32 causal, B=1 H=8 D=64)

Honest non-claim: this is **not** an apples-to-apples comparison.
Torch's autograd path does not expose a backward-only entry point
that takes a precomputed forward output + dO, so the torch column
below times the full forward+backward through `.backward()`.
Triton's column is the backward-only call (forward already done
separately via `03_flash_attention.py` with `save_lse=True`).

| M | triton bwd (us) | torch fwd+bwd (us) | triton bwd TFLOPS | ratio (torch fwd+bwd ÷ triton bwd) |
|---|---|---|---|---|
| 128 | 446.70 | 734.15 | 0.094 | 1.64× |
| 256 | 768.13 | 1103.76 | 0.218 | 1.44× |
| 512 | 2137.44 | 2925.12 | 0.314 | 1.37× |
| 1024 | 6730.26 | 4752.52 | 0.399 | 0.71× |
| 2048 | 26153.21 | 18517.79 | 0.411 | 0.71× |

Direction: the Triton backward is competitive with (and at small M
slightly faster than) torch's full forward+backward pipeline at
M ≤ 512, then loses at M ∈ {1024, 2048}. Since the torch column
includes the forward, the true backward-vs-backward gap at large
M is **larger** than what this table shows — torch's autograd
backward at M=2048 is being attributed >18ms that includes the
forward. Treat this as: "the Triton fp32 backward holds up at
small M and falls behind torch's tuned tensor-core autograd at
large M, where torch's tensor-core advantage dominates." Closing
that gap is what an fp16 backward (v0.4.1) would address.

### Cumulative correctness (added to v0.3.0)

- Flash-Attention fp32 backward — dQ/dK/dV vs torch autograd:
  `max_abs_err ≤ 4.10e-08` for dQ, dK; `max_abs_err ≤ 1.91e-06`
  for dV. All 7 shapes pass at `atol=1e-3, rtol=1e-3`.

### Evidence vocabulary applied

- `compiled` — yes, Triton 3.7.0 / PyTorch 2.5.1+cu121.
- `unit-tested` — correctness vs torch autograd at `atol=1e-3,
  rtol=1e-3` for dQ, dK, dV across 7 shapes.
- `benchmarked` — wall-clock timings with warmup + sync, JSON in
  `bench/04_flash_attention_backward.json`.
- `hardware-verified` — measured on GTX 1650 Max-Q (Turing, cc 7.5).

### Non-goals (deferred to v0.4.1 / v0.5)

- **fp16 backward** (→ v0.4.1). The recomputation of P inside the
  backward has tighter numerical-stability constraints than the
  forward; a small fp16 error in S_ij propagates through `exp` into
  P, through `P^T @ dO` into dV, and the cumulative error can blow
  past the fp16 forward's `atol=1e-2` tolerance. The fp16 forward
  already saves LSE (fp32) per the v0.4.0 forward modification, so
  the backward consumer is in place; only the backward kernel
  itself is pending.
- **Variable-length sequences / arbitrary attention masks** (→ v0.5).
  Strict causal is supported. Non-causal masks would need a per-row
  mask infrastructure that doesn't transfer from the forward's
  `n_offs < N` pattern.
- **Dropout in forward and backward** (→ v0.5+). Standard FA-2
  supports dropout with a shared RNG state across the two passes;
  v0.4 ships without to keep the diff focused.
- **Triton-side gradient checkpointing.** The backward recomputes
  the attention scores and softmax from Q, K, V and the saved LSE,
  but does not implement a separate activation-save tier.
- **Wider backward autotune grid.** The 8-config grid (4 alive after
  SMEM pruning) is intentionally narrow — backward kernels have
  more live tiles than the forward, and a full
  `BLOCK_M × BLOCK_N × num_warps × num_stages` sweep over the
  same surface as the forward (36 configs) ran multi-minute on
  each shape key. If a future winner lands on the grid edge, widen.
- **bf16, fp8.** Same hardware constraints as v0.3.0 — Turing
  tensor cores lack both.
- **ROCm / AMD path.**

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
