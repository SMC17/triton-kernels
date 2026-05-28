# triton-kernels — status

Honest-state register. Pre-1.0 substrate; no stability claims. The
shape of this file mirrors `fleet-sbom-index/STATUS.md` and
`sovereign-edge/STATUS.md`.

## Released

### v0.4.3 — 2026-05-21 — apples-to-apples bench column + 8-kernel sweep script
- `torch_bwd_only_us` column added to both `04` (fp32) and `04b`
  (fp16) backward benchmark functions. Uses
  `torch.autograd.grad(outputs=O, inputs=[Q,K,V], grad_outputs=dO,
  retain_graph=True)` against a once-computed forward graph — times
  torch's backward in isolation, no SDPA backward-only entry needed.
  Existing `torch_fwd_bwd_us` column preserved for back-compat with
  the v0.4.0 / v0.4.1 documented numbers.
- `scripts/run_correctness_sweep.sh` ships the local 8-kernel
  no-regression sweep. Exit non-zero on any fail. Local script, not
  a hosted CI pipeline; pre-tag invocation pattern documented in
  the script header for operator wiring.
- Re-bench + sweep transcript deferred to v0.4.4 — this session's
  `/dev/nvidia-uvm` returned EIO under driver `595.71.05` (kernel
  module reload needed, sudo required). Code + script ship; the
  new numeric column populates on next clean-CUDA run.
- Type-II caught: v0.4.0/v0.4.1's honest-non-claim ("torch has no
  backward-only entry") under-claimed; the measurement was
  straightforward via `autograd.grad(retain_graph=True)`. Filled.

### v0.4.2 — 2026-05-21 — PTX-level diagnosis closing the tensor-core question
- PTX dumps of v0.4.1 fp16 backward + v0.4.0 fp32 backward at the
  autotune-winner shape (B=1, H=8, M=N=512, D=64, BLOCK_M=BLOCK_N=32,
  num_warps=4, num_stages=2).
- Zero `mma.sync` in either kernel; identical `fma.rn.f32` counts
  across fp16/fp32 (2072 dK/dV, 1560 dQ); v0.4.1-only delta is ~128
  `cvt.*` casts per dK/dV iteration — exactly the cast tax that
  explains M=128 fp16 losing to fp32 (952 us vs 447 us).
- Root cause traced in MLIR: `TritonGPUAccelerateMatmul` declines the
  MMA layout assignment on sm_75; dot retains `ttg.blocked` parent
  encoding rather than `#nvidia_mma`. Probed 17 (BLOCK, num_warps,
  num_stages) configs on a standalone fp16 matmul; none unlock
  `mma.sync` on sm_75.
- Confirmed Turing fp16 mma codegen IS present in Triton 3.7.0 binary
  (`callMmaTuringFp16` symbol, `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
  string) — layout-assignment / pipeline-gating regression in the
  MLIR rewrite, NOT a missing codegen path.
- Type-II caught and fixed: v0.4.1 named `m16n8k16` as Turing's fp16
  mma. Wrong. m16n8k16.f32.f16.f16.f32 requires sm_80+ (Ampere).
  Turing's flavor is `m16n8k8`. Docstrings corrected.
- Full writeup: `docs/PTX_INSPECTION.md`.

### v0.4.1 — 2026-05-20 — fp16 backward
- 7 correctness shapes pass at atol=1e-2 rtol=1e-2 (fp16 forward bar).
- Largest dV error 3.91e-03 at M=512 causal.
- Bench ratio 0.86× at M=2048 vs torch fwd+bwd; closes v0.4.0's 0.71×
  but does not surpass torch on this hardware (see v0.4.2 diagnosis).

### v0.4.0 — 2026-05-19 — fp32 backward
- 7 correctness shapes pass at atol=1e-3 rtol=1e-3.
- Optional `save_lse` flag on both fp32 and fp16 forward kernels.

### v0.1.0 — v0.3.0
- v0.1.0: fp32 fused softmax, fused rmsnorm, flash-attention forward.
- v0.2.0: fp16 variants (`01b`, `02b`, `03b`).
- v0.3.0: `@triton.autotune` across all six kernels.

## Blocked

### v0.5 — variable-length sequence masks
**Blocked on hardware × backend.** The motivation for v0.5 was always
to validate the v0.4 backward path under realistic attention masks
(varlen / packed sequences) before claiming flash-attention completeness.
v0.4.2's PTX diagnosis pulls that motivation apart:

- The v0.4.x backward on this hardware (GTX 1650 Max-Q, sm_75) runs
  entirely on the SIMT `fma.rn.f32` path, not Turing tensor cores,
  due to a Triton 3.7.0 MLIR layout-assignment regression on sm_75
  (see v0.4.2 above).
- A v0.5 varlen kernel built on the same `tl.dot` calls would inherit
  the same SIMT path. Shipping varlen on this foundation would
  cement a perf characteristic we already know is the wrong target.
- Foundation-first: do not start v0.5 until the layer beneath
  (tensor-core engagement) is solid.

**Three forward paths, none of which are this-session work:**

1. **Patch upstream Triton's `TritonGPUAccelerateMatmul` for sm_75.**
   Multi-week MLIR lane. Highest leverage — would unlock Turing
   tensor cores for every Triton user on sm_75, not just this repo.
   Candidate for the upstream-contribution program register
   (`~/upstream-targets/`). Foundation: read the existing
   AccelerateMatmul pass, compare sm_80 path against sm_75
   gating logic, write the missing layout-assignment case for
   m16n8k8 fp16. Pre-requisite: clone triton-lang/triton, build
   from source on sm_75, reproduce the layout-decline, add a
   failing test, propose the fix in a PR.
2. **Downgrade to Triton 2.x where Turing MMA was on the main
   lowering path.** Days, not weeks. Cost: gives up downstream
   ecosystem features added in 3.x (autotune extensions, recent
   `tl.dot` semantics, etc.). Effectively pins this repo to a
   legacy Triton until path 1 lands upstream. Trade-off is
   asymmetric — accepting maintenance debt to unlock the perf gate
   that path 1 would unlock cleanly later.
3. **Move the kernel work to an Ampere+ GPU (sm_80+).** No hardware
   currently in fleet. PS4/Xbox are AMD (no CUDA path). Buying an
   Ampere card is the cleanest unlock but conflicts with the
   no-cloud-rental / sovereign-hardware doctrine in the same week
   we're standing up the homelab. Defer.

**Recommended sequencing (not yet committed):**
- Path 1 as a new entry in `~/upstream-targets/` queued behind T1
  (nixpkgs CrowdSec). The MLIR patch is a high-aura-density target;
  diagnosis evidence is already in `docs/PTX_INSPECTION.md`.
- v0.5 stays parked until path 1 lands OR a path-3 unlock arrives.
- DO NOT pursue path 2 (downgrade) without a path-1 timeline.

## Bench-host CUDA path

As of 2026-05-27 the bench host's Triton venv at
`~/.cache/triton-venv-host/` is **not present** (wiped at some prior
point and not yet rebuilt). The `~/.local/bin/triton-python` wrapper
exits non-zero with:

```
triton-python: venv python not found at /home/user/.cache/triton-venv-host/bin/python
                check ~/.cache/triton-venv-host/ exists and is healthy
```

Until the venv is rebuilt and a fresh smoke-run of the eight-kernel
correctness sweep is captured, the bench numbers in `bench/` reflect
the most recent successful run on this bench host (the v0.4.x history
in `CHANGELOG.md`). Code is unchanged; the gap is the local runtime.

Out of scope for this register: rebuilding the venv. That work needs
the kernel driver state + `nvidia_uvm` reload noted in v0.4.3 above
and is a separate operator task, not a kernel-side change.

## Continuous integration

`.github/workflows/test.yml` runs on every push and PR. It exercises
what a CPU-only GitHub-hosted runner can verify:

- `ruff check` across `kernels/`, `ptx-inspection/`, `tests/` with a
  documented ignore-set (autotune-lambda grids, intentional benchmark
  timing-loop sinks, paper-canonical variable names).
- `python -m py_compile` against every kernel file to catch syntax
  regressions.
- `pytest tests/` — the smoke-test contract in `tests/test_smoke.py`
  verifies kernel inventory, public-API entry-point names, and
  autotune-grid helper presence.

What CI CANNOT do on free hosted runners (substrate-honest limitation):

- Compile any kernel. Triton's MLIR/LLVM backends need a CUDA runtime;
  free GitHub runners are CPU-only.
- Exercise the bench harnesses or the local 8-kernel sweep.
- Validate benchmark JSON against a fresh run.

Closing that gap would need either (a) a self-hosted CUDA-capable
runner, or (b) a parallel runner on a hosted CUDA service. Neither is
in scope for this artifact; STATUS.md remains the source of truth for
the bench-host run record.

## Open

- **v0.4.4 — populate `torch_bwd_only_us` numbers.** Code shipped in
  v0.4.3; numbers blocked on a clean-CUDA session (driver `595.71.05`
  + `nvidia_uvm` reload). Once unblocked, re-run benches at the same
  shapes (B=1, H=8, D=64, causal=True, M ∈ {128, 256, 512, 1024, 2048})
  and refresh `bench/04_flash_attention_backward.json` +
  `bench/04b_flash_attention_backward_fp16.json` with the new column.
  Bench JSON files left unmodified in v0.4.3 to preserve v0.4.x
  numeric history.
- **8-kernel sweep transcript capture.** Script shipped in v0.4.3;
  the actual `PASS / ... / PASS` transcript is gated on the same
  CUDA recovery. Once captured, paste into v0.4.4's CHANGELOG
  sample-output block in place of the v0.4.3 annotated template.
