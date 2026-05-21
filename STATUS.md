# triton-kernels — status

Honest-state register. Pre-1.0 substrate; no stability claims. The
shape of this file mirrors `fleet-sbom-index/STATUS.md` and
`sovereign-edge/STATUS.md`.

## Released

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

## Open

- **v0.4.3 candidate — measure SIMT path against torch's SIMT path.**
  v0.4.0–v0.4.2 compare triton bwd-only against torch fwd+bwd, which
  is honest but not load-bearing for the "is the kernel fast" question.
  Adding a torch-eager backward-only invocation (via custom autograd
  function exposing `.backward()` separately) would give apples-to-
  apples SIMT-vs-SIMT numbers. Small lane; clarifying not unlocking.
- **No-regression sweep across all six kernels.** v0.3.0's six fp32
  and fp16 forwards still pass per agent-spot-checks, but a CI script
  that runs every correctness suite on every commit is not in place.
  Manual confidence; not audited.
