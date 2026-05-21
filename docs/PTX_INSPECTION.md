# PTX inspection — v0.4.1 fp16 backward vs v0.4.0 fp32 backward

Triton 3.7.0 + CUDA 12.1 + GTX 1650 Max-Q (Turing, sm_75).

## Question

v0.4.1 left an open Type-II flag: the fp16 backward was *slower* than
the v0.4.0 fp32 backward at small M (951.94 us vs 446.70 us at M=128)
and only closed the gap at large M (28865 us vs 26153 us at M=2048,
ratio 1.10×). That is the wrong signature for a Turing tensor-core-
engaged path — under tensor cores fp16 should beat fp32 at every M,
not lose at small M. The v0.4.1 doc claimed all five matmul steps
route through fp16 mma via `tl.dot(fp16, fp16, out_dtype=fp32)`. PTX
inspection settles whether that claim is true.

## Method

Dumped PTX for both backward variants at a representative shape
(B=1, H=8, M=N=512, D=64, BLOCK_M=BLOCK_N=32, num_warps=4,
num_stages=2 — the autotune winner across most of the v0.4.1 sweep).
PTX written to `ptx-inspection/v0.4.1/` and `ptx-inspection/v0.4.0/`.
Driver script: `ptx-inspection/dump_ptx.py`.

For each kernel, counted occurrences of:

- `mma.sync.aligned.m16n8k16` — sm_80+ fp16 mma (Ampere+ only)
- `mma.sync.aligned.m16n8k8` — sm_75+ fp16 mma (Turing-supported)
- `mma.sync.aligned.m8n8k4` — sm_70+ fp16 mma (Volta legacy)
- `wmma.mma.sync` — older wmma intrinsic
- `ldmatrix.sync.aligned` — tensor-core shared-mem load
- `fma.rn.f32` — SIMT fp32 fused multiply-add (no tensor cores)

## Result

| Kernel                                | mma.sync.* (any) | m16n8k16 | m16n8k8 | m8n8k4 | wmma | ldmatrix | fma.rn.f32 |
|---------------------------------------|-----------------:|---------:|--------:|-------:|-----:|---------:|-----------:|
| **v0.4.1 fp16** preprocess            | 0                | 0        | 0       | 0      | 0    | 0        | 28         |
| **v0.4.1 fp16** dK/dV                 | 0                | 0        | 0       | 0      | 0    | 0        | 2072       |
| **v0.4.1 fp16** dQ                    | 0                | 0        | 0       | 0      | 0    | 0        | 1560       |
| **v0.4.0 fp32** preprocess            | 0                | 0        | 0       | 0      | 0    | 0        | 24         |
| **v0.4.0 fp32** dK/dV                 | 0                | 0        | 0       | 0      | 0    | 0        | 2072       |
| **v0.4.0 fp32** dQ                    | 0                | 0        | 0       | 0      | 0    | 0        | 1560       |

Identical `fma.rn.f32` counts in fp16 vs fp32 for the dK/dV and dQ
kernels. Zero `mma.sync` instructions in either. Both variants are
running on the SIMT fp32 fma path — neither engages Turing tensor
cores.

Sample (v0.4.1 dK/dV kernel, kernel inner loop):

```ptx
fma.rn.f32 	%r617, %r233, %r489, 0f00000000;
fma.rn.f32 	%r618, %r234, %r490, %r617;
fma.rn.f32 	%r619, %r235, %r491, %r618;
fma.rn.f32 	%r620, %r236, %r492, %r619;
```

These are scalar SIMT fma chains, exactly as in the v0.4.0 fp32 dK/dV
kernel — no tensor-core mma, no ldmatrix.

The only difference between v0.4.1 and v0.4.0 PTX is the fp16↔fp32
cast traffic:

| Kernel              | cvt.f32.f16 | cvt.rn.f16.f32 |
|---------------------|------------:|---------------:|
| v0.4.1 fp16 dK/dV   |          80 |             48 |
| v0.4.0 fp32 dK/dV   |           0 |              0 |

128 extra cast instructions per dK/dV kernel iteration. That is the
load-bearing perf overhead the fp16 variant pays *without* any
tensor-core upside.

## Diagnosis

**The v0.4.1 doc's claim that `tl.dot` with fp16 inputs + `out_dtype=tl.float32`
routes through Turing m16n8k16 fp16 mma is wrong on two counts:**

1. **Wrong instruction name.** `mma.sync.aligned.m16n8k16` with
   `.f32.f16.f16.f32` requires **sm_80+ (Ampere)**. Turing's
   fp16 tensor-core mma is `mma.sync.aligned.m16n8k8` and (legacy)
   `mma.sync.aligned.m8n8k4`. The v0.4.1 source comments name an
   instruction that does not exist on this hardware.

2. **No Turing fp16 mma is emitted either.** Even the correct
   `m16n8k8.f32.f16.f16.f32` (which the Triton 3.7.0 backend does
   contain — verified by string-grep on `libtriton.so`) is never
   reached. Across every tile-size config we probed
   (`(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)` ∈
   {(16, 16, 16, 4, 2), (32, 32, 16, 4, 2), (32, 32, 32, 4, 2),
   (32, 64, 32, 4, 2), (32, 64, 64, 4, 3), (64, 64, 16, 4, 2),
   (64, 64, 16, 8, 2), (64, 64, 32, 1, 2), (64, 64, 32, 4, 2),
   (64, 64, 64, 4, 2), (64, 128, 32, 4, 2), (64, 128, 64, 8, 2),
   (128, 64, 16, 4, 2), (128, 64, 32, 4, 2), (128, 128, 16, 8, 2),
   (128, 128, 32, 8, 2), (128, 256, 64, 8, 2)}), Triton 3.7.0
   compiled the same `tl.dot(fp16, fp16, out_dtype=fp32)` to SIMT
   `fma.rn.f32` chains. Tile-size knobs do not unlock tensor cores
   on this backend.

### Root cause — Triton 3.7.0 sm_75 MMA layout decline

MLIR pass-output dump (via `MLIR_ENABLE_DUMP=1`) shows the dot
operation result type *before* and *after* the `TritonGPUAccelerateMatmul`
pass on sm_75:

```mlir
// Before TritonGPUAccelerateMatmul:
%acc = tt.dot %a, %b, %acc_in, inputPrecision = tf32
  : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>

// After TritonGPUAccelerateMatmul:
%acc = tt.dot %a_dotop, %b_dotop, %acc_in, inputPrecision = tf32
  : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked8}>>
  * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked8}>>
  -> tensor<64x64xf32, #blocked8>
```

The dot result retains `parent = #blocked8` (a `ttg.blocked` layout)
rather than getting promoted to `#nvidia_mma` (`NvidiaMmaEncodingAttr`).
With a `#blocked` parent layout, the subsequent lowering through
`add_lower_mma` produces SIMT fma rather than `mma.sync`. The
AccelerateMatmul pass declined to promote on sm_75.

The Triton 3.7.0 compiler.py confirms the structural gating
(`make_ttgir`, lines 273-299): the modern loop-pipelining / matmul-
scheduling passes run only when `capability // 10 in [8, 9]` (sm_80,
sm_89) or `capability // 10 >= 10` (Blackwell). sm_75 (Turing)
falls into the `else` branch that runs only `add_triton_licm`. The
`add_accelerate_matmul` pass *does* still run for sm_75, but with
Turing's reduced post-promotion pipeline (no `add_pipeline`,
no `add_schedule_loops`), the pass appears to consistently
fall back to a `#blocked` layout instead of `#nvidia_mma`.

The Turing fp16 mma codegen path is present in the binary
(symbol `callMmaTuringFp16`, string
`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`), so this is a
layout-assignment / pipeline gating regression in Triton's MLIR
rewrite, not an outright codegen-missing case. The path is reachable
in principle but is not selected from kernel-Python on this backend
version.

### Why does fp16 close the gap to fp32 at large M but not win?

PTX shows fp16 and fp32 do the *same* number of `fma.rn.f32` ops
per kernel — identical 2072 in dK/dV, 1560 in dQ. The fma counts
are tile-size-driven, not dtype-driven, because the operation
running on the SIMT path operates on fp32-extended operands either
way.

The fp16 variant pays:

- **Cast overhead.** 80 `cvt.f32.f16` + 48 `cvt.rn.f16.f32` per dK/dV
  iteration on top of the same fma count. At small M (low fma count
  per launch), the cast tax dominates the runtime difference.
- **Three-kernel launch overhead.** Same as fp32 (both variants launch
  preprocess + dK/dV + dQ), so this isn't a v0.4.1-specific tax — but
  combined with the cast tax above, fp16's per-instruction cost is
  strictly higher than fp32's.

The fp16 variant gains:

- **Halved global-memory bandwidth on Q, K, V, dO, dQ, dK, dV.** At
  large M the kernel becomes more bandwidth-bound; the halved I/O
  cost shows up as the closing ratio (M=2048: fp32 26153 us → fp16
  28865 us is a 10% loss vs fp32; M=128: 447 us → 952 us is a 113%
  loss). The bandwidth savings amortize as M grows, but they cannot
  *beat* fp32 without tensor-core compute speedup, only narrow.

So the observed signature — fp16 loses big at small M, closes the
gap at large M, never wins — is exactly what we'd expect from a
backend that produces SIMT compute for both variants. Tensor cores
would have flipped fp16 to a win at every M, with the gap widening
(not closing) as M grows. PTX confirms no tensor cores.

## Verdict

- **Type-II open from v0.4.1 — CLOSED.** Tensor cores are *not*
  engaged in the v0.4.1 fp16 backward. The original doc claim was
  wrong on both the instruction name (`m16n8k16` requires sm_80+;
  Turing supports only `m16n8k8` for fp16) and the engagement itself
  (Triton 3.7.0's `TritonGPUAccelerateMatmul` pass declines the MMA
  layout promotion on sm_75, leaving the dot in a blocked layout that
  lowers to SIMT fma).
- **Type-I avoided.** v0.4.1 prose said "Type-II open" and "v0.4.2
  should inspect the generated PTX directly to confirm." That hedging
  language matches what the PTX revealed; the doc didn't overclaim
  tensor-core engagement as a fact.
- **Type-II caught.** The v0.4.1 doc *did* incorrectly name
  `m16n8k16` as Turing's fp16 mma instruction. Corrected here:
  Turing's fp16 mma is `m16n8k8`. v0.4.2 strikes the wrong instruction
  name from `kernels/04b_flash_attention_backward_fp16.py` docstring.

## Fix considered, not shipped

A kernel-side fix is not available. `tl.dot` has no `input_precision`
mode that affects fp16×fp16 layout selection (the parameter applies
only to fp32×fp32). No tile-size combination we probed unlocks the
MMA layout on this backend version. The fix would require either:

1. Patching `TritonGPUAccelerateMatmul` to assign `#nvidia_mma` on
   sm_75 — an upstream Triton change.
2. Downgrading to Triton 2.x where Turing MMAv1/v2 was on the main
   lowering path. Out of scope: Triton 2.x lacks features our other
   v0.x kernels depend on.

Both are out of scope for v0.4.2's "ship the diagnosis" target. The
kernels themselves are not modified.

## Side note — v0.3.0 fp16 forward is affected too

The same Triton 3.7.0 sm_75 layout-decline behavior affects the
v0.3.0 fp16 forward kernel. PTX dumped at the same shape
(B=1, H=8, M=N=512, D=64, BLOCK_M=BLOCK_N=32, num_warps=4) shows:

| Kernel                                  | mma.sync.* | ldmatrix.sync | fma.rn.f32 |
|-----------------------------------------|-----------:|--------------:|-----------:|
| v0.3.0 fp16 forward (`03b`)             |          0 |             0 |       1028 |

Zero tensor-core instructions. The v0.2.0 / v0.3.0 docstring claim
that the fp16 forward "routes through Turing's m16n8k16 / m16n8k8
fp16 tensor-core MMA instructions" is not borne out by the PTX on
this Triton + sm_75 stack. The kernel's perf numbers in
`bench/03b_flash_attention_fp16_autotuned.json` reflect SIMT compute
+ halved global bandwidth + the autotune sweep — not tensor cores.

The v0.3.0 fp16 forward docstring is corrected in v0.4.2 to point at
this document. The bench numbers themselves are unchanged; they
just no longer carry an unjustified tensor-core implication.

PTX: `ptx-inspection/v0.4.1/03b-forward-reference.ptx`.

## Files

- `ptx-inspection/dump_ptx.py` — repro script
- `ptx-inspection/v0.4.1/04b-preprocess.ptx`
- `ptx-inspection/v0.4.1/04b-dkdv.ptx`
- `ptx-inspection/v0.4.1/04b-dq.ptx`
- `ptx-inspection/v0.4.1/03b-forward-reference.ptx` — v0.3.0 fp16
  forward at matching shape, for the side-note table above
- `ptx-inspection/v0.4.0/04-preprocess.ptx`
- `ptx-inspection/v0.4.0/04-dkdv.ptx`
- `ptx-inspection/v0.4.0/04-dq.ptx`
