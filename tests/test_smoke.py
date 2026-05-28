"""
Smoke tests for triton-kernels.

These tests run on a stock Python 3.10+ install with NO CUDA, NO Triton,
NO PyTorch, and NO GPU. They cannot exercise kernel correctness or
performance — that requires the bench-host venv per `README.md` (Build /
run) and a CUDA-capable Turing+ GPU.

What these tests DO verify:
- Every kernel file parses as valid Python (ast.parse).
- Every kernel file declares the public entry-point function this repo
  has documented as its API surface (e.g. `triton_softmax` in
  `01_fused_softmax.py`).
- The set of kernel files matches the documented inventory (8 kernels:
  01 / 01b / 02 / 02b / 03 / 03b / 04 / 04b).
- The autotune-config helper exists in each kernel that v0.3.0 declared
  autotuned.

What these tests CANNOT verify (substrate-honest non-claim):
- That the kernels actually compile under Triton.
- That CUDA initialises in the bench environment.
- That the correctness sweeps in `scripts/run_correctness_sweep.sh`
  pass on this hardware. STATUS.md tracks the current bench-host
  CUDA path status as the source of truth for that.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNELS_DIR = REPO_ROOT / "kernels"


# Public-API contract: filename -> (public-entry-point fn name, autotune-grid fn name or None)
EXPECTED_KERNELS: dict[str, tuple[str, str | None]] = {
    "01_fused_softmax.py":              ("triton_softmax",                "_softmax_autotune_configs"),
    "01b_fused_softmax_fp16.py":        ("triton_softmax_fp16",           "_softmax_fp16_autotune_configs"),
    "02_fused_rmsnorm.py":              ("triton_rmsnorm",                "_rmsnorm_autotune_configs"),
    "02b_fused_rmsnorm_fp16.py":        ("triton_rmsnorm_fp16",           "_rmsnorm_fp16_autotune_configs"),
    "03_flash_attention.py":            ("triton_flash_attn",             "_flash_attn_autotune_configs"),
    "03b_flash_attention_fp16.py":      ("triton_flash_attn_fp16",        "_flash_attn_fp16_autotune_configs"),
    "04_flash_attention_backward.py":   ("triton_flash_attn_backward",    None),
    "04b_flash_attention_backward_fp16.py": ("triton_flash_attn_backward_fp16", None),
}


def _top_level_function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _parse(path: pathlib.Path) -> ast.Module:
    src = path.read_text(encoding="utf-8")
    return ast.parse(src, filename=str(path))


def test_kernel_inventory_matches() -> None:
    """The set of *.py files in kernels/ must match the declared 8-kernel inventory."""
    actual = {p.name for p in KERNELS_DIR.glob("*.py")}
    expected = set(EXPECTED_KERNELS.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"missing kernel files: {sorted(missing)}"
    assert not extra, f"unexpected kernel files: {sorted(extra)}"


def test_each_kernel_parses() -> None:
    """ast.parse must succeed on every kernel file."""
    for name in EXPECTED_KERNELS:
        path = KERNELS_DIR / name
        # Will raise SyntaxError on failure.
        _parse(path)


def test_each_kernel_declares_public_entrypoint() -> None:
    """Every kernel file must declare its documented public entry-point function."""
    failures: list[str] = []
    for name, (entrypoint, _) in EXPECTED_KERNELS.items():
        path = KERNELS_DIR / name
        names = _top_level_function_names(_parse(path))
        if entrypoint not in names:
            failures.append(f"{name}: missing public entry-point `{entrypoint}` (found: {sorted(names)})")
    assert not failures, "\n".join(failures)


def test_autotuned_kernels_declare_config_grid() -> None:
    """Per v0.3.0, kernels 01/01b/02/02b/03/03b ship an autotune-grid helper."""
    failures: list[str] = []
    for name, (_, grid_fn) in EXPECTED_KERNELS.items():
        if grid_fn is None:
            continue
        path = KERNELS_DIR / name
        names = _top_level_function_names(_parse(path))
        if grid_fn not in names:
            failures.append(f"{name}: missing autotune-grid helper `{grid_fn}` (found: {sorted(names)})")
    assert not failures, "\n".join(failures)


def test_kernels_import_triton() -> None:
    """Every kernel file must declare `import triton` (sanity: nobody silently fell back to a non-Triton implementation)."""
    failures: list[str] = []
    for name in EXPECTED_KERNELS:
        path = KERNELS_DIR / name
        src = path.read_text(encoding="utf-8")
        if "import triton" not in src:
            failures.append(f"{name}: no `import triton` statement found")
    assert not failures, "\n".join(failures)


def test_ptx_inspection_script_parses() -> None:
    """The PTX dump helper used in v0.4.2 diagnosis must remain parseable."""
    path = REPO_ROOT / "ptx-inspection" / "dump_ptx.py"
    assert path.exists(), "ptx-inspection/dump_ptx.py missing"
    _parse(path)


def test_correctness_sweep_script_present() -> None:
    """v0.4.3 ships a local 8-kernel correctness sweep. Must exist + be executable."""
    path = REPO_ROOT / "scripts" / "run_correctness_sweep.sh"
    assert path.exists(), "scripts/run_correctness_sweep.sh missing"
    # We do NOT execute it in CI — it requires triton-python on PATH + CUDA.
    src = path.read_text(encoding="utf-8")
    # Verify the script enumerates all 8 kernels by their filename stem.
    expected_refs: Iterable[str] = ("01_fused_softmax", "01b_fused_softmax_fp16", "02_fused_rmsnorm",
                                    "02b_fused_rmsnorm_fp16", "03_flash_attention", "03b_flash_attention_fp16",
                                    "04_flash_attention_backward", "04b_flash_attention_backward_fp16")
    missing = [ref for ref in expected_refs if ref not in src]
    assert not missing, f"correctness sweep script missing references to: {missing}"


if __name__ == "__main__":
    # Allow `python tests/test_smoke.py` without pytest installed.
    # Each function is named `test_*`; run them sequentially, print result, exit non-zero on any fail.
    import sys
    import traceback
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
