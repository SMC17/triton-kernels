"""
Dump PTX for v0.4.1 fp16 backward kernels and v0.4.0 fp32 backward
kernels at a representative shape, then write the PTX text to
ptx-inspection/{v0.4.1,v0.4.0}/*.ptx so we can diff for tensor-core
instructions.

The strategy is to route triton through TRITON_CACHE_DIR (a fresh
tmp dir per kernel pass), launch the kernel at the target shape, then
walk the cache for the .ptx file Triton emitted.

Representative shape: B=1, H=8, M=N=512, D=64, causal=True. The
autotuner gets a single config forced via env so we know exactly which
(BLOCK_M, BLOCK_N, num_warps) the PTX corresponds to: BLOCK_M=BLOCK_N=32,
num_warps=4, num_stages=2 — matches the autotune winner on v0.4.1 at
M=128/256/512/1024 and at M=512 for both kernels.
"""
from __future__ import annotations
import importlib.util
import os
import shutil
import sys
import tempfile
import pathlib

import torch
import triton

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
KERNEL_DIR = REPO / "kernels"


def _load(path: pathlib.Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _collect_ptx(cache_dir: pathlib.Path) -> dict[str, str]:
    """Walk cache dir, return {kernel_basename: ptx_text}."""
    out: dict[str, str] = {}
    for ptx in cache_dir.rglob("*.ptx"):
        # Triton names the .ptx file after the kernel function.
        name = ptx.stem
        # Pick the first instance per kernel; if there are multiple
        # configs Triton drops one .ptx per config — append a counter.
        if name in out:
            counter = 1
            while f"{name}_alt{counter}" in out:
                counter += 1
            name = f"{name}_alt{counter}"
        out[name] = ptx.read_text()
    return out


def _restrict_to_single_config(mod, kernel_attr_names: list[str], BLOCK_M=32, BLOCK_N=32, num_warps=4, num_stages=2):
    """Replace each autotuned kernel's config list with a single config so
    only one PTX gets emitted per kernel. Mutates the module in place."""
    only = [triton.Config({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
                          num_warps=num_warps, num_stages=num_stages)]
    for attr in kernel_attr_names:
        kern = getattr(mod, attr)
        # `kern` is a triton.Autotuner instance. Replace its configs list.
        kern.configs = only
        # Drop any cached selection so the next launch recompiles fresh.
        kern.cache = {}


def dump_v041_fp16():
    cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="triton_ptx_v041_"))
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    # Force-reload Triton driver cache so the new env takes effect.
    # (Cleanest is to import the kernel module after env is set.)
    mod = _load(KERNEL_DIR / "04b_flash_attention_backward_fp16.py", "mod_v041")
    _restrict_to_single_config(mod, [
        "flash_attn_bwd_dkdv_kernel_fp16",
        "flash_attn_bwd_dq_kernel_fp16",
    ])
    torch.manual_seed(0)
    B, H, M, D = 1, 8, 512, 64
    q = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
    k = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
    v = (torch.randn(B, H, M, D, device="cuda", dtype=torch.float16) * 0.1)
    sm_scale = 1.0 / (D ** 0.5)
    o_t, lse_t = mod._fa_fwd_fp16.triton_flash_attn_fp16(
        q, k, v, sm_scale=sm_scale, is_causal=True, save_lse=True,
    )
    do = torch.ones_like(o_t)
    mod.triton_flash_attn_backward_fp16(
        q, k, v, o_t, do, lse_t, sm_scale=sm_scale, is_causal=True,
    )
    torch.cuda.synchronize()
    ptxs = _collect_ptx(cache_dir)
    return ptxs, cache_dir


def dump_v040_fp32():
    cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="triton_ptx_v040_"))
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    mod = _load(KERNEL_DIR / "04_flash_attention_backward.py", "mod_v040")
    _restrict_to_single_config(mod, [
        "flash_attn_bwd_dkdv_kernel",
        "flash_attn_bwd_dq_kernel",
    ])
    torch.manual_seed(0)
    B, H, M, D = 1, 8, 512, 64
    # fp32 inputs for v0.4.0.
    q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
    sm_scale = 1.0 / (D ** 0.5)
    # Use the forward from v0.4.0's sibling forward (03_flash_attention.py)
    # to produce o + lse.
    fwd_mod = _load(KERNEL_DIR / "03_flash_attention.py", "mod_v040_fwd")
    o_t, lse_t = fwd_mod.triton_flash_attn(
        q, k, v, sm_scale=sm_scale, is_causal=True, save_lse=True,
    )
    do = torch.ones_like(o_t)
    mod.triton_flash_attn_backward(
        q, k, v, o_t, do, lse_t, sm_scale=sm_scale, is_causal=True,
    )
    torch.cuda.synchronize()
    ptxs = _collect_ptx(cache_dir)
    return ptxs, cache_dir


def write_ptx(ptxs: dict[str, str], out_dir: pathlib.Path, label_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Map kernel-function-name -> friendly filename.
    # Triton sometimes prefixes/sufixes the .ptx stem with hashes; match
    # by substring of the kernel symbol.
    desired = {
        "preprocess": label_prefix + "-preprocess.ptx",
        "dkdv":       label_prefix + "-dkdv.ptx",
        "dq":         label_prefix + "-dq.ptx",
    }
    written = {}
    for name, text in ptxs.items():
        for needle, out_name in desired.items():
            if needle in name and out_name not in written:
                target = out_dir / out_name
                target.write_text(text)
                written[out_name] = name
                break
    print(f"=== {label_prefix} ===")
    for out_name, src_name in written.items():
        print(f"  {out_name}  <-  {src_name}")
    leftover = set(ptxs) - {src for src in written.values()}
    if leftover:
        print(f"  unmapped ptx kernels (also written verbatim):")
        for n in sorted(leftover):
            extra = out_dir / f"{label_prefix}-extra-{n}.ptx"
            extra.write_text(ptxs[n])
            print(f"    {extra.name}")
    return written


def main():
    print("Dumping v0.4.1 fp16 backward PTX ...")
    ptxs41, cd41 = dump_v041_fp16()
    print(f"  cache dir: {cd41}")
    print(f"  ptx files found: {sorted(ptxs41)}")
    write_ptx(ptxs41, HERE / "v0.4.1", "04b")

    print()
    print("Dumping v0.4.0 fp32 backward PTX ...")
    ptxs40, cd40 = dump_v040_fp32()
    print(f"  cache dir: {cd40}")
    print(f"  ptx files found: {sorted(ptxs40)}")
    write_ptx(ptxs40, HERE / "v0.4.0", "04")

    # Clean tmp caches.
    shutil.rmtree(cd41, ignore_errors=True)
    shutil.rmtree(cd40, ignore_errors=True)


if __name__ == "__main__":
    main()
