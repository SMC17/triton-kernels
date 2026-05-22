#!/usr/bin/env bash
# run_correctness_sweep.sh — v0.4.3 no-regression sweep across all 8 kernels.
#
# Runs each kernel's `--correctness` mode sequentially, captures pass/fail,
# prints a summary table, and exits non-zero on any fail. The set was
# called "the six-kernel sweep" in v0.3.0; v0.4.0 + v0.4.1 added flash-
# attention backward (04, 04b) bringing the count to 8.
#
# THIS IS NOT A HOSTED CI PIPELINE. It's a local script the operator
# (or a pre-commit / pre-tag hook) runs by hand. No GitHub Actions yet;
# the substrate is sovereign-Forgejo-first.
#
# Usage:
#   scripts/run_correctness_sweep.sh
#
# Pre-tag hook (optional, NOT installed by this script — operator choice):
#   cat > .git/hooks/pre-tag <<'EOF'
#   #!/usr/bin/env bash
#   exec scripts/run_correctness_sweep.sh
#   EOF
#   chmod +x .git/hooks/pre-tag
# Note: git does not natively call a `pre-tag` hook; wire this into the
# operator's release script that invokes `git tag` if desired.
#
# Requires: triton-python wrapper on PATH (or set TRITON_PY env var to
# point at a python with triton + torch + a working CUDA runtime).

set -uo pipefail

TRITON_PY="${TRITON_PY:-triton-python}"
if ! command -v "$TRITON_PY" >/dev/null 2>&1; then
  echo "run_correctness_sweep: $TRITON_PY not on PATH" >&2
  echo "                       set TRITON_PY=/path/to/python with triton+torch" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

KERNELS=(
  "kernels/01_fused_softmax.py"
  "kernels/01b_fused_softmax_fp16.py"
  "kernels/02_fused_rmsnorm.py"
  "kernels/02b_fused_rmsnorm_fp16.py"
  "kernels/03_flash_attention.py"
  "kernels/03b_flash_attention_fp16.py"
  "kernels/04_flash_attention_backward.py"
  "kernels/04b_flash_attention_backward_fp16.py"
)

declare -a NAMES=()
declare -a STATUSES=()
declare -a DURATIONS_S=()

T_TOTAL_START="$(date +%s)"
ANY_FAIL=0

for k in "${KERNELS[@]}"; do
  name="$(basename "$k" .py)"
  printf '== %s ==\n' "$name"
  t0="$(date +%s)"
  if "$TRITON_PY" "$k" --correctness; then
    status="PASS"
  else
    status="FAIL"
    ANY_FAIL=1
  fi
  t1="$(date +%s)"
  dur=$((t1 - t0))
  NAMES+=("$name")
  STATUSES+=("$status")
  DURATIONS_S+=("$dur")
  printf '  -> %s (%ds)\n\n' "$status" "$dur"
done

T_TOTAL_END="$(date +%s)"
TOTAL=$((T_TOTAL_END - T_TOTAL_START))

echo "================================================================"
echo "  correctness sweep summary  (${#KERNELS[@]} kernels, total ${TOTAL}s)"
echo "================================================================"
printf '  %-40s  %-6s  %5s\n' "kernel" "status" "secs"
printf '  %-40s  %-6s  %5s\n' "------" "------" "----"
for i in "${!NAMES[@]}"; do
  printf '  %-40s  %-6s  %5s\n' "${NAMES[$i]}" "${STATUSES[$i]}" "${DURATIONS_S[$i]}"
done
echo

if [[ "$ANY_FAIL" -ne 0 ]]; then
  echo "FAIL — one or more kernels failed their correctness check."
  exit 1
fi
echo "PASS — all ${#KERNELS[@]} kernels green."
