#!/usr/bin/env bash
# Re-run complete-path changing-input Graph checks with the bytewise comparator.
#SBATCH --job-name=moeflow-bytewise-graph
#SBATCH --gres=gpu:2
#SBATCH --time=00:02:00
#SBATCH --output=logs/bytewise_graph_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/bytewise_graph_$SLURM_JOB_ID"
GPUS=2
source scripts/lib/slurm_prologue.sh
timeout -s KILL 100 srun --overlap -n1 --gres=gpu:2 "$PYTHON" -m pytest \
  'tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms[direct_output]' -q \
  --basetemp="$OUT/pytest" --junitxml="$OUT/graph.xml" > "$OUT/graph.log" 2>&1
"$PYTHON" - "$OUT" <<'PY'
import shutil
import sys
from pathlib import Path

out = Path(sys.argv[1])
records = list((out / "pytest").rglob("paired.json"))
assert len(records) == 1, records
shutil.copy2(records[0], out / "paired.json")
PY
