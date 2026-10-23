#!/usr/bin/env bash
# Check the external expert-output allocation survives changing-input collective Graph replays.
#SBATCH --job-name=moeflow-direct-output
#SBATCH --gres=gpu:2
#SBATCH --time=00:02:00
#SBATCH --output=logs/direct_output_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/direct_output_$SLURM_JOB_ID"
GPUS=2
source scripts/lib/slurm_prologue.sh
timeout -s KILL 100 srun --overlap -n1 --gres=gpu:2 "$PYTHON" -m pytest \
  'tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms[direct_output]' -q \
  --junitxml="$OUT/graph.xml" > "$OUT/graph.log" 2>&1
