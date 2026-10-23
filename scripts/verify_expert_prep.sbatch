#!/usr/bin/env bash
# Byte-exact expert inputs and changing-input Graph gate, then EP2 complete-layer reference/A-B.
#SBATCH --job-name=moeflow-expert-prep-gate
#SBATCH --gres=gpu:2
#SBATCH --time=00:03:00
#SBATCH --output=logs/expert_prep_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/expert_prep_$SLURM_JOB_ID"
GPUS=2
source scripts/lib/slurm_prologue.sh
timeout -s KILL 90 srun --overlap -n1 --gres=gpu:2 "$PYTHON" -m pytest tests/test_expert_prepare.py -q \
  --junitxml="$OUT/kernel.xml" > "$OUT/kernel.log" 2>&1
timeout -s KILL 75 srun --overlap -n1 --gres=gpu:2 "$PYTHON" -m pytest \
  'tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms[expert_prep]' -q \
  --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
