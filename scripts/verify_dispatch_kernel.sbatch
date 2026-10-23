#!/usr/bin/env bash
# Byte-level communication metadata and 1000 changing-input Graph replays; one GPU only.
#SBATCH --job-name=moeflow-dispatch-kernel
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --output=logs/dispatch_kernel_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/dispatch_kernel_$SLURM_JOB_ID"
GPUS=1
source scripts/lib/slurm_prologue.sh
timeout -s KILL 120 srun --overlap -n1 --gres=gpu:1 "$PYTHON" -m pytest tests/test_dispatch_pack.py -q \
  --junitxml="$OUT/kernel.xml" > "$OUT/kernel.log" 2>&1
