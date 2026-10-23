#!/usr/bin/env bash
# Exact rounding and changing-input Graph gate, then four-rank combine-only A/B.
#SBATCH --job-name=moeflow-fused-ring
#SBATCH --gres=gpu:4
#SBATCH --time=00:03:00
#SBATCH --output=logs/fused_ring_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/fused_ring_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 90 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest tests/test_ring_kernel.py -q \
  --junitxml="$OUT/kernel.xml" > "$OUT/kernel.log" 2>&1
timeout -s KILL 75 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((38000 + SLURM_JOB_ID % 1000)) scripts/bench_ring_combine.py \
  --out "$OUT/combine_ab.json" > "$OUT/combine_ab.log" 2>&1
