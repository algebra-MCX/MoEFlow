#!/usr/bin/env bash
# Direct combined A/B; do not sum percentages from the individual ablations.
#SBATCH --job-name=moeflow-total-fusion
#SBATCH --gres=gpu:4
#SBATCH --time=00:04:00
#SBATCH --output=logs/total_fusion_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/total_fusion_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 210 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((45000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py \
  --a-fixed-ring --b-combine-mode bf16_ring_fused --b-fused-dispatch --b-fused-expert-prep \
  --check-graph-replays 1000 --out "$OUT/paired.json" > "$OUT/paired.log" 2>&1
