#!/usr/bin/env bash
# Keep dispatch/combination identical for the preparation A/B; native is a separate comparison.
#SBATCH --job-name=moeflow-expert-prep-layer
#SBATCH --gres=gpu:4
#SBATCH --time=00:07:00
#SBATCH --output=logs/expert_prep_layer_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/expert_prep_layer_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
for arm in preparation native; do
  EXTRA=()
  if [ "$arm" = preparation ]; then
    EXTRA+=(--a-fixed-combine-mode bf16_ring_fused --a-fused-dispatch --check-graph-replays 1000)
  fi
  timeout -s KILL 210 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((44000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py \
    --b-combine-mode bf16_ring_fused --b-fused-dispatch --b-fused-expert-prep "${EXTRA[@]}" \
    --out "$OUT/${arm}_paired.json" > "$OUT/${arm}_paired.log" 2>&1
done
