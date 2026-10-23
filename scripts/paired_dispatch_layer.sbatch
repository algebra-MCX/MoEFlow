#!/usr/bin/env bash
# EP4 collective gate, dispatch-only full-layer fusion A/B, then native against the new path.
#SBATCH --job-name=moeflow-dispatch-layer
#SBATCH --gres=gpu:4
#SBATCH --time=00:06:00
#SBATCH --output=logs/dispatch_layer_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/dispatch_layer_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 120 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((41000 + SLURM_JOB_ID % 1000)) tests/multigpu/fixed_graph_check.py \
  --fused-dispatch --metadata-kernel --fused-reduce --fused-combine --capacities 1,8,64,128 \
  --replays 1000 --out "$OUT/collective_graph.json" > "$OUT/collective_graph.log" 2>&1
for arm in dispatch native; do
  EXTRA=()
  if [ "$arm" = dispatch ]; then EXTRA+=(--a-fixed-combine-mode bf16_ring_fused); fi
  timeout -s KILL 180 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((42000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py \
    --b-combine-mode bf16_ring_fused --b-fused-dispatch "${EXTRA[@]}" \
    --out "$OUT/${arm}_paired.json" > "$OUT/${arm}_paired.log" 2>&1
done
