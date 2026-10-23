#!/usr/bin/env bash
# Reference gate, full-layer Ring fusion A/B, and native vs the fused compatibility mode.
#SBATCH --job-name=moeflow-ring-layer
#SBATCH --gres=gpu:4
#SBATCH --time=00:07:00
#SBATCH --output=logs/ring_layer_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/ring_layer_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 120 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms -q \
  --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
for arm in ring native; do
  EXTRA=()
  if [ "$arm" = ring ]; then EXTRA+=(--a-fixed-ring); fi
  timeout -s KILL 180 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((39000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py \
    --b-combine-mode bf16_ring_fused "${EXTRA[@]}" --tokens 1,2,4,8,16,32,64,128 \
    --out "$OUT/${arm}_paired.json" > "$OUT/${arm}_paired.log" 2>&1
done
