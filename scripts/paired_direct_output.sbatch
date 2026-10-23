#!/usr/bin/env bash
# Hold Ring/dispatch/preparation constant; only the return-buffer copy differs.
#SBATCH --job-name=moeflow-direct-output-ab
#SBATCH --gres=gpu:4
#SBATCH --time=00:03:00
#SBATCH --output=logs/direct_output_ab_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/direct_output_ab_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 160 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((47000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py \
  --a-fixed-combine-mode bf16_ring_fused --a-fused-dispatch --a-fused-expert-prep \
  --b-combine-mode bf16_ring_fused --b-fused-dispatch --b-fused-expert-prep --b-direct-expert-output \
  --check-graph-replays 1000 --out "$OUT/paired.json" > "$OUT/paired.log" 2>&1
"$PYTHON" scripts/summarize_backend_pair.py "$OUT/paired.json" --out "$OUT/report.md"
