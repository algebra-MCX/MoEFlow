#!/usr/bin/env bash
# Alternating same-GEMM comparisons; validate every token on every rank before timing.
#SBATCH --job-name=moeflow-skewed-routes
#SBATCH --gres=gpu:4
#SBATCH --time=00:06:00
#SBATCH --output=logs/skewed_routes_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/skewed_routes_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
failed=0
for distribution in zipf hotspot; do
  status=0
  timeout -s KILL 170 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((37000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py \
    --dist "$distribution" --b-fused-combine --out "$OUT/$distribution.json" \
    > "$OUT/$distribution.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$OUT/$distribution.exitcode"
  if [ "$status" -ne 0 ]; then failed=1; fi
done
exit "$failed"
