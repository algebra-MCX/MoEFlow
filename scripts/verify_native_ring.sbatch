#!/usr/bin/env bash
# Check observed Ring rounding, then compare fresh native/fixed full-model outputs on DP 0/1.
#SBATCH --job-name=moeflow-native-ring
#SBATCH --gres=gpu:4
#SBATCH --time=00:12:00
#SBATCH --output=logs/native_ring_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_COMBINE_MODE=${MOEFLOW_COMBINE_MODE:-bf16_ring}
OUT="results/stage1/native_ring_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 90 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest tests/test_runner_ops.py -q \
  --junitxml="$OUT/runner_ops.xml" > "$OUT/runner_ops.log" 2>&1
MODEL="$MOEFLOW_MODEL_PATH"
for rank in 0 1; do
  export MOEFLOW_GREEDY_DP_RANK="$rank"
  export MOEFLOW_ROUTING_TRACE_DIR="$PWD/$OUT/rank$rank/routing"
  export MOEFLOW_VALIDATE_FIXED=1
  timeout -s KILL 330 scripts/e2e_bench.sh "$MODEL" 4 "$OUT/rank$rank" \
    'native_agrs moeflow_fixed' '' --enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512
done
