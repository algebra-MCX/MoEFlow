#!/usr/bin/env bash
# Full native-FP8 adapter: inspect every active layer sample, then require complete greedy output.
# Usage: sbatch scripts/verify_full_adapter.sbatch dispatch|expert_prep
#   dispatch:    fused dispatch packing only
#   expert_prep: fused dispatch packing plus fused expert input preparation
#SBATCH --job-name=moeflow-full-adapter
#SBATCH --gres=gpu:4
#SBATCH --time=00:06:00
#SBATCH --output=logs/full_adapter_%j.log
set -euo pipefail
STAGE="${1:?usage: verify_full_adapter.sbatch dispatch|expert_prep}"
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_FUSED_DISPATCH=1
case "$STAGE" in
  dispatch) ;;
  expert_prep) export MOEFLOW_FUSED_EXPERT_PREP=1 ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
export MOEFLOW_VALIDATE_FIXED=1
OUT="results/stage1/full_${STAGE}_$SLURM_JOB_ID"
export MOEFLOW_ROUTING_TRACE_DIR="$PWD/$OUT/routing"
GPUS=4
source scripts/lib/slurm_prologue.sh
MODEL="$MOEFLOW_MODEL_PATH"
timeout -s KILL 330 scripts/e2e_bench.sh "$MODEL" 4 "$OUT" 'native_agrs moeflow_fixed' '' \
  --enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512
