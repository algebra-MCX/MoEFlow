#!/usr/bin/env bash
# Full-vocabulary raw logits for a bounded request prefix; correctness only, eager serving.
#SBATCH --job-name=moeflow-full-logits
#SBATCH --gres=gpu:4
#SBATCH --time=00:14:00
#SBATCH --output=logs/full_logits_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_LOGITS_VOCAB_SIZE=154880
export MOEFLOW_LOGITS_STEPS=8
export VLLM_SERVER_DEV_MODE=1
OUT="results/stage1/full_logits_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
MODEL="$MOEFLOW_MODEL_PATH"
for rank in 0 1; do
  export MOEFLOW_GREEDY_DP_RANK="$rank"
  timeout -s KILL 390 scripts/e2e_bench.sh "$MODEL" 4 "$OUT/rank$rank" \
    'native_agrs moeflow_fixed' '' --enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512 \
    --logprobs-mode raw_logits --max-logprobs -1 --host 127.0.0.1
done
