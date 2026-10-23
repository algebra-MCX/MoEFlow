#!/usr/bin/env bash
# Control scheduler/cache effects: require native reproducibility before testing fixed.
#SBATCH --job-name=moeflow-prefill-sync
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=logs/prefill_sync_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
export MOEFLOW_EXACT_GREEDY=1
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_FUSED_DISPATCH=1
export MOEFLOW_FUSED_EXPERT_PREP=1
export MOEFLOW_GREEDY_PROMPTS="$PWD/results/stage1/long_prefill_9270/prompts.json"
OUT="results/stage1/prefill_sync_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
MODEL="$MOEFLOW_MODEL_PATH"
unset MOEFLOW_REFERENCE_JSON
for arm in native_first native_repeat moeflow_fixed; do
  mode=native_agrs
  if [ "$arm" = moeflow_fixed ]; then mode=moeflow_fixed; fi
  if [ "$arm" != native_first ]; then
    export MOEFLOW_REFERENCE_JSON="$PWD/$OUT/native_first/greedy_native_agrs.json"
  fi
  timeout -s KILL 150 scripts/e2e_bench.sh "$MODEL" 4 "$OUT/$arm" "$mode" '' \
    --no-async-scheduling --no-enable-prefix-caching \
    --max-num-seqs 4 --max-num-batched-tokens 512 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
done
