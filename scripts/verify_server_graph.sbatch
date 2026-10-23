#!/usr/bin/env bash
# Serving Graph gate: 50 serial native/fixed requests per arm, pinned DP 0, strict complete outputs.
# No performance claim. Usage: sbatch scripts/verify_server_graph.sbatch full|trimmed
#   full:    complete GLM-5.3-Flash (MOEFLOW_SERVE_TIMEOUT overrides the 660 s watchdog)
#   trimmed: GLM-5.3-Flash-4L
#SBATCH --job-name=moeflow-server-graph
#SBATCH --gres=gpu:4
#SBATCH --time=00:12:00
#SBATCH --output=logs/server_graph_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
MODEL_KIND="${1:?usage: verify_server_graph.sbatch full|trimmed}"
case "$MODEL_KIND" in
  full)
    MODEL="$MOEFLOW_MODEL_PATH"
    WATCHDOG="${MOEFLOW_SERVE_TIMEOUT:-660}" ;;
  trimmed)
    MODEL="$MOEFLOW_TRIMMED_MODEL_PATH"
    WATCHDOG=420 ;;
  *) echo "unknown model kind: $MODEL_KIND" >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_GREEDY_REQUESTS=50
OUT="results/stage1/${MODEL_KIND}_server_graph_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL "$WATCHDOG" scripts/e2e_bench.sh "$MODEL" 4 "$OUT" \
  'native_agrs moeflow_fixed' '' --max-num-seqs 4 --max-num-batched-tokens 512 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
