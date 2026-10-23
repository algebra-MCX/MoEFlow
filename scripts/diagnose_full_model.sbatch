#!/usr/bin/env bash
# Identical-input layer comparisons; this synchronous eager run measures no performance.
# Usage: sbatch scripts/diagnose_full_model.sbatch <native_greedy.json> [moeflow_fixed|native_agrs]
#SBATCH --job-name=moeflow-full-diagnostic
#SBATCH --gres=gpu:4
#SBATCH --time=00:06:00
#SBATCH --output=logs/full_diagnostic_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_REFERENCE_JSON
MOEFLOW_REFERENCE_JSON=$(realpath "${1:?Native greedy reference is required}")
test -f "$MOEFLOW_REFERENCE_JSON"
MODE=${2:-moeflow_fixed}
case "$MODE" in moeflow_fixed|native_agrs) ;; *) echo "Unsupported diagnostic mode: $MODE"; exit 2 ;; esac
OUT="results/stage1/full_diagnostic_$SLURM_JOB_ID"
mkdir -p "$OUT"
export MOEFLOW_ROUTING_TRACE_DIR="$PWD/$OUT/routing"
export MOEFLOW_VALIDATE_FIXED=1
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 330 scripts/e2e_bench.sh \
  "$MOEFLOW_MODEL_PATH" \
  4 "$OUT" "$MODE" '' --enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512
