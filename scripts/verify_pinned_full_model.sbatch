#!/usr/bin/env bash
# Check native repeatability before comparing fixed, with every request pinned to DP rank 0.
#SBATCH --job-name=moeflow-pinned-full
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/pinned_full_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
unset MOEFLOW_REFERENCE_JSON MOEFLOW_ROUTING_TRACE_DIR MOEFLOW_VALIDATE_FIXED
OUT="results/stage1/pinned_full_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
MODEL="$MOEFLOW_MODEL_PATH"
SERVER_ARGS=(--enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512)
timeout -s KILL 180 scripts/e2e_bench.sh "$MODEL" 4 "$OUT/native" native_agrs '' "${SERVER_ARGS[@]}"
export MOEFLOW_REFERENCE_JSON="$PWD/$OUT/native/greedy_native_agrs.json"
timeout -s KILL 180 scripts/e2e_bench.sh "$MODEL" 4 "$OUT/native_repeat" native_agrs '' "${SERVER_ARGS[@]}"
export MOEFLOW_ROUTING_TRACE_DIR="$PWD/$OUT/fixed/routing"
export MOEFLOW_VALIDATE_FIXED=1
timeout -s KILL 180 scripts/e2e_bench.sh "$MODEL" 4 "$OUT/fixed" moeflow_fixed '' "${SERVER_ARGS[@]}"
