#!/usr/bin/env bash
# Request-only collection; no throughput/latency claim from this instrumented eager run.
#SBATCH --job-name=moeflow-full-routing
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=logs/full_routing_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
OUT="results/stage1/full_routing_$SLURM_JOB_ID"
mkdir -p "$OUT"
export MOEFLOW_ROUTING_TRACE_DIR="$PWD/$OUT/routing"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 840 scripts/e2e_bench.sh \
  "$MOEFLOW_MODEL_PATH" \
  4 "$OUT" 'native_agrs moeflow_fixed' '' --enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512
