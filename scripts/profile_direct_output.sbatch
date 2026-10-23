#!/usr/bin/env bash
# One validated candidate trace; compare launch counts with the retained native/Ring/pack traces.
#SBATCH --job-name=moeflow-expert-prep-trace
#SBATCH --gres=gpu:4
#SBATCH --time=00:03:00
#SBATCH --output=logs/direct_output_trace_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="${MOEFLOW_RESULTS_ROOT:-$PWD/results/stage1}/direct_output_trace_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
export MOEFLOW_TRACE_OUT="$OUT/direct_output"
status=0
timeout -s KILL 150 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((46000 + SLURM_JOB_ID % 1000)) --no-python scripts/profile_rank.sh \
  --backend vllm:moeflow_fixed --tokens 1 --weights real:3 --reps 10 --inner 5 \
  --fixed-combine-mode bf16_ring_fused --fixed-fused-dispatch --fixed-fused-expert-prep --fixed-direct-expert-output \
  --out "$OUT/direct_output.json" > "$OUT/direct_output.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$OUT/direct_output.exitcode"
if [ "$status" -ne 0 ]; then exit "$status"; fi
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
  "$MOEFLOW_TRACE_OUT.nsys-rep" > "$OUT/direct_output.stats.csv" 2> "$OUT/direct_output.stats.log"
"$PYTHON" scripts/summarize_replay_trace.py "$OUT" --variants direct_output --out "$OUT/replay_report.md"
