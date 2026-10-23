#!/usr/bin/env bash
# Serial measurements of the existing fixed path and baseline traces; no flags are enabled by default.
#SBATCH --job-name=moeflow-stage1-analysis
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=logs/stage1_analysis_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/analysis_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
date --iso-8601=seconds
git rev-parse HEAD
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.mem,memory.used --format=csv
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv

failed=0
port=$((27000 + SLURM_JOB_ID % 1000))
run_task() {
  local name=$1
  shift
  local status=0
  timeout -s KILL 600 srun --overlap -n1 --gres=gpu:4 "$@" > "$OUT/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$OUT/$name.exitcode"
  echo "$name exit=$status"
  if [ "$status" -ne 0 ]; then failed=1; fi
  port=$((port + 1))
}

# PR #1 is kept in its own checkout, so it can be measured without merging unapproved PRs.
# Prepare it with: git worktree add /tmp/moeflow-bounds feat/bench-bounds-tables
run_task bounds /usr/bin/env PYTHONPATH=/tmp/moeflow-bounds "$TORCHRUN" --nproc-per-node=4 \
  --master-port="$port" /tmp/moeflow-bounds/scripts/bench_bounds.py --out "$PWD/$OUT/bounds_ep4.json"

for backend in allgather_reducescatter deepep_high_throughput deepep_low_latency; do
  export MOEFLOW_TRACE_OUT="$PWD/$OUT/profile_$backend"
  extra=()
  if [ "$backend" = deepep_high_throughput ]; then extra=(--no-graph); fi
  run_task "profile_$backend" "$TORCHRUN" --nproc-per-node=4 --master-port="$port" \
    --no-python scripts/profile_rank.sh --backend "vllm:$backend" --tokens 1,128 \
    --weights real:3 --reps 10 --inner 5 "${extra[@]}" --out "$OUT/profile_$backend.json"
  if [ -f "$MOEFLOW_TRACE_OUT.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
      "$MOEFLOW_TRACE_OUT.nsys-rep" > "$OUT/profile_$backend.stats.csv" 2> "$OUT/profile_$backend.stats.log" || failed=1
  fi
done

run_task stages "$TORCHRUN" --nproc-per-node=4 --master-port="$port" \
  scripts/bench_stages.py --tokens 1,8,64,128 --weights real:3 --out "$OUT/stages.json"
run_task ab_metadata "$TORCHRUN" --nproc-per-node=4 --master-port="$port" \
  scripts/ab_fixed.py --a '' --b metadata_kernel --tokens 1,8,64,128 --weights real:3 --out "$OUT/ab_metadata.json"
run_task ab_reduce "$TORCHRUN" --nproc-per-node=4 --master-port="$port" \
  scripts/ab_fixed.py --a metadata_kernel --b metadata_kernel,fused_reduce --tokens 1,8,64,128 \
  --weights real:3 --out "$OUT/ab_reduce.json"
run_task ab_combine "$TORCHRUN" --nproc-per-node=4 --master-port="$port" \
  scripts/ab_fixed.py --a '' --b fused_combine --tokens 1,8,64,128 --weights real:3 --out "$OUT/ab_combine.json"
date --iso-8601=seconds
exit "$failed"
