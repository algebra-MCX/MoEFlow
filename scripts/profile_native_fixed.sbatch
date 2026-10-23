#!/usr/bin/env bash
# Diagnose same-GEMM paths; instrumented timings are separate from paired baseline data.
#SBATCH --job-name=moeflow-native-fixed-trace
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=logs/native_fixed_trace_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/native_fixed_trace_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
failed=0
for variant in native fixed fixed_combine; do
  backend=vllm:allgather_reducescatter
  extra=()
  if [ "$variant" != native ]; then backend=vllm:moeflow_fixed; fi
  if [ "$variant" = fixed_combine ]; then extra=(--fixed-fused-combine); fi
  export MOEFLOW_TRACE_OUT="$PWD/$OUT/$variant"
  status=0
  timeout -s KILL 150 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((36000 + SLURM_JOB_ID % 1000)) --no-python scripts/profile_rank.sh \
    --backend "$backend" --tokens 1 --weights real:3 --reps 10 --inner 5 "${extra[@]}" \
    --out "$OUT/$variant.json" > "$OUT/$variant.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$OUT/$variant.exitcode"
  if [ "$status" -ne 0 ]; then failed=1; continue; fi
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    "$MOEFLOW_TRACE_OUT.nsys-rep" > "$OUT/$variant.stats.csv" 2> "$OUT/$variant.stats.log" || failed=1
done
exit "$failed"
