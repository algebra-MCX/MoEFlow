#!/usr/bin/env bash
# Measure kernel counts with the same experts; instrumented latency is not an A/B speed result.
#SBATCH --job-name=moeflow-fused-pipeline-trace
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=logs/fused_pipeline_trace_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="${MOEFLOW_RESULTS_ROOT:-$PWD/results/stage1}/fused_pipeline_trace_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
timeout -s KILL 90 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  tests/test_fixed_vllm_backend.py::test_layer_cli_records_fused_modes_and_reference_checks -q \
  --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
failed=0
for variant in native ring ring_dispatch; do
  backend=vllm:allgather_reducescatter
  extra=()
  if [ "$variant" != native ]; then
    backend=vllm:moeflow_fixed
    extra+=(--fixed-combine-mode bf16_ring_fused)
  fi
  if [ "$variant" = ring_dispatch ]; then extra+=(--fixed-fused-dispatch); fi
  export MOEFLOW_TRACE_OUT="$OUT/$variant"
  status=0
  timeout -s KILL 150 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((43000 + SLURM_JOB_ID % 1000)) --no-python scripts/profile_rank.sh \
    --backend "$backend" --tokens 1 --weights real:3 --reps 10 --inner 5 "${extra[@]}" \
    --out "$OUT/$variant.json" > "$OUT/$variant.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$OUT/$variant.exitcode"
  if [ "$status" -ne 0 ]; then failed=1; continue; fi
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    "$MOEFLOW_TRACE_OUT.nsys-rep" > "$OUT/$variant.stats.csv" 2> "$OUT/$variant.stats.log" || failed=1
done
if [ "$failed" = 0 ]; then
  "$PYTHON" scripts/summarize_replay_trace.py "$OUT" --variants native ring ring_dispatch \
    --out "$OUT/replay_report.md"
fi
exit "$failed"
