#!/usr/bin/env bash
# Layer baselines on one machine: every backend, same inputs, real layer-3 weights.
# Usage: MOEFLOW_JOBID=<alloc> scripts/run_baselines.sh <out_dir> [dist] [nproc]
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/config.sh
OUT=$1; DIST=${2:-uniform}; NPROC=${3:-4}
JOB=${MOEFLOW_JOBID:?}
mkdir -p "$OUT"
failed=0
for be in ${MOEFLOW_BACKENDS:-vllm:allgather_reducescatter vllm:deepep_high_throughput vllm:deepep_low_latency vllm:moeflow moeflow:triton_grouped moeflow:fixed}; do
  tag=$(echo "$be" | tr ':' '_')_${DIST}_ep${NPROC}
  EXTRA=()
  # DeepEP HT is not graph-capturable, and a failed capture left its buffer hung on the next
  # dispatch (2026-10-22, docs/limitations.md L11): time it eagerly; capture is probed separately.
  [ "$be" = vllm:deepep_high_throughput ] && EXTRA=(--no-graph)
  TOKENS=${MOEFLOW_TOKENS:-1,2,4,8,16,32,64,128,512,2048}
  # LL's dispatch buffer is sized from the largest point, so one oversized prefill point
  # prevents even T=1 from running. Keep its decode grid separate from the prefill grid.
  if [ "$be" = vllm:deepep_low_latency ]; then
    TOKENS=${MOEFLOW_LL_TOKENS:-1,2,4,8,16,32,64,128}
  fi
  srun --jobid="$JOB" --overlap -n1 --gres=gpu:"$NPROC" nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$OUT/$tag.preflight_procs.txt"
  timeout -s KILL 3600 srun --jobid="$JOB" --overlap -n1 --gres=gpu:"$NPROC" "$TORCHRUN" --nproc-per-node="$NPROC" \
    --master-port=$((22000 + RANDOM % 5000)) scripts/bench_layer.py --backend "$be" --weights real:3 --dist "$DIST" \
    --tokens "$TOKENS" --reps 10 --inner 20 "${EXTRA[@]}" \
    --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "$be exit=$rc"
  printf '%s\n' "$rc" > "$OUT/$tag.exitcode"
  [ "$rc" -ne 0 ] && failed=1
done
exit "$failed"
