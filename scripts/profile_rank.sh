#!/usr/bin/env bash
# torchrun --no-python wrapper: collect one rank, while every rank runs the same layer calls.
# Traced timings are diagnostic and must not be mixed with uninstrumented baseline records.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/config.sh
if [ "${RANK:?}" = 0 ]; then
  exec nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
    --force-overwrite=true --output="${MOEFLOW_TRACE_OUT:?}" \
    "$PYTHON" scripts/bench_layer.py --profile "$@"
fi
exec "$PYTHON" scripts/bench_layer.py "$@"
