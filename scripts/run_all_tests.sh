#!/usr/bin/env bash
# One command for the whole test suite. Inside a Slurm allocation with >= 4 GPUs it also runs the
# single-GPU, multi-GPU and (with MOEFLOW_JOBID set) the vLLM smoke tests; elsewhere they skip.
# Usage: MOEFLOW_JOBID=<alloc> scripts/run_all_tests.sh [extra pytest args]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/config.sh
mkdir -p results/pytest
STAMP=$(date +%Y%m%d-%H%M%S)
ARGS=(-q -p no:cacheprovider --junitxml="results/pytest/suite-$STAMP.xml" "$@")
if [ -n "${MOEFLOW_JOBID:-}" ]; then
  exec srun --jobid="$MOEFLOW_JOBID" --overlap -n1 --gres=gpu:4 --export=ALL "$PYTHON" -m pytest "${ARGS[@]}"
fi
exec "$PYTHON" -m pytest "${ARGS[@]}"
