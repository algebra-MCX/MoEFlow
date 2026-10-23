#!/usr/bin/env bash
# Usage: scripts/run_multigpu_check.sh <nproc> [ep_nccl_check.py args...]
# Runs inside an existing Slurm allocation (SLURM_JOB_ID or MOEFLOW_JOBID) with an outer watchdog.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/config.sh
NPROC=$1; shift
JOB=${MOEFLOW_JOBID:-${SLURM_JOB_ID:?set MOEFLOW_JOBID or run inside an allocation}}
PORT=$((20000 + (JOB * 7 + NPROC) % 10000))
exec timeout -s KILL "${MOEFLOW_OUTER_TIMEOUT:-600}" srun --jobid="$JOB" --overlap -n1 --gres=gpu:"$NPROC" \
  "$TORCHRUN" --nproc-per-node="$NPROC" --master-port="$PORT" tests/multigpu/ep_nccl_check.py "$@"
