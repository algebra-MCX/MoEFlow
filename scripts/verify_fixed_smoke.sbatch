#!/usr/bin/env bash
# Finish the bandwidth probe and verify fixed-mode integration on the trimmed checkpoint.
#SBATCH --job-name=moeflow-fixed-smoke
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/fixed_smoke_%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
OUT="results/stage1/fixed_smoke_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
date --iso-8601=seconds
git rev-parse HEAD
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv

timeout -s KILL 180 srun --overlap -n1 --gres=gpu:4 /usr/bin/env PYTHONPATH=/tmp/moeflow-bounds \
  "$TORCHRUN" --nproc-per-node=4 --master-port=$((29000 + SLURM_JOB_ID % 1000)) \
  /tmp/moeflow-bounds/scripts/bench_bounds.py --out "$PWD/$OUT/bounds_ep4.json" > "$OUT/bounds.log" 2>&1
bounds_status=$?
echo "$bounds_status" > "$OUT/bounds.exitcode"

export MOEFLOW_SMOKE_ARTIFACT_DIR="$PWD/$OUT/live"
export MOEFLOW_SMOKE_DP=4
timeout -s KILL 540 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  'tests/test_smoke_vllm.py::test_moeflow_runner_smoke[fixed]' -q \
  --junitxml="$OUT/fixed_smoke.xml" > "$OUT/fixed_smoke.log" 2>&1
smoke_status=$?
echo "$smoke_status" > "$OUT/fixed_smoke.exitcode"
date --iso-8601=seconds
if [ "$bounds_status" -ne 0 ] || [ "$smoke_status" -ne 0 ]; then exit 1; fi
