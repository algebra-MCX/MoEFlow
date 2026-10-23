#!/usr/bin/env bash
#SBATCH --job-name=moeflow-bounds
#SBATCH --gres=gpu:4
#SBATCH --time=00:03:00
#SBATCH --output=logs/bounds_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/bounds_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
date --iso-8601=seconds
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 60 srun --overlap -n1 --gres=gpu:1 /usr/bin/env PYTHONPATH=/tmp/moeflow-bounds \
  "$PYTHON" -m pytest /tmp/moeflow-bounds/tests/test_bench_bounds.py::test_bandwidth_probe_small_gpu_buffer \
  -q --junitxml="$PWD/$OUT/small.xml" > "$OUT/small.log" 2>&1
timeout -s KILL 120 srun --overlap -n1 --gres=gpu:4 /usr/bin/env PYTHONPATH=/tmp/moeflow-bounds \
  "$TORCHRUN" --nproc-per-node=4 --master-port=$((32000 + SLURM_JOB_ID % 1000)) \
  /tmp/moeflow-bounds/scripts/bench_bounds.py --out "$PWD/$OUT/bounds_ep4.json" > "$OUT/bounds.log" 2>&1
date --iso-8601=seconds
