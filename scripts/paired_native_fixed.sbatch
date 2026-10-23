#!/usr/bin/env bash
#SBATCH --job-name=moeflow-paired-backends
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=logs/paired_backends_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/paired_backends_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 120 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms -q \
  --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
timeout -s KILL 240 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((34000 + SLURM_JOB_ID % 1000)) scripts/ab_native_fixed.py --b-fused-combine \
  --out "$OUT/paired.json" > "$OUT/paired.log" 2>&1
