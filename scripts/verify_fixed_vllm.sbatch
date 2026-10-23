#!/usr/bin/env bash
# Same router and expert kernels; measure transport and combine separately from custom GEMM.
#SBATCH --job-name=moeflow-fixed-vllm
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=logs/fixed_vllm_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/fixed_vllm_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 180 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  tests/test_fixed_vllm_backend.py -q --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
for variant in native fixed fixed_combine; do
  backend=vllm:allgather_reducescatter
  extra=()
  if [ "$variant" != native ]; then backend=vllm:moeflow_fixed; fi
  if [ "$variant" = fixed_combine ]; then extra=(--fixed-fused-combine); fi
  timeout -s KILL 240 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((33000 + SLURM_JOB_ID % 1000)) scripts/bench_layer.py --backend "$backend" \
    --tokens 1,2,4,8,16,32,64,128 --weights real:3 "${extra[@]}" --out "$OUT/$variant.json" \
    > "$OUT/$variant.log" 2>&1
done
