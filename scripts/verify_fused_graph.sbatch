#!/usr/bin/env bash
# Validate all existing fusion switches under changing inputs before drawing optimisation conclusions.
#SBATCH --job-name=moeflow-fused-graph
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=logs/fused_graph_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
OUT="results/stage1/fused_graph_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
date --iso-8601=seconds
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 120 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  tests/test_ep_nccl.py::test_fixed_fused_path_cuda_graph_gate -q \
  --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
for variant in baseline fused; do
  flags=()
  if [ "$variant" = fused ]; then flags=(--metadata-kernel --fused-reduce --fused-combine); fi
  timeout -s KILL 180 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
    --master-port=$((30000 + SLURM_JOB_ID % 1000)) tests/multigpu/fixed_graph_check.py \
    --shape full --capacities 1,8,64,128 --replays 1000 "${flags[@]}" \
    --out "$OUT/full_$variant.json" > "$OUT/full_$variant.log" 2>&1
done
timeout -s KILL 300 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((31000 + SLURM_JOB_ID % 1000)) scripts/ab_fixed.py --a '' \
  --b metadata_kernel,fused_reduce,fused_combine --tokens 1,8,64,128 --weights real:3 \
  --out "$OUT/ab_all.json" > "$OUT/ab_all.log" 2>&1
date --iso-8601=seconds
