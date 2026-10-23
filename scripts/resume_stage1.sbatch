#!/usr/bin/env bash
# Serial continuation of the interrupted Claude session; submit from the repository root.
#SBATCH --job-name=moeflow-stage1-resume
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --output=logs/stage1_resume_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export MOEFLOW_JOBID="$SLURM_JOB_ID"
OUT="results/stage1/resume_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
date --iso-8601=seconds
git rev-parse HEAD
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-gpu=index,uuid,name,clocks.sm,clocks.mem,memory.used --format=csv
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
timeout -s KILL 360 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((24000 + SLURM_JOB_ID % 1000)) tests/multigpu/fixed_graph_check.py \
  --shape small --capacities 1,8,64 --replays 1000 --out "$OUT/fixed_graph_small_ep4.json" \
  > "$OUT/fixed_graph_small_ep4.log" 2>&1
timeout -s KILL 360 srun --overlap -n1 --gres=gpu:4 "$PYTHON" -m pytest \
  'tests/test_ep_nccl.py::test_fixed_path_cuda_graph_gate[4]' -q \
  --junitxml="$OUT/graph_regression.xml" > "$OUT/graph_regression.log" 2>&1
timeout -s KILL 480 srun --overlap -n1 --gres=gpu:4 "$TORCHRUN" --nproc-per-node=4 \
  --master-port=$((25000 + SLURM_JOB_ID % 1000)) tests/multigpu/fixed_graph_check.py \
  --shape full --capacities 1,8,64,128 --replays 1000 --out "$OUT/fixed_graph_full_ep4.json" \
  > "$OUT/fixed_graph_full_ep4.log" 2>&1
scripts/run_baselines.sh "$OUT/baseline" uniform 4
date --iso-8601=seconds
