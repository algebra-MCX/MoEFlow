#!/usr/bin/env bash
# EP2 end-to-end collective gate, then a small complete-layer pack A/B reference regression.
#SBATCH --job-name=moeflow-dispatch-graph
#SBATCH --gres=gpu:2
#SBATCH --time=00:04:00
#SBATCH --output=logs/dispatch_graph_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/dispatch_graph_$SLURM_JOB_ID"
GPUS=2
source scripts/lib/slurm_prologue.sh
timeout -s KILL 150 srun --overlap -n1 --gres=gpu:2 "$TORCHRUN" --nproc-per-node=2 \
  --master-port=$((40000 + SLURM_JOB_ID % 1000)) tests/multigpu/fixed_graph_check.py \
  --fused-dispatch --metadata-kernel --fused-reduce --fused-combine --capacities 1,8,64,128 \
  --replays 1000 --out "$OUT/collective_graph.json" > "$OUT/collective_graph.log" 2>&1
timeout -s KILL 75 srun --overlap -n1 --gres=gpu:2 "$PYTHON" -m pytest \
  'tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms[dispatch]' -q \
  --junitxml="$OUT/small.xml" > "$OUT/small.log" 2>&1
