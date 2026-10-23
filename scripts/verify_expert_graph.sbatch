#!/usr/bin/env bash
# Full external-expert path: changing inputs/routes, 1000 Graph checks at each of two capacities.
#SBATCH --job-name=moeflow-expert-graph-gate
#SBATCH --gres=gpu:2
#SBATCH --time=00:03:00
#SBATCH --output=logs/expert_graph_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT="results/stage1/expert_graph_$SLURM_JOB_ID"
GPUS=2
source scripts/lib/slurm_prologue.sh
timeout -s KILL 150 srun --overlap -n1 --gres=gpu:2 "$PYTHON" -m pytest \
  'tests/test_fixed_vllm_backend.py::test_native_fixed_paired_comparison_checks_both_arms[expert_prep]' -q \
  --junitxml="$OUT/graph.xml" > "$OUT/graph.log" 2>&1
