#!/usr/bin/env bash
# First bounded end-to-end point; six workloads and repeat statistics remain separate.
#SBATCH --job-name=moeflow-e2e-w1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/e2e_w1_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
MODES=${1:-"native_agrs moeflow_fixed"}
case "$MODES" in
  "native_agrs moeflow_fixed"|"moeflow_fixed native_agrs") ;;
  *) echo "Expected a native/fixed pair in either order"; exit 2 ;;
esac
if [ "$MODES" = "moeflow_fixed native_agrs" ] && [ -z "${MOEFLOW_REFERENCE_JSON:-}" ]; then
  echo "Reverse order requires an existing four-rank native reference"; exit 2
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
unset MOEFLOW_GREEDY_DP_RANK
export MOEFLOW_GREEDY_DP_RANKS="0 1 2 3"
export MOEFLOW_GREEDY_REQUESTS=4
export MOEFLOW_EXACT_GREEDY=1
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_FUSED_DISPATCH=1
export MOEFLOW_FUSED_EXPERT_PREP=1
OUT="results/stage1/e2e_w1_$SLURM_JOB_ID"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
git rev-parse HEAD
git status --porcelain --untracked-files=no
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
export MOEFLOW_GREEDY_PROMPTS="$PWD/$OUT/prompts.json"
"$PYTHON" - "$MOEFLOW_GREEDY_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path

prompt = json.loads(Path("results/stage1/long_prefill_9270/prompts.json").read_text())[1]
assert len(prompt) == 1024
Path(sys.argv[1]).write_text(json.dumps([prompt]))
PY
MODEL="$MOEFLOW_MODEL_PATH"
timeout -s KILL "${MOEFLOW_SERVE_TIMEOUT:-570}" scripts/e2e_bench.sh "$MODEL" 4 "$OUT" "$MODES" W1 \
  --max-num-seqs 4 --max-num-batched-tokens 512 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
