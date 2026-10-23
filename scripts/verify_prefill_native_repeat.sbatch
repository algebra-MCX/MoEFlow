#!/usr/bin/env bash
# Reuse the exact prompts/config from 9270 to separate native serving variation from fixed transport.
#SBATCH --job-name=moeflow-prefill-control
#SBATCH --gres=gpu:4
#SBATCH --time=00:04:00
#SBATCH --output=logs/prefill_control_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
SOURCE="$PWD/results/stage1/long_prefill_9270"
export MOEFLOW_GREEDY_PROMPTS="$SOURCE/prompts.json"
export MOEFLOW_REFERENCE_JSON="$SOURCE/greedy_native_agrs.json"
OUT="results/stage1/prefill_control_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
MODEL="$MOEFLOW_MODEL_PATH"
timeout -s KILL 210 scripts/e2e_bench.sh "$MODEL" 4 "$OUT" native_agrs '' \
  --max-num-seqs 4 --max-num-batched-tokens 512 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = json.loads((out / "compare_native_agrs.json").read_text())
summary = [{k: v for k, v in row.items() if k != "prompt"} for row in rows]
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
assert all(row["passed"] and row["max_abs_logprob_diff_on_prefix"] == 0 for row in rows), summary
PY
