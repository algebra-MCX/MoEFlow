#!/usr/bin/env bash
# Diagnose long-context MoE on identical hidden inputs; do not substitute this for an end-to-end score gate.
#SBATCH --job-name=moeflow-long-layer-probe
#SBATCH --gres=gpu:4
#SBATCH --time=00:06:00
#SBATCH --output=logs/long_layer_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
export MOEFLOW_EXACT_GREEDY=1
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_FUSED_DISPATCH=1
export MOEFLOW_FUSED_EXPERT_PREP=1
export MOEFLOW_DIRECT_EXPERT_OUTPUT=0
export MOEFLOW_VALIDATE_FIXED=1
OUT="results/stage1/long_layer_$SLURM_JOB_ID"
export MOEFLOW_ROUTING_TRACE_DIR="$PWD/$OUT/routing"
export MOEFLOW_GREEDY_PROMPTS="$PWD/$OUT/prompts.json"
GPUS=4
source scripts/lib/slurm_prologue.sh
"$PYTHON" - "$MOEFLOW_GREEDY_PROMPTS" <<'PY'
import json
import sys
from pathlib import Path

prompt = json.loads(Path("results/stage1/long_prefill_9270/prompts.json").read_text())[3]
assert len(prompt) == 8192
Path(sys.argv[1]).write_text(json.dumps([prompt]))
PY
MODEL="$MOEFLOW_MODEL_PATH"
status=0
timeout -s KILL "${MOEFLOW_LONG_LAYER_TIMEOUT:-330}" scripts/e2e_bench.sh "$MODEL" 4 "$OUT" 'native_agrs moeflow_fixed' '' \
  --enforce-eager --max-num-seqs 4 --max-num-batched-tokens 512 || status=$?
printf '%s\n' "$status" > "$OUT/serving.exitcode"
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
samples = [json.loads(line) for path in (out / "routing").glob("layer_comparison_*.jsonl")
           for line in path.read_text().splitlines()]
active = [row for row in samples if row["rows_compared"]]
comparison = json.loads((out / "compare_moeflow_fixed.json").read_text())
assert len(comparison) == 1
native = json.loads((out / "greedy_native_agrs.json").read_text())[0]
fixed = json.loads((out / "greedy_moeflow_fixed.json").read_text())[0]
summary = {"active_samples": len(active), "layers": len({row["layer"] for row in active}),
           "byte_equal_samples": sum(row["bitwise_equal"] for row in active),
           "equality_check": "bytes" if active and all(row.get("equality_check") == "bytes" for row in active)
                             else "legacy_or_missing",
           "all_finite": bool(active) and all(row["finite"] for row in active),
           "complete_outputs_match": native["tokens"] == fixed["tokens"] and native["text"] == fixed["text"],
           "first_selected_logprob_diff": abs(native["logprobs"][0] - fixed["logprobs"][0]),
           "exact_selected_logprob_gate": all(row["passed"] for row in comparison),
           "max_selected_logprob_diff": max(row["max_abs_logprob_diff_on_prefix"] for row in comparison)}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
assert summary["active_samples"] == 84 and summary["layers"] == 42
assert summary["all_finite"] and summary["byte_equal_samples"] == 84 and summary["equality_check"] == "bytes"
PY
exit "$status"
