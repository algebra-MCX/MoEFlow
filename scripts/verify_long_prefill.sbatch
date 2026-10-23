#!/usr/bin/env bash
# Fixed decode -> native chunked prefill -> fixed decode, with full-model output comparison.
#SBATCH --job-name=moeflow-long-prefill
#SBATCH --gres=gpu:4
#SBATCH --time=00:08:00
#SBATCH --output=logs/long_prefill_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/lib/config.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MOEFLOW_JOBID="$SLURM_JOB_ID"
export MOEFLOW_GREEDY_DP_RANK=0
export MOEFLOW_COMBINE_MODE=bf16_ring_fused
export MOEFLOW_FUSED_DISPATCH=1
export MOEFLOW_FUSED_EXPERT_PREP=1
OUT="results/stage1/long_prefill_$SLURM_JOB_ID"
GPUS=4
source scripts/lib/slurm_prologue.sh
MODEL="$MOEFLOW_MODEL_PATH"
export MOEFLOW_GREEDY_PROMPTS="$PWD/$OUT/prompts.json"
"$PYTHON" - "$MODEL" "$MOEFLOW_GREEDY_PROMPTS" <<'PY'
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
short = "The capital of France is"
filler = tokenizer.encode("MoE models route tokens to specialized experts. ", add_special_tokens=False)
suffix = tokenizer.encode("\nThe capital of France is", add_special_tokens=False)
prompts = [short]
for length in (1024, 8192):
    prefix = (filler * ((length + len(filler) - 1) // len(filler)))[:length - len(suffix)]
    prompts.extend([prefix + suffix, short])
with open(sys.argv[2], "w") as output:
    json.dump(prompts, output)
PY
timeout -s KILL 450 scripts/e2e_bench.sh "$MODEL" 4 "$OUT" 'native_agrs moeflow_fixed' '' \
  --max-num-seqs 4 --max-num-batched-tokens 512 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
"$PYTHON" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {}
for mode in ("native_agrs", "moeflow_fixed"):
    records = json.loads((out / f"greedy_{mode}.json").read_text())
    lengths = [row["prompt_tokens"] for row in records]
    assert len(records) == 5 and lengths[1::2] == [1024, 8192], lengths
    assert lengths[0] == lengths[2] == lengths[4] and 0 < lengths[0] <= 256, lengths
    assert all(row["tokens"] for row in records), "Empty output cannot pass the gate"
    summary[mode] = {"prompt_tokens": lengths, "output_tokens": [len(row["tokens"]) for row in records]}
comparison = json.loads((out / "compare_moeflow_fixed.json").read_text())
assert all(row["passed"] and row["max_abs_logprob_diff_on_prefix"] == 0 for row in comparison)
summary["complete_outputs_and_logprobs_match"] = True
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
