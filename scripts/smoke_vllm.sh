#!/usr/bin/env bash
# MoEFlowRunner integration smoke test: serve the trimmed model with DP attention + EP,
# send greedy text requests with vLLM's own runner ("native") and with MoEFlowRunner ("moeflow").
# Both modes pin the modular Triton expert backend so the runner can inject top-k (limitations L3).
# Usage: MOEFLOW_JOBID=<alloc> scripts/smoke_vllm.sh <model_dir> <dp_size> <out_dir> [route_check|fixed]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/config.sh
MODEL=$1; DP=$2; OUT=$3
RUNNER_MODE=${4:-route_check}
case "$RUNNER_MODE" in route_check|fixed) ;; *) echo "unknown runner mode: $RUNNER_MODE"; exit 2;; esac
JOB=${MOEFLOW_JOBID:?}
mkdir -p "$OUT/runner_log"
OUT=$(realpath "$OUT")
for mode in native moeflow; do
  PORT=$((20000 + (JOB * 13 + DP + ${#mode}) % 10000))
  RUNNER=$([ $mode = moeflow ] && echo 1 || echo 0)
  MOEFLOW_RUNNER=$RUNNER MOEFLOW_MODE=$RUNNER_MODE MOEFLOW_SMOKE_DIR=$OUT/runner_log VLLM_DEEP_GEMM_WARMUP=skip \
  timeout -s KILL 1800 srun --jobid="$JOB" --overlap -n1 --gres=gpu:"$DP" --export=ALL \
    "$VLLM" serve "$MODEL" --port "$PORT" --data-parallel-size "$DP" --enable-expert-parallel \
      --all2all-backend allgather_reducescatter --moe-backend triton --enforce-eager --max-model-len 32768 \
      --gpu-memory-utilization 0.6 --served-model-name smoke > "$OUT/server_$mode.log" 2>&1 &
  SERVER=$!
  ok=0
  for _ in $(seq 360); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then ok=1; break; fi
    kill -0 $SERVER 2>/dev/null || break
    sleep 5
  done
  if [ $ok = 1 ]; then
    "$PYTHON" scripts/smoke_client.py --port "$PORT" --out "$OUT/client_$mode.json" || ok=0
  fi
  kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true
  echo "mode=$mode server_ready_and_client_ok=$ok"
done
"$PYTHON" scripts/smoke_client.py --require-match --compare "$OUT/client_native.json" "$OUT/client_moeflow.json" > "$OUT/compare.json"
cat "$OUT/compare.json"
echo "moeflow_wrap_lines=$(grep -c 'FusedMoEFactory wrapped' "$OUT/server_moeflow.log" || true)"
echo "native_wrap_lines=$(grep -c 'FusedMoEFactory wrapped' "$OUT/server_native.log" || true)"
cat "$OUT"/runner_log/*.jsonl 2>/dev/null | "$PYTHON" -c '
import json, sys
rows = [json.loads(l) for l in sys.stdin]
icp = [r for r in rows if r.get("intercepted") and r.get("tokens")]
route_checks = [r for r in icp if "token_expert_set_match" in r]
print(json.dumps({"runner_calls_logged": len(rows), "intercepted_with_tokens": len(icp),
  "fixed_calls_logged": sum(r.get("mode") == "fixed" for r in icp),
  "not_intercepted": sum(1 for r in rows if not r.get("intercepted")),
  "min_token_expert_set_match": min((r["token_expert_set_match"] for r in route_checks), default=None),
  "max_abs_weight_diff_on_match": max((r["max_abs_weight_diff_on_match"] or 0 for r in route_checks), default=None),
  "layers": sorted({r["layer"] for r in rows})}))'
