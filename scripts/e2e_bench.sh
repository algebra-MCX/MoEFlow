#!/usr/bin/env bash
# End-to-end comparison in vLLM: one server per mode, same model/flags otherwise.
#   MOEFLOW_JOBID=<alloc> scripts/e2e_bench.sh <model_dir> <dp> <out_dir> [modes] [workloads] [server args...]
# An explicit empty workloads argument runs only the greedy correctness probe.
# modes:     native_agrs native_deepep_ll moeflow_fixed native_default   (default: first three)
# workloads: W1..W6 (see docs/architecture.md)
# Every server: DP attention + EP, text only, max-model-len 32768, FP8 KV cache (official recipe),
# CUDA graphs on. Expert GEMM = vLLM Triton FP8 in all three compared modes.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib/config.sh
MODEL=$1; DP=$2; OUT=$3
MODES=${4:-"native_agrs native_deepep_ll moeflow_fixed"}
WORKLOADS=${5-"W1 W2 W3 W4 W5 W6"}
SERVER_ARGS=("${@:6}")
JOB=${MOEFLOW_JOBID:?}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
TRACE_DIR=${MOEFLOW_ROUTING_TRACE_DIR:-}
if [ -n "$TRACE_DIR" ]; then
  eager=0
  for arg in "${SERVER_ARGS[@]}"; do if [ "$arg" = --enforce-eager ]; then eager=1; fi; done
  if [ "$eager" != 1 ]; then echo "Routing traces require --enforce-eager"; exit 2; fi
  mkdir -p "$TRACE_DIR"
fi
failed=0
REFERENCE=${MOEFLOW_REFERENCE_JSON:-}
LOGITS_REFERENCE=""
PROBE_ARGS=()
COMPARE_ARGS=(--require-match)
if [ "${MOEFLOW_EXACT_GREEDY:-0}" = 1 ]; then COMPARE_ARGS+=(--require-exact-logprobs); fi
if [ -n "${MOEFLOW_GREEDY_DP_RANK:-}" ]; then PROBE_ARGS+=(--data-parallel-rank "$MOEFLOW_GREEDY_DP_RANK"); fi
if [ -n "${MOEFLOW_GREEDY_DP_RANKS:-}" ]; then
  read -ra PROBE_RANKS <<< "$MOEFLOW_GREEDY_DP_RANKS"
  PROBE_ARGS+=(--data-parallel-ranks "${PROBE_RANKS[@]}")
fi
if [ -n "${MOEFLOW_GREEDY_REQUESTS:-}" ]; then PROBE_ARGS+=(--requests "$MOEFLOW_GREEDY_REQUESTS"); fi
if [ -n "${MOEFLOW_GREEDY_PROMPTS:-}" ]; then PROBE_ARGS+=(--prompts-file "$MOEFLOW_GREEDY_PROMPTS"); fi

declare -A ISL=([W1]=1024 [W2]=1024 [W3]=1024 [W4]=1024 [W5]=8192 [W6]=8192)
declare -A OSL=([W1]=1024 [W2]=1024 [W3]=8192 [W4]=8192 [W5]=1024 [W6]=1024)
declare -A CONC=([W1]=4 [W2]=128 [W3]=4 [W4]=64 [W5]=4 [W6]=64)
declare -A NPROMPT=([W1]=16 [W2]=256 [W3]=8 [W4]=64 [W5]=16 [W6]=128)

for mode in $MODES; do
  ENVS=(VLLM_DEEP_GEMM_WARMUP=skip MOEFLOW_RUNNER=0 MOEFLOW_ALL2ALL=0)
  case $mode in
    native_agrs)      ARGS=(--all2all-backend allgather_reducescatter --moe-backend triton) ;;
    native_deepep_ll) ARGS=(--all2all-backend deepep_low_latency --moe-backend triton) ;;
    moeflow_fixed)     ARGS=(--all2all-backend allgather_reducescatter --moe-backend triton)
                      ENVS+=(MOEFLOW_RUNNER=1 MOEFLOW_MODE=fixed MOEFLOW_SMOKE_DIR="$OUT/runner_log_$mode") ;;
    native_default)   ARGS=() ;;
    *) echo "unknown mode $mode"; exit 2 ;;
  esac
  mkdir -p "$OUT/runner_log_$mode"
  if [ -n "$TRACE_DIR" ]; then rm -f "$TRACE_DIR/enabled"; fi
  PORT=$((20000 + (JOB * 17 + ${#mode} * 7) % 10000))
  srun --jobid="$JOB" --overlap -n1 --gres=gpu:"$DP" nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    > "$OUT/$mode.preflight_procs.txt"
  /usr/bin/env "${ENVS[@]}" timeout -s KILL 7200 srun --jobid="$JOB" --overlap -n1 --gres=gpu:"$DP" --export=ALL \
    "$VLLM" serve "$MODEL" --port "$PORT" --served-model-name glm --data-parallel-size "$DP" \
      --enable-expert-parallel --max-model-len 32768 --kv-cache-dtype fp8 \
      --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.85 "${ARGS[@]}" "${SERVER_ARGS[@]}" \
      > "$OUT/server_$mode.log" 2>&1 &
  SERVER=$!
  ready=0
  for _ in $(seq 360); do  # 1800 s model-load watchdog
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then ready=1; break; fi
    kill -0 $SERVER 2>/dev/null || break
    sleep 5
  done
  echo "mode=$mode ready=$ready"
  if [ $ready = 1 ]; then
    if [ -n "$TRACE_DIR" ]; then touch "$TRACE_DIR/enabled"; fi
    "$PYTHON" scripts/smoke_client.py --port "$PORT" --model glm --max-tokens 64 \
      "${PROBE_ARGS[@]}" --out "$OUT/greedy_$mode.json" > "$OUT/greedy_$mode.log" 2>&1
    greedy_rc=$?
    if [ -n "$TRACE_DIR" ]; then rm -f "$TRACE_DIR/enabled"; fi
    echo "mode=$mode greedy_exit=$greedy_rc"
    if [ "$greedy_rc" -ne 0 ]; then
      ready=0
    elif [ -n "$REFERENCE" ]; then
      if ! "$PYTHON" scripts/smoke_client.py "${COMPARE_ARGS[@]}" --compare \
        "$REFERENCE" "$OUT/greedy_$mode.json" > "$OUT/compare_$mode.json" 2> "$OUT/compare_$mode.log"; then
        ready=0
      fi
    elif [[ "$mode" == native_* ]]; then
      REFERENCE="$OUT/greedy_$mode.json"
    else
      echo "A native greedy reference is required before benchmarking $mode"
      ready=0
    fi
  fi
  if [ "$ready" = 1 ] && [ -n "${MOEFLOW_LOGITS_VOCAB_SIZE:-}" ]; then
    LOGITS_ARGS=()
    if [ -n "${MOEFLOW_GREEDY_DP_RANK:-}" ]; then
      LOGITS_ARGS+=(--data-parallel-rank "$MOEFLOW_GREEDY_DP_RANK")
    fi
    if ! "$PYTHON" scripts/logits_client.py --port "$PORT" --model glm \
      --vocab-size "$MOEFLOW_LOGITS_VOCAB_SIZE" --steps "${MOEFLOW_LOGITS_STEPS:-8}" \
      "${LOGITS_ARGS[@]}" --out "$OUT/logits_$mode.json" > "$OUT/logits_$mode.log" 2>&1; then
      ready=0
    elif [ -n "$LOGITS_REFERENCE" ]; then
      if ! "$PYTHON" scripts/logits_client.py --compare "$LOGITS_REFERENCE" "$OUT/logits_$mode.json" \
        --out "$OUT/compare_logits_$mode.json" > "$OUT/compare_logits_$mode.log" 2>&1; then ready=0; fi
    elif [[ "$mode" == native_* ]]; then
      LOGITS_REFERENCE="$OUT/logits_$mode.json"
    else
      echo "A native raw-logits reference is required before benchmarking $mode"
      ready=0
    fi
  fi
  if [ "$ready" = 1 ]; then
    for w in $WORKLOADS; do
      timeout -s KILL 3600 "$VLLM" bench serve --backend vllm --port "$PORT" --model glm --tokenizer "$MODEL" \
        --dataset-name random --random-input-len "${ISL[$w]}" --random-output-len "${OSL[$w]}" --ignore-eos \
        --num-prompts "${NPROMPT[$w]}" --max-concurrency "${CONC[$w]}" --num-warmups 2 --seed 0 --temperature 0 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
        --metadata "moeflow_git=$(git rev-parse HEAD)" "moeflow_mode=$mode" "moeflow_workload=$w" \
        --save-result --save-detailed --result-dir "$OUT" --result-filename "${mode}_${w}.json" \
        > "$OUT/${mode}_${w}.log" 2>&1
      rc=$?
      if [ "$rc" = 0 ]; then
        "$PYTHON" scripts/validate_serving_result.py "$OUT/${mode}_${w}.json" \
          --requests "${NPROMPT[$w]}" --input-len "${ISL[$w]}" --output-len "${OSL[$w]}" \
          --concurrency "${CONC[$w]}" > "$OUT/${mode}_${w}.validation.json" 2> "$OUT/${mode}_${w}.validation.log"
        rc=$?
      fi
      printf '%s\n' "$rc" > "$OUT/${mode}_${w}.exitcode"
      echo "mode=$mode workload=$w exit=$rc"
      [ "$rc" -ne 0 ] && failed=1
    done
  else
    failed=1
    echo "mode=$mode correctness_or_readiness_failed=1; performance skipped"
  fi
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  sleep 10
done
exit "$failed"
