#!/usr/bin/env bash
# Runs experiments 1-3 against vLLM on one NVIDIA GPU, restarting the
# server with the right flags for each configuration, then draws charts.
#
#   pip install vllm httpx matplotlib
#   bash experiments/tail/run_vllm_experiments.sh                 # all three
#   EXPS="2" bash experiments/tail/run_vllm_experiments.sh        # just one
#   MODEL=meta-llama/Llama-3.2-3B-Instruct bash experiments/tail/run_vllm_experiments.sh
#
# Takes roughly 30-60 minutes on an L4 / A10G with the default model.
# vLLM renames flags between releases; if one is rejected, check
# `vllm serve --help` and adjust the *_ARGS lines below.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
EXPS="${EXPS:-1 2 3}"
OUT="${OUT:-results/tail}"
BASE="http://localhost:${PORT}"
mkdir -p "$OUT/logs"

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

start_server() {  # start_server <label> <extra vllm args...>
  local label="$1"; shift
  stop_server
  echo ">>> starting vLLM [$label]: $*"
  vllm serve "$MODEL" --port "$PORT" "$@" > "$OUT/logs/server_${label}.log" 2>&1 &
  SERVER_PID=$!
  python - "$BASE" <<'PY'
import asyncio, sys
from experiments.tail.client import wait_healthy
asyncio.run(wait_healthy(sys.argv[1]))
PY
}

# record the hardware alongside the results
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c "import vllm; print('vllm', vllm.__version__)"; } \
  > "$OUT/environment.txt" 2>&1 || true

for e in $EXPS; do
  case "$e" in
    1)
      start_server default --max-model-len 4096
      python -m experiments.tail.exp1_load_sweep --base-url "$BASE" --label default \
        --rates 1 2 4 8 12 16 20 24 --duration-s 60 --out "$OUT/exp1_load_sweep.json"
      ;;
    2)
      for budget in 512 16384; do
        start_server "budget-$budget" --max-model-len 16384 --max-num-batched-tokens "$budget"
        python -m experiments.tail.exp2_prefill_stall --base-url "$BASE" --label "budget-$budget" \
          --out "$OUT/exp2_budget-$budget.json"
      done
      ;;
    3)
      for blocks in 8192 2048 1024 512; do
        start_server "blocks-$blocks" --max-model-len 4096 --num-gpu-blocks-override "$blocks"
        python -m experiments.tail.exp3_kv_pressure --base-url "$BASE" --label "blocks-$blocks" \
          --kv-blocks "$blocks" --out "$OUT/exp3_blocks-$blocks.json"
      done
      ;;
  esac
done
stop_server

python -m experiments.tail.plot --results "$OUT" --theme light
python -m experiments.tail.plot --results "$OUT" --theme dark
echo "Done. Results in $OUT/, charts in $OUT/figs-light and $OUT/figs-dark"
