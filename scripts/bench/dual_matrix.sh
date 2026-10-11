#!/usr/bin/env bash
# Run the one-vs-two-GPU matrix: for each variant start the server, take a speed card, then sweep agents.
# usage: dual_matrix.sh OUTDIR "variant:levels variant:levels ..."   (e.g. "single:1,2,4,6,8 dp2:1,2,4,8,12,16")
# levels "-" takes the speed card only; "niah" runs the long-context retrieval test instead of the agent sweep.
# Each variant writes OUTDIR/<variant>/{server.txt,speed.txt,requests.jsonl,monitor.csv}; OUTDIR/matrix.done at the end.
set -uo pipefail
OUT=$1; PLAN=$2
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=$(cat /dd/neuron/configs/api-key)
mkdir -p "$OUT"; rm -f "$OUT/matrix.done"
for item in $PLAN; do
  V=${item%%:*}; LEVELS=${item#*:}; D="$OUT/$V"; mkdir -p "$D"
  echo "$(date +%T) == $V" | tee -a "$OUT/matrix.log"
  T0=$(date +%s)
  if ! "$HERE/bench-variant.sh" "$V" > "$D/start.txt" 2>&1; then
    cat "$D/start.txt" >> "$OUT/matrix.log"; cp "/dd/neuron/logs/bench-$V.log" "$D/server.log" 2>/dev/null; continue
  fi
  { cat "$D/start.txt"; grep -hE "max_total_num_tokens|max_running_requests|context_len|Load weight end|Capture cuda graph end|mamba|KV Cache is allocated" "/dd/neuron/logs/bench-$V.log" | tail -12; } > "$D/server.txt"
  {
    for i in 1 2 3; do python3 "$HERE/bench_chat.py" --api-key "$KEY" --concurrency 1 --max-tokens 1024 --nonce | tail -1; done
    python3 "$HERE/bench_chat.py" --api-key "$KEY" --concurrency 1 --max-tokens 64 --prompt-tokens 38000 --nonce | head -1
    for c in 4 8 16; do python3 "$HERE/bench_chat.py" --api-key "$KEY" --concurrency $c --max-tokens 1024 --nonce | tail -1; done
  } > "$D/speed.txt" 2>&1
  if [[ $LEVELS == niah ]]; then
    python3 "$HERE/niah.py" --tokens ${NIAH_TOKENS:-131072,262144,524288,800000,990000} --out "$D/niah.jsonl" >> "$OUT/matrix.log" 2>&1
  elif [[ $LEVELS != - ]]; then
    RK=""; [[ $V == dp2-sticky ]] && RK=--routing-key
    python3 "$HERE/agent_bench.py" --out "$D" --label "$V" --levels "$LEVELS" --duration 300 --warmup 60 $RK >> "$OUT/matrix.log" 2>&1
  fi
  for c in $(docker ps -a --format '{{.Names}}' | grep '^neuron-bench'); do echo "### $c"; docker logs "$c" 2>&1; done > "$D/server.log"
  echo "$(date +%T) == $V finished in $(( $(date +%s) - T0 )) s" >> "$OUT/matrix.log"
done
docker rm -f neuron-bench >/dev/null 2>&1
date +%T > "$OUT/matrix.done"
