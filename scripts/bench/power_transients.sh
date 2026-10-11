#!/usr/bin/env bash
# Compare coincident GPU power transients under TP=2 and PP=2 with the PSU's own telemetry.
# For each variant: start it, apply load (LOAD=decode: 16 concurrent long generations; LOAD=agents: 8 agents
# from agent_bench.py, which is prefill-heavy), and for 120 s sample
#   - nvidia-smi power.draw.instant for both GPUs every 100 ms, and
#   - the PSU's reported output power (corsair-psu hwmon) every 1 s.
# usage: power_transients.sh OUTDIR [variants...]   (default: tp2 pp2). Writes OUTDIR/<variant>-{gpu,psu}.csv
set -uo pipefail
OUT=$1; shift; VARIANTS=${*:-tp2 pp2}
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=$(cat /dd/neuron/configs/api-key)
mkdir -p "$OUT"
for V in $VARIANTS; do
  "$HERE/bench-variant.sh" "$V" || continue
  if [[ ${LOAD:-decode} == agents ]]; then
    python3 "$HERE/agent_bench.py" --out "$OUT/$V-agents" --label "$V" --levels 8 --duration 120 --warmup 60 > "$OUT/$V-load.txt" 2>&1 &
    PID=$!; sleep 60
  else
    python3 "$HERE/bench_chat.py" --api-key "$KEY" --concurrency 16 --max-tokens 4096 --nonce > "$OUT/$V-load.txt" 2>&1 &
    PID=$!; sleep 20
  fi
  timeout 120 nvidia-smi --query-gpu=timestamp,index,power.draw.instant --format=csv,noheader,nounits -lms 100 > "$OUT/$V-${LOAD:-decode}-gpu.csv" &
  for _ in $(seq 1 120); do
    echo "$(date +%s.%N),$(sensors -j corsairpsu-hid-3-3 | python3 -c 'import json,sys; print(json.load(sys.stdin)["corsairpsu-hid-3-3"]["power total"]["power1_input"])')"
    sleep 1
  done > "$OUT/$V-${LOAD:-decode}-psu.csv"
  wait $PID
done
docker rm -f neuron-bench neuron-bench-w0 neuron-bench-w1 >/dev/null 2>&1
touch "$OUT/done"
