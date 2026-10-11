#!/usr/bin/env bash
# Sweep GPU power limits under sustained load; log temperature, fan, clocks, decode and prefill speed.
# usage: power_sweep.sh [watts...]   (default: 300 280 250 220); restores 300 W at the end
set -uo pipefail
cd "$(dirname "$0")"
if [ $# -gt 0 ]; then LIMITS=("$@"); else LIMITS=(300 280 250 220); fi
OUT=power-sweep-$(date +%Y%m%d-%H%M).log
B="python3 -I bench_chat.py --nonce"

for w in "${LIMITS[@]}"; do
  sudo nvidia-smi -pl "$w" >/dev/null
  echo "=== power limit ${w} W ===" | tee -a "$OUT"
  # sustained decode load (~4 min): 4 concurrent streams, repeated
  end=$((SECONDS + 240))
  ( while [ $SECONDS -lt $end ]; do $B --concurrency 4 --max-tokens 2048 >> "$OUT.decode.$w" 2>&1; done ) &
  load=$!
  for i in $(seq 1 24); do
    sleep 10
    nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,clocks.sm,clocks.mem,clocks_throttle_reasons.active --format=csv,noheader >> "$OUT.smi.$w"
  done
  wait $load
  echo "decode (C=4):   $(grep aggregate "$OUT.decode.$w" | awk '{s+=$NF; n++} END {printf "per-run aggregate avg %.1f tok/s over %d runs", s/n, n}')" | tee -a "$OUT"
  echo "decode (C=1):   $($B --max-tokens 1024 | tail -1)" | tee -a "$OUT"
  echo "prefill (38K):  $($B --prompt-tokens 30000 --max-tokens 16 | head -1)" | tee -a "$OUT"
  echo "steady state (last 60 s): $(tail -6 "$OUT.smi.$w" | awk -F', ' '{t+=$1; f+=$2; p+=$3; c+=$4} END {printf "temp %.0f C, fan %.0f %%, power %.0f W, sm %.0f MHz", t/NR, f/NR, p/NR, c/NR}')" | tee -a "$OUT"
  echo "max temp: $(sort -n "$OUT.smi.$w" | tail -1 | cut -d, -f1) C" | tee -a "$OUT"
done
sudo nvidia-smi -pl 300 >/dev/null
echo "restored 300 W" | tee -a "$OUT"
