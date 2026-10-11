#!/usr/bin/env bash
# Does a periodic 1-token health generation keep an idle GPU out of P8?
# Runs N-minute idle phases with /health polled every 0 (none), 60, 30, 15 s, sampling P-state every 5 s.
# usage: idle_wake_test.sh [minutes-per-phase]
set -uo pipefail
MIN=${1:-6}
OUT=/dd/neuron/bench/results/idle-wake-$(date +%H%M).csv
echo "phase,ts,pstate,power_w,sm_mhz,util" > "$OUT"
for interval in ${PHASES:-0 30 15 10}; do  # extra /health polling, on top of the container healthcheck
  end=$((SECONDS + MIN * 60))
  next=$SECONDS
  while [ $SECONDS -lt $end ]; do
    if [ "$interval" -gt 0 ] && [ $SECONDS -ge $next ]; then
      curl -s -o /dev/null --max-time 20 localhost:30000/health &
      next=$((SECONDS + interval))
    fi
    echo "$interval,$(date +%s),$(nvidia-smi --query-gpu=pstate,power.draw,clocks.sm,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" >> "$OUT"
    sleep 5
  done
done
echo "$OUT"
python3 - "$OUT" <<'EOF'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
by = defaultdict(list)
for r in rows:
    by[r["phase"]].append(r)
for k, rs in by.items():
    rs = rs[12:]  # drop the first minute of each phase
    p2 = sum(r["pstate"] == "P2" for r in rs) / len(rs)
    pw = sorted(float(r["power_w"]) for r in rs)[len(rs) // 2]
    print(f"health every {k:>2}s extra (on top of the container healthcheck): P2 share {p2:.0%}, power p50 {pw:.1f} W")
EOF
