#!/usr/bin/env python3
"""Per idle window of a soak_test.py run: P-state mix, median power and SM clock (skipping the first 60 s)."""
import csv
import datetime as dt
import sys
from collections import Counter

run = sys.argv[1]
skip = float(sys.argv[2]) if len(sys.argv) > 2 else 60
rows = [r for r in csv.DictReader(open(f"{run}/monitor.csv")) if r["temp_c"].strip().isdigit()]
events = [l.rstrip("\n").split(" ", 2) for l in open(f"{run}/events.log")]
marks = [(dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S").timestamp(), m) for d, t, m in events]
for i, (s, m) in enumerate(marks):
    if m != "idle":
        continue
    end = marks[i + 1][0] if i + 1 < len(marks) else s + 3600
    w = [r for r in rows if s + skip <= int(r["ts"]) < end]
    if not w:
        continue
    p = sorted(float(r["power_w"]) for r in w)[len(w) // 2]
    print(f"idle at {dt.datetime.fromtimestamp(s):%H:%M:%S}: n={len(w)} pstates={dict(Counter(r['pstate'] for r in w))} "
          f"power p50={p:.0f} W sm={Counter(r['sm_mhz'] for r in w).most_common(1)[0][0]} MHz")
