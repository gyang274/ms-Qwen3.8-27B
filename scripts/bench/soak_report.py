#!/usr/bin/env python3
"""Summarize a soak_test.py run: per-kind request stats, hourly drift, GPU/container trends, probe consistency.

usage: soak_report.py RUN_DIR   (also prints journal/docker log checks if run on the server)
"""
import csv
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def num(x):
    try:
        return float(str(x).rstrip("%"))
    except ValueError:
        return None


def main():
    run = Path(sys.argv[1])
    reqs = [json.loads(l) for l in open(run / "requests.jsonl")]
    t0 = min(r["ts"] for r in reqs)
    hours = int((max(r["ts"] for r in reqs) - t0) // 3600) + 1

    print(f"# Soak report: {run.name}\n\n{len(reqs)} requests over ~{hours} h\n")
    print("## Requests by kind\n\n| kind | n | errors | TTFT p50 / p95 (s) | decode p50 (tok/s) | max prompt tokens |\n|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in reqs:
        by[r["kind"]].append(r)
    for k, rs in sorted(by.items()):
        ok = [r for r in rs if r["ok"]]
        tt = [r["ttft"] for r in ok if "ttft" in r]
        dt = [r["decode_tps"] for r in ok if r.get("decode_tps")]
        mp = max((r.get("prompt_tokens", 0) for r in ok), default=0)
        print(f"| {k} | {len(rs)} | {len(rs) - len(ok)} | {pct(tt, 50):.2f} / {pct(tt, 95):.2f} | {pct(dt, 50):.1f} | {mp} |")
    errs = [r for r in reqs if not r["ok"]]
    tc = sum(r.get("tool_calls", 0) for r in reqs)
    bad = sum(r.get("bad_tool_args", 0) for r in reqs)
    print(f"\nerrors: {len(errs)}" + (f" (first: {errs[0].get('error')})" if errs else "") + f"; tool calls {tc}, malformed arguments {bad}\n")

    print("## Hourly drift (robot + coding, single-stream-ish)\n\n| hour | requests | errors | robot TTFT p50 | decode p50 |\n|---|---|---|---|---|")
    for h in range(hours):
        rs = [r for r in reqs if h * 3600 <= r["ts"] - t0 < (h + 1) * 3600]
        robot = [r["ttft"] for r in rs if r["ok"] and r["kind"].startswith("robot")]
        dec = [r["decode_tps"] for r in rs if r["ok"] and r.get("decode_tps") and r["kind"] in ("robot", "coding", "probe")]
        print(f"| {h + 1} | {len(rs)} | {sum(not r['ok'] for r in rs)} | {pct(robot, 50):.2f} | {pct(dec, 50):.1f} |")

    rows = list(csv.DictReader(open(run / "monitor.csv")))
    rows = [r for r in rows if num(r.get("temp_c")) is not None]
    temp = [num(r["temp_c"]) for r in rows]
    fan = [num(r["fan_pct"]) for r in rows]
    vram = [num(r["vram_mib"]) for r in rows]
    hostm = [num(r["host_mem_used_mib"]) for r in rows]
    idle = [r for r in rows if num(r["gpu_util"]) == 0 and num(r["power_w"]) < 40]
    restarts = {r["restarts"] for r in rows}
    unhealthy = [r for r in rows if r["health"] != "healthy"]
    thermal = [r for r in rows if int(r["throttle"], 16) & 0x60]
    q = max(1, len(rows) // 10)
    print("\n## GPU / container\n")
    print(f"- temperature max {max(temp):.0f} C, p95 {pct(temp, 95):.0f} C; fan max {max(fan):.0f}%")
    print(f"- thermal slowdown samples: {len(thermal)}")
    print(f"- VRAM first 10% avg {st.mean(vram[:q]):.0f} MiB, last 10% avg {st.mean(vram[-q:]):.0f} MiB, max {max(vram):.0f}")
    print(f"- host RAM used first 10% avg {st.mean(hostm[:q]):.0f} MiB, last 10% avg {st.mean(hostm[-q:]):.0f} MiB")
    print(f"- container restart count values seen: {sorted(restarts)}; non-healthy samples: {len(unhealthy)}")
    if idle:
        print(f"- idle samples {len(idle)}: pstates {sorted({r['pstate'] for r in idle})}, power p50 {pct([num(r['power_w']) for r in idle], 50):.1f} W, "
              f"container CPU p50 {pct([num(r['ctr_cpu_pct']) for r in idle], 50):.1f}%")

    pf = run / "probes.jsonl"
    probes = [json.loads(l) for l in open(pf)] if pf.exists() else []
    texts = {p["text"] for p in probes if p["ok"]}
    print(f"\n## Consistency probe\n\n- {len(probes)} probes, {sum(p['ok'] for p in probes)} ok, {len(texts)} distinct answer(s)")
    for t in texts:
        print(f"  - {t[:120]!r}")

    print("\n## System logs during the run\n")
    since = f"@{int(t0)}"
    checks = {
        "kernel Xid / AER / MCE": f"journalctl -k --since {since} --no-pager | grep -ciE 'xid|aer:|machine check|hardware error' || true",
        "fan health warnings": f"journalctl -u gpu-fan-curve --since {since} --no-pager | grep -c WARNING || true",
        "sglang errors/tracebacks": f"docker logs --since {int(t0)} neuron-sglang 2>&1 | grep -ciE 'traceback|error|out of memory' || true",
        "OOM killer": f"journalctl -k --since {since} --no-pager | grep -ci 'out of memory' || true",
        "apt/unattended activity": f"grep -c 'Packages that will be upgraded' /var/log/unattended-upgrades/unattended-upgrades.log 2>/dev/null || echo 0",
    }
    for name, cmd in checks.items():
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()
        print(f"- {name}: {out}")


if __name__ == "__main__":
    main()
