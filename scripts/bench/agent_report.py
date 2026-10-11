#!/usr/bin/env python3
"""Summarize agent_bench.py runs: one row per (label, agents).

usage: agent_report.py DIR [DIR...] [--duration 300] [--json out.json]
Columns: turns scored, turns per agent-minute, aggregate output and new-prefill tok/s, per-stream decode
p50/p10, TTFT p50/p90, turn latency p50/p90, prefix-cache hit share of prompt tokens, errors.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--json")
    args = ap.parse_args()
    groups = defaultdict(list)
    for d in args.dirs:
        for line in open(d / "requests.jsonl"):
            r = json.loads(line)
            groups[(r["label"], r["level"])].append(r)
    rows = []
    for (label, n), rs in sorted(groups.items()):
        ok = [r for r in rs if r["ok"]]
        prompt = sum(r.get("prompt_tokens", 0) for r in ok)
        cached = sum(r.get("cached_tokens", 0) for r in ok)
        row = {
            "label": label, "agents": n, "turns": len(ok), "errors": len(rs) - len(ok),
            "turns_per_agent_min": len(ok) / n / (args.duration / 60),
            "out_tps": sum(r.get("completion_tokens", 0) for r in ok) / args.duration,
            "new_prefill_tps": (prompt - cached) / args.duration,
            "decode_p50": statistics.median([r["decode_tps"] for r in ok if "decode_tps" in r] or [float("nan")]),
            "decode_p10": pct([r["decode_tps"] for r in ok if "decode_tps" in r], 10),
            "ttft_p50": pct([r["ttft"] for r in ok], 50), "ttft_p90": pct([r["ttft"] for r in ok], 90),
            "turn_p50": pct([r["total"] for r in ok], 50), "turn_p90": pct([r["total"] for r in ok], 90),
            "cache_hit": cached / prompt if prompt else float("nan"),
            "ctx_p50": pct([r.get("prompt_tokens", 0) for r in ok], 50),
        }
        rows.append(row)
    hdr = f"{'label':<14}{'N':>3}{'turns':>6}{'err':>4}{'t/ag/min':>9}{'out t/s':>8}{'prefill':>8}{'dec p50':>8}{'dec p10':>8}{'ttft50':>7}{'ttft90':>7}{'turn50':>7}{'turn90':>7}{'cache':>6}{'ctx50':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['label']:<14}{r['agents']:>3}{r['turns']:>6}{r['errors']:>4}{r['turns_per_agent_min']:>9.2f}{r['out_tps']:>8.0f}{r['new_prefill_tps']:>8.0f}"
              f"{r['decode_p50']:>8.1f}{r['decode_p10']:>8.1f}{r['ttft_p50']:>7.1f}{r['ttft_p90']:>7.1f}{r['turn_p50']:>7.1f}{r['turn_p90']:>7.1f}{r['cache_hit']:>6.0%}{r['ctx_p50']:>7}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
