#!/usr/bin/env python3
"""Coding-agent capacity benchmark: how many concurrent agents can one server carry?

Each simulated agent behaves like a typical coding agent (Claude Code, Codex CLI, opencode, pi, Cline...):
  * every session starts from the same ~9K-token agent system prompt with six tool definitions (shared
    across agents, so it is a prefix-cache hit after the first request), plus a ~3K-token session-specific
    repository map and task;
  * each turn the model replies (tool calls or text, up to 512 tokens), the client "executes" the tools for
    1-3 s and appends 1-4K tokens of tool output (synthetic code), so the context grows ~3K tokens per turn;
  * when the context passes a per-session cap (40-80K tokens) the agent starts a new session, the way agents
    compact or start over.
The run is closed-loop: N agents loop for --duration seconds per level, after a --warmup that is not scored.

usage: agent_bench.py --out DIR --label NAME [--levels 1,2,4,8] [--duration 300] [--warmup 60] [--routing-key]
Writes DIR/requests.jsonl (one row per turn) and DIR/monitor.csv; summarize with agent_report.py.
"""
import argparse
import csv
import json
import random
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

LOCK = threading.Lock()

TOOL_NAMES = ["read_file", "write_file", "edit_file", "run_shell", "grep", "list_dir"]
TOOLS = [{"type": "function", "function": {"name": n, "description": f"{n.replace('_', ' ').capitalize()} in the workspace. "
          "Paths are relative to the repository root. Output longer than 40,000 characters is truncated.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "arg": {"type": "string"}},
                         "required": ["path"]}}} for n in TOOL_NAMES]
TASKS = [
    "The tests in tests/test_scheduler.py fail intermittently. Find the cause and fix it.",
    "Add retry with exponential backoff and jitter to the HTTP client in src/client.py.",
    "Memory grows over time in src/worker.py. Find the leak and patch it.",
    "Refactor src/config.py to load settings from environment variables with typed defaults.",
    "Add a --dry-run flag to the CLI in src/cli.py and thread it through to the executor.",
    "The cache in src/cache.py returns stale entries after a write. Fix the invalidation.",
]


def synthetic_code(n_tokens, seed):
    rnd = random.Random(seed)
    out = []
    for i in range(max(1, n_tokens // 48)):  # ~48 tokens per line pair
        k = rnd.randint(0, 10**6)
        out.append(f"def fn_{seed % 10**6}_{i}(items, limit={k % 97}):\n    return [x * {k % 13} for x in items if x % {k % 7 + 2} == 0][:limit]\n")
    return "\n".join(out)


SYSTEM = ("You are a coding agent working in a git repository. Inspect the code with the tools before changing "
          "anything, keep edits minimal, run the tests after each change, and report what you changed.\n\n"
          "Repository conventions and shared library code:\n" + synthetic_code(9000, 7))


class Bench:
    def __init__(self, args):
        self.args, self.key = args, Path(args.key_file).read_text().strip()
        self.level, self.score_from, self.score_to = 0, 0.0, 0.0

    def chat(self, messages, session=None):
        body = {"model": self.args.model, "messages": messages, "tools": TOOLS, "max_tokens": 512, "stream": True,
                "temperature": 0.7, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        rec = {"ts": time.time(), "level": self.level, "label": self.args.label, "ok": False}
        msg = {"role": "assistant", "content": "", "tool_calls": []}
        t0, first = time.time(), None
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
            if self.args.routing_key and session:
                headers["X-SGL-Routing-Key"] = session  # session affinity for sglang-router's manual policy
            req = urllib.request.Request(self.args.url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if d.get("usage"):
                        u = d["usage"]
                        rec.update(prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"],
                                   cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                    if not d.get("choices"):
                        continue
                    delta = d["choices"][0].get("delta", {})
                    if first is None and (delta.get("content") or delta.get("tool_calls")):
                        first = time.time()
                    msg["content"] += delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        i = tc.get("index", 0)
                        while len(msg["tool_calls"]) <= i:
                            msg["tool_calls"].append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        slot = msg["tool_calls"][i]
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["function"]["name"] += fn.get("name") or ""
                        slot["function"]["arguments"] += fn.get("arguments") or ""
            end = time.time()
            rec.update(ok=True, ttft=(first or end) - t0, total=end - t0)
            if first and rec.get("completion_tokens", 0) > 1 and end > first:
                rec["decode_tps"] = (rec["completion_tokens"] - 1) / (end - first)
            rec["tool_calls"] = len(msg["tool_calls"])
        except Exception as e:  # noqa: BLE001 - failures are data
            detail = e.read().decode(errors="replace")[:300] if hasattr(e, "read") else ""
            rec.update(error=f"{type(e).__name__}: {e} {detail}"[:600], total=time.time() - t0)
        if not msg["tool_calls"]:
            del msg["tool_calls"]
        if self.score_from <= rec["ts"] < self.score_to:
            with LOCK, open(self.args.out / "requests.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        return msg, rec

    def agent(self, aid, until):
        rnd = random.Random(hash((self.args.label, self.level, aid, time.time())))
        time.sleep(rnd.uniform(0, min(30, self.args.warmup / 2)))  # stagger session starts
        while time.time() < until:
            seed = rnd.randint(0, 10**9)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Repository map:\n" + synthetic_code(3000, seed) + "\n\nTask: " + rnd.choice(TASKS)}]
            cap, ctx = rnd.randint(40_000, 80_000), 0
            while ctx < cap and time.time() < until:
                reply, rec = self.chat(msgs, session=f"{self.args.label}-{aid}-{seed}")
                if not rec["ok"]:
                    time.sleep(5)
                    break
                ctx = rec.get("prompt_tokens", ctx) + rec.get("completion_tokens", 0)
                msgs.append(reply)
                time.sleep(rnd.uniform(1, 3))  # tool execution
                for tc in reply.get("tool_calls", []):
                    msgs.append({"role": "tool", "tool_call_id": tc["id"] or "call",
                                 "content": synthetic_code(rnd.randint(1000, 4000), rnd.randint(0, 10**9))})
                if not reply.get("tool_calls"):
                    msgs.append({"role": "user", "content": "Now check the related module:\n" + synthetic_code(rnd.randint(1000, 2000), rnd.randint(0, 10**9))})


def monitor(args, stop):
    path = args.out / "monitor.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "label", "gpu", "temp_c", "fan_pct", "power_w", "pstate", "sm_mhz", "util", "vram_mib", "throttle", "psu_in_w"])
        while not stop.is_set():
            psu = ""
            try:
                j = json.loads(subprocess.run(["sensors", "-j"], capture_output=True, text=True, timeout=10).stdout or "{}")
                for chip, v in j.items():
                    if chip.startswith("corsairpsu"):
                        psu = next((x["power1_input"] for k, x in v.items() if k.startswith("power total") and "power1_input" in x), "")
            except Exception:  # noqa: BLE001
                pass
            smi = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,fan.speed,power.draw,pstate,clocks.sm,utilization.gpu,memory.used,clocks_event_reasons.active",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            now = int(time.time())
            for row in smi.strip().splitlines():
                w.writerow([now, args.label, *[c.strip() for c in row.split(",")], psu])
            f.flush()
            stop.wait(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--warmup", type=float, default=60)
    ap.add_argument("--url", default="http://localhost:30000/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--key-file", default="/dd/neuron/configs/api-key")
    ap.add_argument("--routing-key", action="store_true", help="send X-SGL-Routing-Key per agent session")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    b, stop = Bench(args), threading.Event()
    threading.Thread(target=monitor, args=(args, stop), daemon=True).start()
    for n in [int(x) for x in args.levels.split(",")]:
        b.level = n
        start = time.time()
        b.score_from, b.score_to = start + args.warmup, start + args.warmup + args.duration
        print(f"{time.strftime('%H:%M:%S')} {args.label}: {n} agents", flush=True)
        ts = [threading.Thread(target=b.agent, args=(i, b.score_to)) for i in range(n)]
        [t.start() for t in ts]
        [t.join() for t in ts]  # in-flight turns finish but are not scored
    stop.set()
    print(f"{time.strftime('%H:%M:%S')} {args.label}: done", flush=True)


if __name__ == "__main__":
    main()
