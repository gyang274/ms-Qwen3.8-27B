#!/usr/bin/env python3
"""Mixed-workload soak test for the neuron SGLang server.

Each hour-long cycle: coding-agent sessions with tool calls and growing context (min 0-35),
home-robot short chats incl. an image (min 0-35), a 6-request burst (min 35), a temperature-0
consistency probe (min 38), then 20 min idle. Some cycles add a ~150K-token prefill or replace the
coding phase with 30 min of full load. A monitor samples GPU/container state every 30 s.

Writes <out>/{requests.jsonl, monitor.csv, probes.jsonl, events.log}; analyze with soak_report.py.

usage: soak_test.py --out DIR [--hours 10] [--model qwen3.8-27b] [--minute-seconds 60]
       (--minute-seconds 1 compresses each 'minute' to 1 s for a dry run)
"""
import argparse
import base64
import csv
import json
import random
import struct
import subprocess
import threading
import time
import urllib.request
import zlib
from pathlib import Path

URL = "http://localhost:30000/v1/chat/completions"
LOCK = threading.Lock()
STOP = threading.Event()

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
CODING_TASKS = [
    "The tests in tests/test_scheduler.py are failing intermittently. Investigate and propose a fix.",
    "Add retry with exponential backoff to the HTTP client in src/client.py. Look at the code first.",
    "Find why memory grows over time in the worker process (src/worker.py) and suggest a patch.",
    "Refactor src/config.py to load settings from environment variables with defaults. Read it first.",
]
ROBOT_PROMPTS = [
    "小智，现在提醒我下午三点给奶奶打电话。",
    "小智，给孩子讲一个两分钟的睡前故事，主角是一只小刺猬。",
    "小智，冰箱里有鸡蛋、西红柿和米饭，晚饭能做什么？",
    "小智，客厅温度28度，湿度70%，要不要开空调除湿？",
    "Xiaozhi, what's a good 10-minute stretching routine for the morning?",
    "小智，9乘以7再减去15等于多少？",
]
PROBE = "List the first 12 prime numbers, then give their sum. Answer concisely."


def log_event(out, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOCK, open(out / "events.log", "a") as f:
        f.write(line + "\n")


def synthetic_code(n_tokens, seed):
    rnd = random.Random(seed)
    lines = []
    for i in range(max(1, n_tokens // 48)):  # ~48 tokens per generated line
        k = rnd.randint(0, 10**6)
        lines.append(f"def fn_{seed}_{i}(items, limit={k % 97}):\n    return [x * {k % 13} for x in items if x % {k % 7 + 2} == 0][:limit]\n")
    return "\n".join(lines)


def png_image():
    """Tiny 64x64 PNG: left half red, right half blue."""
    w = h = 64
    raw = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff") for x in range(w)) for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


class Client:
    def __init__(self, args):
        self.args, self.key = args, Path(args.key_file).read_text().strip()

    def chat(self, kind, messages, max_tokens, think=False, tools=None, temperature=0.7):
        """Streaming chat; returns (message dict, record). Never raises; errors are recorded."""
        body = {"model": self.args.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "temperature": temperature, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": think}}
        if tools:
            body["tools"] = tools
        rec = {"ts": time.time(), "kind": kind, "ok": False}
        msg = {"role": "assistant", "content": "", "tool_calls": []}
        t0 = time.time()
        first = None
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    if d.get("usage"):
                        rec["prompt_tokens"], rec["completion_tokens"] = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
                        rec["cached_tokens"] = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                    if not d.get("choices"):
                        continue
                    delta = d["choices"][0].get("delta", {})
                    if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
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
            bad = 0
            for tc in msg["tool_calls"]:
                try:
                    json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    bad += 1
            rec["tool_calls"], rec["bad_tool_args"] = len(msg["tool_calls"]), bad
        except Exception as e:  # noqa: BLE001 - every failure is data for the soak report
            rec.update(error=f"{type(e).__name__}: {e}"[:300], total=time.time() - t0)
        if not msg["tool_calls"]:
            del msg["tool_calls"]
        with LOCK, open(self.args.out / "requests.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return msg, rec


def coding_session(cli, wid, until, think):
    """Agent loop: task -> tool calls answered with synthetic code -> context grows; restart when large."""
    while time.time() < until and not STOP.is_set():
        seed = random.randint(0, 10**9)
        msgs = [{"role": "system", "content": "You are a coding agent. Use the tools to inspect the repository before answering.\n\nRepository overview:\n" + synthetic_code(3000, seed)},
                {"role": "user", "content": random.choice(CODING_TASKS)}]
        ctx = 0
        while ctx < random.randint(50_000, 100_000) and time.time() < until and not STOP.is_set():
            reply, rec = cli.chat(f"coding{'-think' if think else ''}", msgs, 2048 if think else 768, think=think, tools=TOOLS)
            if not rec["ok"]:
                time.sleep(5)
                break
            ctx = rec.get("prompt_tokens", ctx)
            msgs.append(reply)
            for tc in reply.get("tool_calls", []):
                msgs.append({"role": "tool", "tool_call_id": tc["id"] or "call", "content": synthetic_code(random.randint(2000, 5000), random.randint(0, 10**9))})
            if not reply.get("tool_calls"):
                msgs.append({"role": "user", "content": "Good. Now check the related module too:\n" + synthetic_code(random.randint(2000, 4000), random.randint(0, 10**9))})
            time.sleep(random.uniform(2, 10))  # client-side "thinking" / tool execution time


def robot_loop(cli, until, image):
    while time.time() < until and not STOP.is_set():
        if random.random() < 0.2:
            content = [{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": "小智，这张图片里有哪些颜色？"}]
            cli.chat("robot-image", [{"role": "user", "content": content}], 200)
        else:
            cli.chat("robot", [{"role": "user", "content": random.choice(ROBOT_PROMPTS)}], 400)
        STOP.wait(random.uniform(60, 240) * cli.args.minute_seconds / 60)


def full_load(cli, until):
    def worker():
        while time.time() < until and not STOP.is_set():
            cli.chat("fullload", [{"role": "user", "content": f"[{random.random()}] Write a detailed technical design doc for a home robot task scheduler, with Python code."}], 2048)
    ts = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]


def long_prefill(cli):
    pad = synthetic_code(150_000, random.randint(0, 10**9))
    cli.chat("long-prefill", [{"role": "user", "content": pad + "\n\nWhich function above has the largest limit default? Answer briefly."}], 128)


def burst(cli):
    ts = [threading.Thread(target=cli.chat, args=("burst", [{"role": "user", "content": f"[{random.random()}] Explain {topic} in 200 words."}], 400))
          for topic in ["TCP slow start", "B-trees", "Kalman filters", "RAID 5", "OAuth2 PKCE", "CRDTs"]]
    [t.start() for t in ts]
    [t.join() for t in ts]


def probe(cli, out):
    msg, rec = cli.chat("probe", [{"role": "user", "content": PROBE}], 200, temperature=0.0)
    with LOCK, open(out / "probes.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "ok": rec["ok"], "text": msg.get("content", "")}, ensure_ascii=False) + "\n")


def monitor(args):
    fields = ["ts", "temp_c", "fan_pct", "power_w", "pstate", "sm_mhz", "gpu_util", "vram_mib", "throttle",
              "restarts", "health", "ctr_cpu_pct", "ctr_mem", "host_mem_used_mib"]
    path = args.out / "monitor.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        while not STOP.is_set():
            try:
                smi = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,fan.speed,power.draw,pstate,clocks.sm,utilization.gpu,memory.used,clocks_throttle_reasons.active",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().split(", ")
                ins = subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}} {{.State.Health.Status}}", "neuron-sglang"],
                                     capture_output=True, text=True, timeout=20).stdout.split()
                st = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}} {{.MemUsage}}", "neuron-sglang"],
                                    capture_output=True, text=True, timeout=30).stdout.split()
                mem = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1].split()[2]
                w.writerow([int(time.time()), *smi, *(ins or ["?", "?"]), st[0] if st else "?", st[1] if len(st) > 1 else "?", mem])
                f.flush()
            except Exception as e:  # noqa: BLE001
                w.writerow([int(time.time()), f"monitor error {e}"])
            STOP.wait(args.monitor_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hours", type=int, default=10)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--key-file", default="/dd/neuron/configs/api-key")
    ap.add_argument("--minute-seconds", type=float, default=60.0)
    ap.add_argument("--monitor-interval", type=float, default=30.0)
    ap.add_argument("--no-long-prefill", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cli, image, m = Client(args), png_image(), args.minute_seconds
    long_prefill_hours = {0, 2, 5, 7}  # ~every 2.5 h
    full_load_hours = {3, 8}

    threading.Thread(target=monitor, args=(args,), daemon=True).start()
    log_event(args.out, f"soak start: {args.hours} cycles, model {args.model}")
    try:
        for h in range(args.hours):
            start = time.time()
            log_event(args.out, f"cycle {h + 1}/{args.hours}")
            if h in long_prefill_hours and not args.no_long_prefill:
                log_event(args.out, "long prefill (~150K tokens)")
                long_prefill(cli)
            active_until = start + 35 * m
            threads = [threading.Thread(target=robot_loop, args=(cli, active_until, image))]
            if h in full_load_hours:
                log_event(args.out, "full load phase (6 concurrent long generations)")
                threads.append(threading.Thread(target=full_load, args=(cli, start + 30 * m)))
            else:
                threads += [threading.Thread(target=coding_session, args=(cli, i, active_until, i == 0)) for i in range(3)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            log_event(args.out, "burst (6 concurrent)")
            burst(cli)
            probe(cli, args.out)
            log_event(args.out, "idle")
            STOP.wait(max(0, start + 60 * m - time.time()))
    finally:
        STOP.set()
        time.sleep(1)
        log_event(args.out, "soak end")


if __name__ == "__main__":
    main()
