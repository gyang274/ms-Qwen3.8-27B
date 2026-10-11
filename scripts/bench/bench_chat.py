#!/usr/bin/env python3
"""Measure TTFT and decode speed of an OpenAI-compatible endpoint with streaming.

usage: bench_chat.py [--url http://localhost:30000/v1] [--model qwen3.8-27b] [--concurrency 1] [--max-tokens 1024]
       [--think] [--prompt-tokens N]   (pads the prompt with filler code to roughly N tokens)
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Write a Python implementation of an LRU cache with O(1) get/put, plus unit tests using pytest.",
    "Explain how a Kalman filter works for a home robot fusing wheel odometry and IMU, with equations and a short Python example.",
    "Write a bash script that rotates log files in a directory, keeping the last 7 days, compressing older ones.",
    "Refactor this into idiomatic TypeScript with types: function f(a,b){var r=[];for(var i=0;i<a.length;i++){if(b(a[i]))r.push(a[i])}return r}",
]
FILLER = "def helper_{i}(x):\n    return [v * {i} for v in x if v % 3 != {m}]\n\n"


def run(args, i):
    prompt = PROMPTS[i % len(PROMPTS)]
    if args.prompt_tokens:
        pad = "".join(FILLER.format(i=k, m=k % 3) for k in range(args.prompt_tokens // 25))
        prompt = f"Here is some code:\n{pad}\nIgnore the code above. {prompt}"
    if args.nonce:  # unique prefix so the radix cache cannot serve the prompt
        prompt = f"[run {random.random():.12f}]\n{prompt}"
    body = {
        "model": args.model, "stream": True, "max_tokens": args.max_tokens, "temperature": 0.7, "top_p": 0.8,
        "messages": [{"role": "user", "content": prompt}],
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    req = urllib.request.Request(args.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"})
    t0 = time.time()
    first, usage = None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if first is None and d.get("choices") and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning_content")):
                first = time.time()
    t1 = time.time()
    out = usage["completion_tokens"]
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": out, "ttft": first - t0, "decode_tps": (out - 1) / (t1 - first)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--nonce", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(lambda i: run(args, i), range(args.concurrency)))
    wall = time.time() - t0
    for r in res:
        print(f"  prompt {r['prompt_tokens']:6d}  out {r['completion_tokens']:5d}  ttft {r['ttft']:6.2f}s  decode {r['decode_tps']:6.1f} tok/s")
    total = sum(r["completion_tokens"] for r in res)
    print(f"C={args.concurrency} think={args.think} per-stream avg {sum(r['decode_tps'] for r in res) / len(res):.1f} tok/s, aggregate {total / wall:.1f} tok/s")


if __name__ == "__main__":
    main()
