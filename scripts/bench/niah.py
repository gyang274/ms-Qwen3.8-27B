#!/usr/bin/env python3
"""Multi-needle retrieval at long context: does the model still find facts at 128K..1M tokens?

The haystack is varied synthetic prose (logbook entries). Eight needles ("The access code for <place> is
<6 digits>.") sit at evenly spaced depths; the question asks for all eight codes at once, so one request
scores retrieval at eight depths. Also reports TTFT (prefill time) and the server-side prompt length.

usage: niah.py --tokens 131072,262144,524288,1000000 [--out results.jsonl] [--needles 8]
"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

PLACES = ["Harbor Gate", "North Archive", "Glass Tower", "Old Mill", "Cedar Lab", "Signal Hill", "Dry Dock", "Moon Vault",
          "Iron Bridge", "Salt Market", "Fox Garden", "Blue Depot"]
SUBJ = ["The night crew", "A courier", "The duty engineer", "Two inspectors", "The quartermaster", "A visiting auditor"]
VERB = ["logged", "inspected", "repaired", "recalibrated", "inventoried", "photographed", "sealed", "reopened"]
OBJ = ["the east pump", "a cracked valve", "the backup generator", "forty crates of rope", "the tide gauge", "the signal lamp",
       "a stack of ledgers", "the cold storage door", "the north conveyor", "a misrouted parcel"]
TAIL = ["before the shift change.", "despite the rain.", "and filed a short report.", "with no further issues.",
        "after a brief delay.", "while the harbor was quiet.", "and noted the readings in the margin."]
TOKENS_PER_SENTENCE = 21.2  # measured for this generator with the Qwen tokenizer (refined below from usage)


def haystack(n_sentences, rnd):
    return [f"Entry {i}: {rnd.choice(SUBJ)} {rnd.choice(VERB)} {rnd.choice(OBJ)} {rnd.choice(TAIL)}" for i in range(n_sentences)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="131072")
    ap.add_argument("--needles", type=int, default=8)
    ap.add_argument("--url", default="http://localhost:30000/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--key-file", default="/dd/neuron/configs/api-key")
    ap.add_argument("--out", default="niah.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    key = Path(args.key_file).read_text().strip()
    tps = TOKENS_PER_SENTENCE
    for target in [int(x) for x in args.tokens.split(",")]:
        rnd = random.Random(args.seed + target)
        lines = haystack(int((target - 600) / tps), rnd)
        places = rnd.sample(PLACES, args.needles)
        codes = {p: f"{rnd.randint(100000, 999999)}" for p in places}
        depths = [(i + 0.5) / args.needles for i in range(args.needles)]
        for p, d in zip(places, depths):
            lines.insert(int(d * len(lines)), f"Note: the access code for {p} is {codes[p]}.")
        question = ("Above is a long logbook. Somewhere in it are notes giving access codes for several places. "
                    f"List the access code for each of these places: {', '.join(sorted(places))}. "
                    "Answer with one line per place in the form 'Place: code'.")
        body = {"model": args.model, "messages": [{"role": "user", "content": "\n".join(lines) + "\n\n" + question}],
                "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time()
        rec = {"target_tokens": target, "needles": args.needles}
        try:
            req = urllib.request.Request(args.url, data=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=7200) as r:
                d = json.load(r)
            text = d["choices"][0]["message"]["content"] or ""
            found = {p: bool(re.search(re.escape(p) + r"\W+" + codes[p], text)) for p in places}
            rec.update(ok=True, prompt_tokens=d["usage"]["prompt_tokens"], seconds=time.time() - t0,
                       hits=sum(found.values()), by_depth=[found[p] for p in places], answer=text[:600])
            tps = (d["usage"]["prompt_tokens"] - 600) / len(lines)  # refine the estimate for the next size
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:300], seconds=time.time() - t0)
        print(json.dumps({k: v for k, v in rec.items() if k != "answer"}), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
