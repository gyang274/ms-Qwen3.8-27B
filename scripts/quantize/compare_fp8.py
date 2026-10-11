#!/usr/bin/env python3
"""Compare two FP8 checkpoints tensor by tensor: exact-match ratio and dequantized relative difference.

usage: compare_fp8.py A_DIR B_DIR
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def index(d):
    m = {}
    for p in sorted(Path(d).glob("*.safetensors")):
        with safe_open(p, "pt") as f:
            for k in f.keys():
                m[k] = p
    return m


def get(m, k):
    with safe_open(m[k], "pt") as f:
        return f.get_tensor(k)


def group(k):
    r = re.search(r"layers\.(\d+)\.", k)
    if k.startswith("mtp"):
        return "mtp"
    if "visual" in k:
        return "visual"
    if r and "language_model" in k:
        return f"layer{int(r.group(1)):02d}"
    return "other"


def main():
    a, b = index(sys.argv[1]), index(sys.argv[2])
    assert set(a) == set(b), "tensor sets differ"
    stats = defaultdict(lambda: [0, 0, 0.0])  # identical tensors, total tensors, max rel diff
    for k in sorted(a):
        if k.endswith("weight_scale_inv"):
            continue
        ta, tb = get(a, k), get(b, k)
        g = stats[group(k)]
        g[1] += 1
        if ta.dtype == torch.float8_e4m3fn:
            same = torch.equal(ta.view(torch.uint8), tb.view(torch.uint8)) and torch.equal(get(a, k + "_scale_inv"), get(b, k + "_scale_inv"))
            fa = ta.float() * get(a, k + "_scale_inv").float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: ta.shape[0], : ta.shape[1]]
            fb = tb.float() * get(b, k + "_scale_inv").float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: tb.shape[0], : tb.shape[1]]
        else:
            same = torch.equal(ta, tb)
            fa, fb = ta.float(), tb.float()
        g[0] += int(same)
        g[2] = max(g[2], ((fa - fb).norm() / fb.norm().clamp(min=1e-12)).item())
    for name in sorted(stats):
        same, total, diff = stats[name]
        print(f"{name:8s} identical {same:3d}/{total:<3d} max rel diff {diff:.4%}")


if __name__ == "__main__":
    main()
