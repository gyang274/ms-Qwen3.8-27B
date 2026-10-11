#!/usr/bin/env python3
"""Rebuild abliterated weights from the original ones as W - alpha * r r^T W, with a single shared direction r
(the sign-aligned mean of the per-tensor edit directions saved by abliteration_svd.py), and measure how close
the rebuild gets to the published abliterated tensors.

usage: reconstruct_check.py --orig BF16_DIR --abl BF16_DIR --directions svd.directions.pt --svd svd.json
                            [--alphas 1.0 1.3] [--out results/abliteration/reconstruct.json]
"""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--svd", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 1.3])
    ap.add_argument("--out")
    args = ap.parse_args()
    o, a = index(args.orig), index(args.abl)
    tensors = [t["tensor"] for t in json.load(open(args.svd))["tensors"]]
    d = torch.load(args.directions)
    ref = d[0] / d[0].norm()
    r = (d * torch.sign(d @ ref)[:, None]).mean(0)
    r = r / r.norm()

    rows = []
    for k in tensors:
        with safe_open(o[k], "pt") as f:
            w = f.get_tensor(k).float()
        with safe_open(a[k], "pt") as f:
            actual = f.get_tensor(k)
        dw = actual.float() - w
        row = {"tensor": k}
        for alpha in args.alphas:
            rebuilt = (w - alpha * torch.outer(r, r @ w)).bfloat16()
            row[f"alpha_{alpha:g}"] = {
                "bit_exact": (rebuilt.view(torch.int16) == actual.view(torch.int16)).float().mean().item(),
                "rel_error_vs_dW": ((rebuilt.float() - actual.float()).norm() / dw.norm()).item(),
            }
        rows.append(row)
        print(k.split("layers.")[1], {k2: {m: round(v, 5) for m, v in v2.items()} for k2, v2 in row.items() if k2 != "tensor"}, flush=True)

    summary = {}
    for alpha in args.alphas:
        key = f"alpha_{alpha:g}"
        be = [r_[key]["bit_exact"] for r_ in rows]
        er = [r_[key]["rel_error_vs_dW"] for r_ in rows]
        summary[key] = {"bit_exact_min": min(be), "bit_exact_mean": sum(be) / len(be), "rel_error_max": max(er)}
    print(json.dumps(summary, indent=1))
    if args.out:
        json.dump({"summary": summary, "tensors": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
