#!/usr/bin/env python3
"""Characterize an abliteration edit: for every tensor that differs between the original and the abliterated
BF16 checkpoints, measure the structure of dW = W_abl - W_orig.

Directional ablation (Arditi et al., 2024) predicts, for a matrix W that writes into the residual stream
(output dim = hidden size), dW = -alpha * r r^T W for a unit "refusal direction" r. So dW should be
(1) rank 1, (2) have its left singular vector u1 = +-r in residual space, and (3) satisfy dW ~= -alpha u1 u1^T W.

Writes a JSON with per-tensor stats and the pairwise |cos| between the u1 directions.

usage: abliteration_svd.py --orig BF16_DIR --abl BF16_DIR --out results/abliteration/svd.json [--device cuda]
"""
import argparse
import json
import re
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


def get(m, k, device):
    with safe_open(m[k], "pt", device="cpu") as f:
        return f.get_tensor(k).to(device=device, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rank", type=int, default=8, help="singular values to report")
    args = ap.parse_args()
    a, o = index(args.abl), index(args.orig)
    assert set(a) == set(o), "tensor sets differ"

    changed, unchanged = [], 0
    for k in sorted(a):
        with safe_open(a[k], "pt") as fa, safe_open(o[k], "pt") as fo:
            if torch.equal(fa.get_tensor(k), fo.get_tensor(k)):
                unchanged += 1
            else:
                changed.append(k)
    print(f"{len(changed)} tensors changed, {unchanged} bit-identical", flush=True)

    rows, dirs = [], []
    for k in changed:
        w = get(o, k, args.device)
        dw = get(a, k, args.device) - w
        u, s, v = torch.svd_lowrank(dw, q=args.rank + 4, niter=4)
        fro2 = dw.pow(2).sum()
        u1 = u[:, 0]
        # best alpha for dW ~= -alpha u1 u1^T W, and how much of dW that explains
        proj = torch.outer(u1, u1 @ w)
        alpha = -(dw * proj).sum() / proj.pow(2).sum()
        resid = (dw + alpha * proj).norm() / dw.norm()
        layer = int(re.search(r"layers\.(\d+)\.", k).group(1))
        kind = k.split(".")[-3] + "." + k.split(".")[-2]
        rows.append({
            "tensor": k, "layer": layer, "kind": kind, "shape": list(w.shape),
            "rel_change": (dw.norm() / w.norm()).item(),
            "singular_values": [x.item() for x in s[: args.rank]],
            "rank1_energy": (s[0] ** 2 / fro2).item(),
            "alpha": alpha.item(),
            "rank1_projection_residual": resid.item(),
        })
        dirs.append(u1.cpu())
        print(f"{k}: rel {rows[-1]['rel_change']:.4%} rank1 energy {rows[-1]['rank1_energy']:.6f} "
              f"alpha {alpha.item():.4f} resid {resid.item():.4f}", flush=True)
        del w, dw, u, s, v, proj

    d = torch.stack(dirs)
    cos = (d @ d.T).abs()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"changed": len(changed), "unchanged": unchanged, "tensors": rows,
               "abs_cos": [[round(x, 6) for x in r] for r in cos.tolist()]}, open(args.out, "w"), indent=1)
    off = cos[~torch.eye(len(d), dtype=torch.bool)]
    print(f"pairwise |cos| between edit directions: min {off.min():.4f} median {off.median():.4f} max {off.max():.4f}")
    torch.save(d, Path(args.out).with_suffix(".directions.pt"))


if __name__ == "__main__":
    main()
