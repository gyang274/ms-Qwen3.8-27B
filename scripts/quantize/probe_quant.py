#!/usr/bin/env python3
"""Try several block-FP8 recipes on one unmodified tensor and report which reproduces the official bytes."""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

SRC, TPL = Path(sys.argv[1]), Path(sys.argv[2])
KEYS = ["model.language_model.layers.0.mlp.down_proj.weight", "model.language_model.layers.3.self_attn.q_proj.weight"]
M = 448.0


def find(d, k):
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(p, "pt") as f:
            if k in f.keys():
                return f.get_tensor(k)
    raise KeyError(k)


def blocks(w):
    o, i = w.shape
    bo, bi = -(-o // 128), -(-i // 128)
    wp = torch.zeros(bo * 128, bi * 128, dtype=torch.float32)
    wp[:o, :i] = w.float()
    return wp.view(bo, 128, bi, 128), (o, i)


def finish(q, shape):
    bo, _, bi, _ = q.shape
    return q.reshape(bo * 128, bi * 128)[: shape[0], : shape[1]]


def recipes(w):
    b, shp = blocks(w)
    amax = b.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    out = {}
    # A: scale rounded to bf16 first, divide
    s = (amax / M).bfloat16().float()
    out["A bf16-scale divide"] = (finish((b / s[:, None, :, None]).clamp(-M, M).to(torch.float8_e4m3fn), shp), s.bfloat16())
    # B: fp32 scale for quantization, stored as bf16
    s = amax / M
    out["B fp32-scale divide"] = (finish((b / s[:, None, :, None]).clamp(-M, M).to(torch.float8_e4m3fn), shp), s.bfloat16())
    # C: fp32 multiply by reciprocal
    inv = M / amax
    out["C fp32 mul-recip"] = (finish((b * inv[:, None, :, None]).clamp(-M, M).to(torch.float8_e4m3fn), shp), (1 / inv).bfloat16())
    # D: bf16 scale, multiply by reciprocal of bf16 scale
    s = (amax / M).bfloat16().float()
    out["D bf16-scale mul-recip"] = (finish((b * (1 / s)[:, None, :, None]).clamp(-M, M).to(torch.float8_e4m3fn), shp), s.bfloat16())
    # E: computations in bf16 (weight bf16 / bf16 scale)
    bb = b.bfloat16()
    s = (bb.abs().amax(dim=(1, 3)).float() / M).bfloat16()
    out["E all-bf16 divide"] = (finish((bb / s[:, None, :, None]).float().clamp(-M, M).to(torch.float8_e4m3fn), shp), s)
    # F: scale rounded up in bf16 (no clipping)
    s = (amax / M).bfloat16()
    s = torch.where(s.float() < amax / M, torch.nextafter(s.float(), torch.tensor(float("inf"))).bfloat16(), s).float()
    out["F bf16-scale ceil"] = (finish((b / s[:, None, :, None]).clamp(-M, M).to(torch.float8_e4m3fn), shp), s.bfloat16())
    return out


for k in KEYS:
    w = find(SRC, k)
    q_ref, s_ref = find(TPL, k), find(TPL, k + "_scale_inv")
    print(k)
    for name, (q, s) in recipes(w).items():
        qm = (q.view(torch.uint8) == q_ref.view(torch.uint8)).float().mean().item()
        sm = (s == s_ref).float().mean().item()
        print(f"  {name:24s} fp8 bytes match {qm:.4%}  scale match {sm:.4%}")
    s_ratio = (s_ref.float() / ((blocks(w)[0].abs().amax(dim=(1, 3))) / M)).flatten()
    print(f"  official scale / (amax/448): min {s_ratio.min():.5f} median {s_ratio.median():.5f} max {s_ratio.max():.5f}")
