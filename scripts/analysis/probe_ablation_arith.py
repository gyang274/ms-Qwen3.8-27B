#!/usr/bin/env python3
"""Probe which arithmetic reproduces the abliterated BF16 bits for one tensor (direction refined by least squares)."""
import glob
import sys

import torch
from safetensors import safe_open

orig_dir, abl_dir, dirs_path = sys.argv[1:4]
k = "model.language_model.layers.17.mlp.down_proj.weight"


def get(d):
    for p in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(p, "pt") as f:
            if k in f.keys():
                return f.get_tensor(k)


W, A = get(orig_dir), get(abl_dir)
D = torch.load(dirs_path)
r = (D * torch.sign(D @ D[0])[:, None]).mean(0)
r = r / r.norm()
dw = A.float() - W.float()
# refine r for this tensor: dW ~= -a r (r^T W)  =>  r ∝ dW (W^T r) (power-iteration style)
for _ in range(5):
    v = r @ W.float()
    r_new = -(dw @ v)
    r = r_new / r_new.norm()


def score(R):
    return (R.view(torch.int16) == A.view(torch.int16)).float().mean().item()


cands = {}
Wf = W.float()
cands["fp32: W - 1.3 r (r^T W)"] = (Wf - 1.3 * torch.outer(r, r @ Wf)).bfloat16()
cands["fp32: W - r (1.3 r^T W) (proj then scale)"] = (Wf - torch.outer(r, 1.3 * (r @ Wf))).bfloat16()
rb = r.bfloat16()
cands["bf16 r, fp32 math"] = (Wf - 1.3 * torch.outer(rb.float(), rb.float() @ Wf)).bfloat16()
cands["all bf16 math"] = (W - 1.3 * torch.outer(rb, rb @ W)).bfloat16()
P = torch.outer(r, r)
cands["fp32 projector: (I - 1.3 r r^T) W"] = ((torch.eye(r.numel()) - 1.3 * P) @ Wf).bfloat16()
cands["bf16 projector: (I - 1.3 r r^T) W"] = ((torch.eye(r.numel()) - 1.3 * P).bfloat16() @ W).bfloat16()
cands["fp64: W - 1.3 r (r^T W)"] = (W.double() - 1.3 * torch.outer(r.double(), r.double() @ W.double())).bfloat16()
for name, R in cands.items():
    print(f"{score(R):.4%}  {name}")
