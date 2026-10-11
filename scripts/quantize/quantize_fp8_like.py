#!/usr/bin/env python3
"""Quantize a BF16 checkpoint to block-wise FP8 (E4M3), using an official FP8 checkpoint as the layout template.

Every tensor that is F8_E4M3 in the template (with a sibling `weight_scale_inv`) is quantized the same way;
every other tensor is copied with the template dtype. Output names/dtypes/shapes match the template exactly.

usage: quantize_fp8_like.py --src BF16_DIR --template FP8_DIR --dst OUT_DIR
"""
import argparse
import json
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def load_layout(model_dir):
    layout = {}
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        for k, v in read_header(p).items():
            layout[k] = (v["dtype"], v["shape"], p)
    return layout


def quantize_block(w, block, scale_dtype):
    """w [out, in] -> (fp8 weight, scale_inv [ceil(out/b), ceil(in/b)]), w ~= fp8 * scale_inv per block."""
    out_f, in_f = w.shape
    bo, bi = (out_f + block - 1) // block, (in_f + block - 1) // block
    wp = torch.zeros(bo * block, bi * block, dtype=torch.float32)
    wp[:out_f, :in_f] = w.float()
    blocks = wp.view(bo, block, bi, block)
    amax = blocks.abs().amax(dim=(1, 3))
    # fp32 scale = amax * (1/448), quantize with the fp32 scale, store it rounded to scale_dtype:
    # this reproduces the official Qwen FP8 checkpoint byte for byte on unmodified tensors
    scale = amax.clamp(min=1e-12) * (1.0 / FP8_MAX)
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.view(bo * block, bi * block)[:out_f, :in_f].contiguous()
    return q, scale.to(scale_dtype)


def dequantize_block(q, scale, block):
    s = scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    return q.float() * s[: q.shape[0], : q.shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, tpl, dst = Path(args.src), Path(args.template), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    tpl_layout = load_layout(tpl)
    src_layout = load_layout(src)
    cfg = json.load(open(tpl / "config.json"))
    qcfg = cfg.get("quantization_config") or cfg["text_config"]["quantization_config"]
    block = qcfg["weight_block_size"][0]
    assert qcfg["weight_block_size"] == [block, block] and qcfg["fmt"] == "e4m3"

    expected = {k for k in tpl_layout if not k.endswith("weight_scale_inv")}
    missing, extra = expected - set(src_layout), set(src_layout) - expected
    assert not missing and not extra, f"name mismatch: missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}"

    weight_map, max_err = {}, 0.0
    for shard in sorted({p for _, _, p in src_layout.values()}):
        out = {}
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                tdtype, tshape, _ = tpl_layout[k]
                assert list(t.shape) == tshape, f"{k}: shape {list(t.shape)} != template {tshape}"
                if tdtype == "F8_E4M3":
                    sk = k + "_scale_inv"
                    sdtype, sshape, _ = tpl_layout[sk]
                    q, s = quantize_block(t, block, DTYPES[sdtype])
                    assert list(s.shape) == sshape, f"{sk}: {list(s.shape)} != {sshape}"
                    rel = ((dequantize_block(q, s, block) - t.float()).norm() / t.float().norm()).item()
                    max_err = max(max_err, rel)
                    out[k], out[sk] = q, s
                else:
                    out[k] = t.to(DTYPES[tdtype])
        save_file(out, dst / shard.name, metadata={"format": "pt"})
        weight_map.update({k: shard.name for k in out})
        print(f"{shard.name}: {len(out)} tensors, max rel err so far {max_err:.4%}", flush=True)

    assert set(weight_map) == set(tpl_layout), "output tensor set differs from template"
    total = sum((dst / n).stat().st_size for n in set(weight_map.values()))
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    # config (carries quantization_config) from the template; tokenizer, chat template etc. from the source
    shutil.copy(tpl / "config.json", dst / "config.json")
    skip = {"config.json", "model.safetensors.index.json", "crc32.txt"}
    for p in src.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and p.name not in skip:
            shutil.copy(p, dst / p.name)
    print(f"done: {len(weight_map)} tensors, {total / 2**30:.2f} GiB, max relative weight error {max_err:.4%}")


if __name__ == "__main__":
    main()
