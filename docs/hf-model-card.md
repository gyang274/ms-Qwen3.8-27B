---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- huihui-ai/Huihui-Qwen3.8-27B-abliterated
base_model_relation: quantized
tags:
- fp8
- qwen3.8
- abliterated
- uncensored
- mtp
- sglang
---

# Huihui-Qwen3.8-27B-abliterated-FP8

**An FP8 quantization of [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)
that reproduces Qwen's official FP8 recipe byte for byte on every tensor the abliteration did not touch.**

The weights are block-wise FP8 E4M3 with 128×128 scales, in the same layout as
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8). The MTP head and vision tower are
preserved, and the checkpoint is 28.75 GiB. It loads through the same code path as the official FP8
checkpoint. This model was built and validated as part of a single-GPU deployment study. The full notes,
scripts and raw data are at **[github.com/gyang274/ms-Qwen3.8-27B](https://github.com/gyang274/ms-Qwen3.8-27B)**.

## What makes this quant verifiable

The quantizer ([quantize_fp8_like.py](https://github.com/gyang274/ms-Qwen3.8-27B/blob/main/scripts/quantize/quantize_fp8_like.py))
uses the official FP8 checkpoint as a layout template. It quantizes exactly the tensors that are FP8 there,
copies the rest in the template's dtype, and asserts that names, dtypes and shapes match. Its scale arithmetic
matches Qwen's: an FP32 block scale `amax × (1/448)` is used for quantization and then stored as BF16.

As a result, comparing this checkpoint with the official one isolates the abliteration edit exactly:

| Tensors | vs. official Qwen3.8-27B-FP8 |
|---|---|
| Layers 0-16 and 52-63, MTP head, vision tower (333), embedding, LM head | **byte-identical** |
| `mlp.down_proj` (35), `linear_attn.out_proj` (26), `self_attn.o_proj` (9) in layers 17-51 | differ: these are the abliteration edit |

Only 70 of 1,606 tensors differ, and all of them write into the residual stream. That is the footprint
directional ablation predicts ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). An SVD against the
official BF16 weights gives the edit's exact form:

- Every changed matrix is rank-1: σ2/σ1 ≤ 0.012.
- All 70 share one unit direction *r*: pairwise |cos| ≥ 0.99999.
- All 70 use one scale: **ΔW = −1.3 · r rᵀW**.

Rebuilding the abliterated weights from the official ones with that single *r* reproduces 91.9% of the BF16
elements bit for bit. See [§8-9 of the notes](https://github.com/gyang274/ms-Qwen3.8-27B/blob/main/docs/deployment-notes.md)
for the recipe probe and the full analysis.

## Evaluation

Both models were served with SGLang v0.5.21 using an identical configuration, thinking off and temperature 0,
on an RTX A6000.

| | Official Qwen3.8-27B-FP8 | This model |
|---|---|---|
| HumanEval pass@1 (164, sandboxed) | 98.2% | 95.7% |
| GSM8K, first 200 | 97.5% | 94.5% |
| Over-refusal: 15 benign prompts that safety-tuned models often refuse | 2 partial | 0 |
| Decode, single stream (RTX A6000) | ~60 tok/s | ~62 tok/s |
| MTP mean accepted length | 3.52 | 3.52 |

Removing refusals has a measurable cost. On GSM8K this model lost 6 problems the official model solved and
gained none. Most of the losses came from indecision on ambiguously worded problems rather than arithmetic
errors.

## Usage

SGLang (tested with v0.5.21). On Ampere (`sm_86`, `sm_80`) FP8 runs automatically as Marlin W8A16. On Ada,
Hopper and Blackwell it uses native FP8.

```bash
python3 -m sglang.launch_server \
  --model-path gyang274/Huihui-Qwen3.8-27B-abliterated-FP8 \
  --context-length 262144 --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

The deployment notes give the memory flags that fit the full 262K context with 4 concurrent requests on a 48 GB
card. The checkpoint uses the same format as the official FP8 release, so other engines that load
`Qwen/Qwen3.8-27B-FP8` (for example vLLM) should load it the same way. Only SGLang was tested here.

## Provenance

| | |
|---|---|
| Source | [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) @ `739e3c5b89849f6c238ce1e5b70008612ae42cdd` |
| Template | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (`config.json` and quantization layout) |
| Quantization | FP8 E4M3, 128×128 blocks, dynamic activation scheme, max relative weight error 2.65% |
| Original model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0 |

## Risks

This model has had its refusal behavior removed and will comply with requests that the original model
declines. It is intended for research and for personal use in controlled settings. It is not suitable for
public-facing or child-facing deployments without your own safeguards. You are responsible for how you use
it and for the outputs it produces.

## Credits

Qwen for Qwen3.8-27B and its FP8 recipe, and huihui-ai for the abliterated model. Licensed Apache-2.0, as are
both upstream models.
