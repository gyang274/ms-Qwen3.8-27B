# Summaries

## One line (project card)

Serving Qwen3.8-27B on one RTX A6000: FP8 on Ampere, a byte-verified abliterated FP8 quant, and a 10-hour soak test.

## Short post (LinkedIn)

Most Qwen3.8 serving guides assume Hopper or Blackwell. I wanted to know how far a five-year-old 48 GB RTX A6000
could go as an always-on home LLM server for a home robot and my coding agents.

What it took:
• FP8 without FP8 tensor cores. SGLang serves Qwen3.8-27B-FP8 as Marlin W8A16 with MTP speculative decoding:
  ~60 tok/s per stream and ~150 tok/s across four, with the full native 262K context.
• Thermals. A 250 W cap and a headless NVML fan curve took the card from 87 °C with thermal slowdown to 74 °C,
  with zero decode loss.
• A verifiable quant. I quantized the abliterated variant to FP8 with a recipe that reproduces Qwen's official
  FP8 checkpoint byte for byte. That isolates the abliteration exactly: 70 residual-stream output projections in
  layers 17-51. An SVD against the original weights shows that the entire edit is one vector and one number,
  ΔW = −1.3 · r rᵀW, with the same refusal direction r in every layer. It costs 2.5-3 points on HumanEval and
  GSM8K.
• Proof it holds. A 10-hour mixed soak served 2,617 requests and 1,324 tool calls with zero errors. It also
  caught a subtle bug: the default health check kept waking the idle GPU, which pinned it at 73 W instead of 7 W.

Notes, scripts and raw data: github.com/gyang274/ms-Qwen3.8-27B
Model: huggingface.co/gyang274/Huihui-Qwen3.8-27B-abliterated-FP8
