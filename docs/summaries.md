# Summaries

## One line (project card)

Serving Qwen3.8-27B on RTX A6000s: FP8 on Ampere, a byte-verified abliterated FP8 quant, a 10-hour soak test, and coding-agent capacity from one to two GPUs and 1M context.

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
• A second card, measured the way coding agents use it. Agent turns read far more than they write, so capacity
  follows the KV cache, not decode speed: one card serves about four busy agents before its 250K-token cache
  overflows. On two cards, data parallelism with a tuned router is best up to eight agents, pipeline parallelism
  (974K-token cache) beyond that, and tensor parallelism decodes fastest at 81 tok/s and holds a full
  1M-token context (8/8 needles found at 992K tokens). Two traps on a consumer board: CUDA reports peer-to-peer
  access that the AMD IOMMU silently drops, and sglang-router's default settings send every agent to one replica.

Notes, scripts and raw data: github.com/gyang274/ms-Qwen3.8-27B
Model: huggingface.co/gyang274/Huihui-Qwen3.8-27B-abliterated-FP8
