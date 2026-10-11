# ms-Qwen3.8-27B

**Serving Qwen3.8-27B on RTX A6000s: FP8 on Ampere, a byte-verified abliterated FP8 quant, a 10-hour soak test, and coding-agent capacity from one to two GPUs and 1M context.**

Most Qwen3.8 serving recipes assume Hopper or Blackwell. This repository records what it took to run
Qwen3.8-27B as an always-on home server (one user, many sessions, backing a home robot and coding agents)
on a 48 GB Ampere card with no FP8 tensor cores. SGLang serves the FP8 checkpoint as Marlin W8A16 with
MTP speculative decoding. The native 262K context fits once the ~8 GB that the default memory budget leaves
unused is reclaimed. A 250 W cap and an NVML fan curve take the card from 87 °C with thermal slowdown to 74 °C
without losing decode speed. Along the way, the abliterated variant was quantized to FP8 with a recipe that
reproduces Qwen's official FP8 checkpoint **byte for byte** on every untouched tensor. That byte-identity
check isolates the abliteration edit exactly: 70 residual-stream output projections in layers 17-51, and
nothing else. An SVD against the official BF16 weights shows that the whole edit is one vector and one scalar:
every changed matrix is ΔW = −1.3 · r rᵀW with the same unit direction r. A mixed 10-hour soak then served
2,617 requests and 1,324 tool calls with zero errors. It also caught a health check that silently kept the idle
GPU at 73 W instead of 7 W. Finally, a second A6000 was added to measure what it buys coding agents. Agent
workloads are prefill-bound, so capacity follows the KV cache: one card handles about four busy agents, data
parallelism with a tuned router is best up to eight, pipeline parallelism beyond that, and tensor parallelism
decodes fastest and holds a full 1M-token context. Along the way: a peer-to-peer path that CUDA reports as
working but the AMD IOMMU silently drops, and a router default that sends every agent to one replica.

| | |
|---|---|
| Decode | ~60 tok/s single stream, ~150-160 tok/s at 4 streams, 47 tok/s at 134K context |
| Context | native 262,144 tokens; 247K-token shared FP8 KV pool; prefix-cache hits answer in 0.14 s |
| Thermals | 74 °C / 75% fan at a 250 W cap (stock: 87 °C, thermal slowdown); idle ~7 W in P8 |
| FP8 quant | byte-identical to the official Qwen FP8 checkpoint on all unmodified tensors |
| Abliteration | 70 of 1,606 tensors differ, every one by ΔW = −1.3 · r rᵀW with a single shared r; costs 2.5 pts HumanEval and 3 pts GSM8K |
| Soak | 10 h, 2,617 requests, 0 errors, no restarts, no drift, no leaks, plus one idle-power bug found and fixed |
| Two GPUs | Agents per card before the cache cliff: ~4. Best at 8 agents: DP=2 with a tuned router (2.8 turns/agent/min); at 16: PP=2. TP=2: 81 tok/s single stream |
| 1M context | TP=2 + YaRN with 32 mamba slots: 1.33M-token KV pool, 8/8 needles found in a 992K-token prompt |

![Where the abliteration lives](figures/abliteration-map.png)

![One direction, one scale](figures/abliteration-svd.png)

![Agent capacity on one and two GPUs](figures/agent-capacity.png)

## Read

- **[Deployment notes](docs/deployment-notes.md)**: the full write-up. It covers `sm_86` support, the memory
  budget, power/clock/temperature/fan balance, the FP8 recipe, the anatomy of the abliteration, the
  official-vs-abliterated evaluation, the soak test's design and results, and the two-GPU study for coding
  agents (§12).
- **Model:** [gyang274/Huihui-Qwen3.8-27B-abliterated-FP8](https://huggingface.co/gyang274/Huihui-Qwen3.8-27B-abliterated-FP8)
  on Hugging Face. Compare it with the official [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).

## Repository

| Path | What |
|---|---|
| [docs/](docs/) | Deployment notes, Hugging Face model card |
| [serve/](serve/) | Docker Compose deployment (SGLang v0.5.21), model switch script, 1- and 2-GPU benchmark variants |
| [system/](system/) | systemd units (power cap, fan curve, persistence), unattended-upgrades blacklist |
| [scripts/quantize/](scripts/quantize/) | Template-driven FP8 quantizer, checkpoint comparer, recipe probe |
| [scripts/analysis/](scripts/analysis/) | Abliteration rank and direction analysis |
| [scripts/bench/](scripts/bench/) | Speed bench, model comparison, power sweep, soak test, link test, coding-agent benchmark, long-context retrieval |
| [scripts/system/](scripts/system/) | Headless NVML fan-curve daemon with fan health check |
| [scripts/figures/](scripts/figures/) | Renders every figure from `results/` |
| [results/](results/) | Raw data behind every number |

## Credits

[Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B and its FP8 release, [huihui-ai](https://huggingface.co/huihui-ai)
for the abliterated model, [SGLang](https://github.com/sgl-project/sglang) for the serving engine, and
Arditi et al. (2024) for the directional-ablation analysis this work builds on.
