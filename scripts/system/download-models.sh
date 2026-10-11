#!/usr/bin/env bash
# Download model weights into /dd/neuron/models (resumable; token from ~/.cache/huggingface/token)
set -euo pipefail
cd /dd/neuron
HF="$HOME/.local/bin/uvx --from huggingface_hub[cli] hf"
$HF download Qwen/Qwen3.8-27B-FP8 --local-dir models/Qwen3.8-27B-FP8 >> logs/download-qwen3.8-27b-fp8.log 2>&1 &
$HF download huihui-ai/Huihui-Qwen3.8-27B-abliterated --revision 739e3c5b89849f6c238ce1e5b70008612ae42cdd \
  --local-dir models/Huihui-Qwen3.8-27B-abliterated-BF16 >> logs/download-huihui-bf16.log 2>&1 &
wait
