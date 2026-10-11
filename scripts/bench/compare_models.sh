#!/usr/bin/env bash
# Run the same speed + quality evaluation on the official and abliterated models (sequentially, one GPU).
# usage: compare_models.sh [official abliterated]
set -euo pipefail
cd "$(dirname "$0")"
KEY=$(cat /dd/neuron/configs/api-key)
RUN=results/compare-$(date +%Y%m%d-%H%M)
mkdir -p "$RUN"
[ $# -gt 0 ] && VARIANTS=("$@") || VARIANTS=(official abliterated)

for v in "${VARIANTS[@]}"; do
  [ "$v" = official ] && name=qwen3.8-27b || name=qwen3.8-27b-abliterated
  out="$RUN/$v"; mkdir -p "$out"
  echo "=== $v ($name) ==="
  ../serve/switch-model.sh "$v"
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 -I bench_chat.py --model "$name" --api-key "$KEY" --max-tokens 128 >/dev/null
  { python3 -I bench_chat.py --model "$name" --api-key "$KEY" --nonce --max-tokens 1024
    python3 -I bench_chat.py --model "$name" --api-key "$KEY" --nonce --max-tokens 1024 --concurrency 4; } | tee "$out/speed.txt"
  /dd/neuron/.venv-tools/bin/python -I eval_compare.py --model "$name" --out "$out"
  docker run --rm --network none --memory 4g --pids-limit 256 -v "$PWD/$out":/w:ro -v "$PWD":/s:ro \
    --entrypoint python3 lmsysorg/sglang:v0.5.21-cu130 /s/grade_humaneval.py /w/humaneval.jsonl | tee "$out/humaneval_grade.json"
  docker logs --since "$since" neuron-sglang 2>&1 | grep -o "accept len: [0-9.]*" | awk '{s+=$3; n++} END {printf "mean MTP accept len %.2f over %d decode logs\n", s/n, n}' | tee "$out/mtp.txt"
done
echo "results in bench/$RUN"
