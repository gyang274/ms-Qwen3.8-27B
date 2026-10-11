#!/usr/bin/env bash
# usage: switch-model.sh official|abliterated   — rewrites .env and recreates the container
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  official)    m=Qwen3.8-27B-FP8;                    n=qwen3.8-27b ;;
  abliterated) m=Huihui-Qwen3.8-27B-abliterated-FP8; n=qwen3.8-27b-abliterated ;;
  *) echo "usage: $0 official|abliterated" >&2; exit 1 ;;
esac
printf "NEURON_MODEL=%s\nNEURON_SERVED_NAME=%s\n" "$m" "$n" > .env
docker compose up -d --force-recreate
echo "starting $n ... (first start compiles CUDA graphs, can take a few minutes)"
for i in $(seq 1 180); do
  curl -sf localhost:30000/health >/dev/null && { echo "ready: $n"; exit 0; }
  [ "$(docker inspect -f "{{.State.Status}}" neuron-sglang)" = running ] || { docker logs --tail 30 neuron-sglang; exit 1; }
  sleep 5
done
echo "timed out waiting for $n" >&2; exit 1
