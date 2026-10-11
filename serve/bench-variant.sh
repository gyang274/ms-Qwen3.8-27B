#!/usr/bin/env bash
# Start one SGLang benchmark variant answering on port 30000 (the production compose project must be down).
# usage: bench-variant.sh single|single-x16|dp2|dp2-rr|dp2-tuned|dp2-sticky|tp2|tp2-nomtp|pp2|tp2-1m|tp2-1m-kv|pp2-1m [extra sglang args...]
# Containers are named neuron-bench*. Writes /dd/neuron/logs/bench-<variant>.log; returns once /health answers.
set -euo pipefail
V=$1; shift
EXTRA="$*"
RISER=GPU-a59b6394-c612-21dd-efab-3fa4fb00d4c9   # PCIEX16_2, chipset Gen3 x4
SLOT1=GPU-09aeaae4-7cb1-677f-203b-39dbf34087a0   # PCIEX16_1, CPU Gen4 x16
IMAGE=lmsysorg/sglang:v0.5.21-cu130
COMMON="--model-path /models/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b --host 0.0.0.0
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --enable-metrics --enable-cache-report"
MTP="--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec"
ONE="--context-length 262144 --mem-fraction-static 0.97 --max-running-requests 4 --max-mamba-cache-size 24 $MTP"
# YaRN to 1M tokens (factor 4 over the native 262,144), exactly as the Qwen model card gives it for SGLang.
YARN1M="--context-length 1000000 --json-model-override-args '{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true,
  \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25,
  \"factor\": 4.0, \"original_max_position_embeddings\": 262144}}}'"

# worker NAME GPUS PORT ARGS: one SGLang server on the host network.
# Peer-to-peer is reported available but every peer write is dropped by the AMD IOMMU on this board, so NCCL
# must stage through host memory and SGLang's P2P custom all-reduce must stay off (see scripts/bench/link_bench.py).
worker() {
  local args; args=$(printf '%s' "$COMMON --port $3 $4 $EXTRA" | tr '\n' ' ')  # a newline would end the bash -c command
  docker run -d --name "$1" --gpus "\"device=$2\"" --network host --ipc host --shm-size 16g \
    -e SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false -e NCCL_P2P_DISABLE=1 -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -v /dd/neuron/models:/models:ro -v /dd/neuron/cache/sglang-root:/root/.cache \
    -v /dd/neuron/configs/api-key:/run/secrets/api-key:ro $IMAGE \
    bash -c "exec python3 -m sglang.launch_server $args --api-key \"\$(cat /run/secrets/api-key)\"" >/dev/null
}
# router POLICY [ARGS]: sglang-router in front of the two dp workers, on port 30000. Health probes hit the
# non-generating /health, so they never wake an idle GPU.
router() {
  docker run -d --name neuron-bench --network host -v /dd/neuron/configs/api-key:/run/secrets/api-key:ro \
    -v /dd/neuron/models:/models:ro --entrypoint bash $IMAGE -c "exec python3 -m sglang_router.launch_router \
    --host 0.0.0.0 --port 30000 --model-path /models/Qwen3.8-27B-FP8 \
    --worker-urls http://127.0.0.1:30001 http://127.0.0.1:30002 --policy $1 ${2:-} --health-check-endpoint /health \
    --request-timeout-secs 3600 --api-key \"\$(cat /run/secrets/api-key)\"" >/dev/null
}

docker rm -f neuron-bench neuron-bench-w0 neuron-bench-w1 >/dev/null 2>&1 || true
NAMES=(neuron-bench)
case $V in
  single)     worker neuron-bench $RISER 30000 "$ONE" ;;
  single-x16) worker neuron-bench $SLOT1 30000 "$ONE" ;;
  dp2|dp2-rr|dp2-tuned|dp2-sticky) worker neuron-bench-w0 $RISER 30001 "$ONE"; worker neuron-bench-w1 $SLOT1 30002 "$ONE"
              NAMES=(neuron-bench-w0 neuron-bench-w1) ;;
  # TP: prefill CUDA-graph capture needs ~3 GB per GPU with NCCL; 0.95 and 0.92 OOMed during capture.
  tp2)        worker neuron-bench $RISER,$SLOT1 30000 "--tp-size 2 --disable-custom-all-reduce --context-length 262144 --mem-fraction-static 0.88 $MTP" ;;
  tp2-nomtp)  worker neuron-bench $RISER,$SLOT1 30000 "--tp-size 2 --disable-custom-all-reduce --context-length 262144 --mem-fraction-static 0.88" ;;
  # Hybrid model: the mamba state cache (169 slots, ~12 GB per GPU) competes with KV. 32 slots -> KV pool > 1M.
  tp2-1m-kv)  worker neuron-bench $RISER,$SLOT1 30000 "--tp-size 2 --disable-custom-all-reduce --mem-fraction-static 0.88 --max-mamba-cache-size 32 $MTP $YARN1M" ;;
  tp2-1m)     worker neuron-bench $RISER,$SLOT1 30000 "--tp-size 2 --disable-custom-all-reduce --mem-fraction-static 0.88 $MTP $YARN1M" ;;
  # PP: SGLang v0.5.21 asserts "PP + speculative decoding (MTP) is only supported on prefill nodes", so no MTP.
  pp2)        worker neuron-bench $RISER,$SLOT1 30000 "--pp-size 2 --context-length 262144 --mem-fraction-static 0.92" ;;
  *)          echo "unknown variant $V"; exit 2 ;;
esac

LOG=/dd/neuron/logs/bench-$V.log
dump() { for n in "${NAMES[@]}"; do echo "### $n"; docker logs "$n" 2>&1; done > "$LOG"; }
T0=$(date +%s)
ready() { for p in "$@"; do curl -sf -m 5 "http://localhost:$p/health" >/dev/null || return 1; done; }
PORTS=(30000); [[ $V == dp2* ]] && PORTS=(30001 30002)
for _ in $(seq 1 180); do
  sleep 10
  for n in "${NAMES[@]}"; do
    if ! docker inspect -f '{{.State.Running}}' "$n" 2>/dev/null | grep -q true; then
      dump; echo "FAILED ($n exited), see $LOG"; exit 1
    fi
  done
  if ready "${PORTS[@]}"; then
    if [[ $V == dp2* ]]; then
      case $V in
        dp2-rr)     router round_robin ;;
        # Default cache_aware rebalances only when loads differ by >64 requests; with a shared agent system
        # prompt and a dozen agents that never happens and every session lands on one worker.
        dp2-tuned)  router cache_aware "--balance-abs-threshold 1 --balance-rel-threshold 1.0" ;;
        # Session affinity: the client sends X-SGL-Routing-Key (one key per agent session); new keys go to
        # the least-loaded worker and stay there.
        dp2-sticky) router manual "--assignment-mode min_load" ;;
        *)          router cache_aware ;;
      esac
      NAMES+=(neuron-bench); sleep 10
      ready 30000 || { dump; echo "FAILED (router)"; exit 1; }
    fi
    dump; echo "READY in $(( $(date +%s) - T0 )) s"; exit 0
  fi
done
dump; echo "TIMEOUT, see $LOG"; exit 1
