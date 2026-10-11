# Idle power vs health-probe frequency

The GPU idles with SGLang loaded and no traffic. `/health` is polled at an extra interval on top of the
container healthcheck. Data from `scripts/bench/idle_wake_test.sh` (6 min per phase, first minute dropped,
P-state sampled every 5 s).

| Phase | `/health` runs a 1-token generation (SGLang default) | `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false` |
|---|---|---|
| container check only | P2 28%, 7.7 W (container check every 60 s) | P2 0%, 7.6 W (container check every 30 s) |
| + every 30 s | P2 73%, 72.5 W | - |
| + every 15 s | P2 100%, 76.7 W | - |
| + every 10 s | P2 100%, 77.3 W | **P2 0%, 7.2 W** |

Files: `generating-health-probe.csv`, `lightweight-health-probe.csv`. The CSV phase label is the extra
polling interval in seconds.
