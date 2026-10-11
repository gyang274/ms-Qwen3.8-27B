# One vs two RTX A6000s (§12 of the deployment notes)

Server variants are defined in `serve/bench-variant.sh`; each matrix was run with `scripts/bench/dual_matrix.sh`.

| Path | Contents |
|---|---|
| `link-p2p-off.json` | `link_bench.py`: host<->device bandwidth, peer-copy correctness (false: writes are dropped), NCCL all-reduce sweep with `NCCL_P2P_DISABLE=1` |
| `m1/{single,tp2,dp2,pp2}/` | Round 1: speed card (`speed.txt`) and agent sweep (`requests.jsonl`, `monitor.csv`) |
| `m2/{tp2,tp2-1m}/niah.jsonl` | Long-context retrieval, native 262K and YaRN 1M (the "131072" target actually produced a 221K-token prompt; the token estimate was recalibrated afterwards) |
| `m2/dp2/` | DP=2 rerun with the router loading the local tokenizer; same collapse as m1 |
| `m2/tp2-nomtp/`, `m2/single-x16/` | TP=2 without MTP; one GPU on the CPU x16 slot (speed card only) |
| `m3/dp2-tuned/`, `m3/dp2-sticky/` | DP=2 with balance thresholds 1 / with session-key affinity |
| `m3/tp2-1m-kv/` | TP=2, YaRN 1M, `--max-mamba-cache-size 32`: 1.33M-token KV pool, retrieval at 130K / 837K / 992K tokens |
| `power/` | GPU instantaneous power vs PSU telemetry |

`server-summary.txt` files hold the KV pool size, request cap and startup timings from each server log. The full
logs are not published because SGLang prints its launch arguments, including the API key.

Summarize any sweep with `python3 scripts/bench/agent_report.py results/dual-gpu/m1/tp2 ...`. Rows with
`"ok": false` are HTTP 400s: a reply cut off by the 512-token cap left a tool call with incomplete JSON
arguments, and SGLang rejects that history on the next turn ("Assistant tool call function.arguments must be
valid JSON"). That message is recorded for 20 of the 27 failed turns (out of 2,197); the other 7 came from the first
run, before the benchmark saved error bodies, and have the same 6 ms rejection signature. They are client-side
artifacts of the benchmark, not server errors.
