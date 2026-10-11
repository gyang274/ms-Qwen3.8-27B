# Soak report: soak-20261009-0119

2617 requests over ~10 h

## Requests by kind

| kind | n | errors | TTFT p50 / p95 (s) | decode p50 (tok/s) | max prompt tokens |
|---|---|---|---|---|---|
| burst | 60 | 0 | 0.25 / 7.42 | 42.2 | 45 |
| coding | 1705 | 0 | 6.57 / 14.19 | 48.4 | 82361 |
| coding-think | 339 | 0 | 8.97 / 17.86 | 23.4 | 75863 |
| fullload | 376 | 0 | 20.18 / 35.13 | 53.2 | 52 |
| long-prefill | 4 | 0 | 200.66 / 200.91 | 47.7 | 149330 |
| probe | 10 | 0 | 0.12 / 0.12 | 64.1 | 31 |
| robot | 103 | 0 | 3.13 / 30.27 | 25.4 | 31 |
| robot-image | 20 | 0 | 3.88 / 32.75 | 42.6 | 87 |

errors: 0; tool calls 1324, malformed arguments 0

## Hourly drift (robot + coding, single-stream-ish)

| hour | requests | errors | robot TTFT p50 | decode p50 |
|---|---|---|---|---|
| 1 | 253 | 0 | 4.09 | 44.9 |
| 2 | 280 | 0 | 3.58 | 44.2 |
| 3 | 263 | 0 | 0.29 | 48.3 |
| 4 | 206 | 0 | 15.97 | 39.2 |
| 5 | 271 | 0 | 4.36 | 48.6 |
| 6 | 268 | 0 | 2.48 | 47.1 |
| 7 | 283 | 0 | 1.68 | 43.9 |
| 8 | 282 | 0 | 0.88 | 52.8 |
| 9 | 211 | 0 | 20.22 | 41.7 |
| 10 | 300 | 0 | 0.33 | 49.7 |

## GPU / container

- temperature max 74 C, p95 73 C; fan max 75%
- thermal slowdown samples: 0
- VRAM first 10% avg 48250 MiB, last 10% avg 48251 MiB, max 48251
- host RAM used first 10% avg 8775 MiB, last 10% avg 8916 MiB
- container restart count values seen: ['0']; non-healthy samples: 12
- idle samples 296: pstates ['P2', 'P8'], power p50 19.3 W, container CPU p50 1.4%

## Consistency probe

- 10 probes, 10 ok, 1 distinct answer(s)
  - '2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37\nSum: 197'

## System logs during the run

- kernel Xid / AER / MCE: 0
- fan health warnings: 0
- sglang errors/tracebacks: 0
- OOM killer: 0
- apt/unattended activity: 2
