# GPU vs PSU power under two-GPU load

From `scripts/bench/power_transients.sh`. For 120 s per run: `nvidia-smi --query-gpu=power.draw.instant -lms 100`
for both GPUs (`*-gpu.csv`: timestamp, index, watts) and the Corsair HX1000i's reported output power from the
`corsair-psu` hwmon driver once per second (`*-psu.csv`: unix time, watts).

| Run | Load |
|---|---|
| `tp2-decode`, `pp2-decode` | 16 concurrent 4,096-token generations |
| `tp2-agents` | 8 closed-loop coding agents (`agent_bench.py`), prefill-heavy |

CPU calibration (GPUs idle, no server): PSU read 122-128 W idle, 252-254 W with 16 busy threads, 240-246 W with 32.
