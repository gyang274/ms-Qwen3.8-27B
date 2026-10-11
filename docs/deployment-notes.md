# Serving Qwen3.8-27B on one RTX A6000: FP8 on Ampere, a verified abliterated quant, and a 10-hour soak

These are the engineering notes behind a single-GPU home LLM server: one user, many sessions, serving a home
robot and coding agents. The hardware is a five-year-old 48 GB Ampere workstation card. Most published Qwen3.8
recipes target Hopper or Blackwell. These notes cover what changes on `sm_86`, how the memory budget was tuned
to fit the native 262K context, how power, clocks, temperature and fan speed were balanced, how Qwen's own FP8
recipe was reproduced byte for byte, what an abliteration edit looks like inside the weights, and how a mixed
10-hour soak test was designed and what it found.

The FP8 abliterated model produced here is published at
[huggingface.co/gyang274/Huihui-Qwen3.8-27B-abliterated-FP8](https://huggingface.co/gyang274/Huihui-Qwen3.8-27B-abliterated-FP8).

## TL;DR

| | Result |
|---|---|
| Engine | SGLang v0.5.21 (`lmsysorg/sglang:v0.5.21-cu130`), Qwen3.8-27B-FP8, MTP speculative decoding |
| FP8 on Ampere | Runs as weight-only W8A16 through Marlin kernels (auto-selected on `sm_86`); FP8 E4M3 KV cache |
| Context | Native 262,144 tokens per request; 247,561-token shared KV pool; 4 concurrent requests |
| Decode | ~60 tok/s single stream (51 with thinking), ~150-160 tok/s aggregate at 4 streams, ~47 tok/s at 134K context |
| Prefill | ~1,000 tok/s; a 134K-token prompt takes 140 s cold, a prefix-cache hit answers in 0.14 s |
| Power/thermals | 250 W cap + custom fan curve: 74 °C / 75% fan under sustained load (stock 300 W: 87 °C and thermal slowdown); no decode loss |
| Idle | ~7-8 W in P8, SGLang CPU ~1.4% (`--sleep-on-idle` plus a non-generating health check, without which the GPU can sit in P2 at ~73 W) |
| FP8 recipe | Our quantizer reproduces the official FP8 checkpoint byte for byte on every unmodified tensor |
| Abliteration | Exactly 70 tensors differ: the residual-stream writers of layers 17-51. Every edit is ΔW = −1.3 · r rᵀW with **one shared direction r** |
| Cost of abliteration | HumanEval 98.2% -> 95.7%, GSM8K-200 97.5% -> 94.5%, over-refusals 2/15 -> 0/15, same speed |
| Soak | 10 h mixed workload, 2,617 requests, 0 errors, 1,324 tool calls with 0 malformed, no restarts, no Xid/OOM. It also exposed an idle-power bug (a generating health check pinned the GPU in P2 at 73 W), which was root-caused and fixed |

## 1. Goals and constraints

The server backs a home robot and coding agents (for example as the backend of a
[pi agent](https://github.com/badlogic/pi-mono)). There is one user with several concurrent sessions. In
priority order:

1. **Stability.** It runs unattended for weeks.
2. **Low idle cost.** It sits idle most of the day, so idle GPU power and idle CPU usage matter.
3. **Precision.** FP8 is preferred over NVFP4, and BF16 does not fit. Accuracy loss should be measured, not
   assumed.
4. **Context.** The native 262K window (1M via YaRN is out of reach on 48 GB, see §5).
5. **Speed.** The ideal target was over 100 tok/s, but this is the most negotiable goal.

## 2. Platform

| Component | Detail |
|---|---|
| GPU | NVIDIA RTX A6000, 48 GB GDDR6, GA102 `sm_86`, 768 GB/s, 300 W blower |
| Host | AMD Ryzen 9 5950X (16C/32T), 128 GB RAM, ASUS ROG STRIX B550-F, GPU on CPU PCIe 4.0 x16 |
| Storage | Samsung 980 PRO 2 TB NVMe for models and container storage |
| OS | Ubuntu 24.04.4 LTS, HWE kernel 6.17, NVIDIA driver 590.48.01 (open), CUDA 13.1 |
| Runtime | Docker 29 + NVIDIA Container Toolkit 1.20; image and container storage on the NVMe |

Platform decisions:

- **Headless.** With GDM running, Xorg and gnome-shell keep 120-250 MB of VRAM on the compute GPU even with no
  monitor attached. Booting to `multi-user.target` brings this to 33 MiB. The desktop is started on demand with
  `systemctl start gdm`.
- **No second display GPU.** The only spare card was a GTX 1080 Ti (Pascal). Driver branch 590 no longer
  supports Pascal (580 is the last), so one driver cannot run both cards. A second card is also unnecessary. A
  model that is fully resident in VRAM barely uses PCIe, so even an x8/x8 split would not slow down single-GPU
  decoding.
- **Persistence mode.** Ubuntu's `nvidia-persistenced` unit starts with `--no-persistence-mode`. A drop-in
  override removes that flag ([system/nvidia-persistenced-override.conf](../system/nvidia-persistenced-override.conf)).
- **Pinned GPU stack.** `unattended-upgrades` was enabled, and its security origin carries NVIDIA driver
  updates. A silent driver update leaves the running kernel module and the new user-space libraries mismatched,
  and the server stays broken until reboot. The 24 NVIDIA packages are `apt-mark hold`-ed, and the driver and
  kernel are blacklisted from unattended upgrades
  ([system/51neuron-unattended-blacklist](../system/51neuron-unattended-blacklist)). Everything else still
  auto-updates. During the soak test it upgraded libc6, systemd and openssh mid-run with no effect on serving.

## 3. Choosing the engine

Three options were on the table for this hardware:

| Option | Verdict |
|---|---|
| **SGLang + Qwen3.8-27B-FP8** | Chosen. Mature, multi-session scheduling, radix prefix cache, MTP speculative decoding, OpenAI/Anthropic APIs, reasoning and tool-call parsers |
| NInfer (community `sm_86` fork) + Qwen3.8-27B INT8 | Fast (about 71 tok/s single stream on a 3090), and INT8 group-64 W8A16 is in the same precision tier as FP8. But it is a best-effort fork with open Linux qualification items and fixed concurrency. Kept as a benchmark fallback |
| Strata + Qwen3.8-Flash-Next (125B MoE) | Rejected. It fits only with 2-3-bit quantization plus RAM/SSD expert offload, serves one request at a time, and is CPU-heavy |

## 4. FP8 on Ampere: what actually runs

Ampere has no FP8 tensor cores. SGLang v0.5.21 detects this and switches the block-quantized FP8 checkpoint to
**Marlin weight-only FP8 (W8A16)** on its own, with no flag needed:

```
Your GPU does not have native support for FP8 computation but FP8 quantization is being used.
Weight-only FP8 compression will be used leveraging the Marlin kernel.
```

The weights stay FP8 in memory, so the bandwidth cost is the same as native FP8. Each GEMM dequantizes them to
BF16, so the math runs in BF16 with no activation quantization. Decoding at batch 1-4 is memory-bound, which is
where weight-only quantization pays off: about 27 GB of weights are read per token at 768 GB/s. That puts the
ceiling near 28 tok/s without speculation, and MTP more than doubles it (§6). Prefill is compute-bound, and
there Marlin W8A16 on GA102 manages about 1,000 tok/s.

The KV cache uses FP8 E4M3 (`--kv-cache-dtype fp8_e4m3`). It stores FP8 and computes in BF16, and it halves the
KV footprint, which is what makes 262K fit.

## 5. Fitting 262K context in 48 GB

Qwen3.8-27B is a hybrid model. Of its 64 layers, 48 are Gated DeltaNet (linear attention with a fixed-size
recurrent state) and only 16 are full attention, with 4 KV heads × 256 dims each. KV therefore costs about
32 KiB/token in FP8, so 262K tokens need about 8 GiB. Each concurrent request also holds a recurrent "mamba"
state of about 150 MB.

Tuning steps, all at `context_length=262144`:

| Step | KV pool (tokens) | Max running requests | Unused VRAM at end |
|---|---|---|---|
| `--mem-fraction-static 0.88`, defaults | 122,119 | 2 (capped by mamba state slots) | 7.9 GB |
| + `--enable-linear-replayssm-spec`, `--max-mamba-cache-size 24`, mem 0.92 | 170,505 | 4 | 5.7 GB |
| mem 0.97 | **247,561** | **4** | 3.3 GB |

Two findings drive this:

- **The MTP draft head's embedding and LM head are shared *after* the KV pool is sized.** The MTP module loads
  its own 248K-vocab embedding and LM head (5.3 GB). SGLang then ties them to the target model's weights and
  frees the copies, but this happens after the KV pool has been allocated, so the freed memory sits idle. Setting
  `--mem-fraction-static 0.97`, which looks aggressive, simply gives that memory to the KV pool.
- **ReplaySSM spec-verify** (`--enable-linear-replayssm-spec`) replaces the per-draft full recurrent-state
  snapshots used during MTP verification (1.69 GB) with a small replay window. That frees enough room for 24
  mamba slots and 4 concurrent requests.

1M context via YaRN would need about 64 GiB of KV in BF16 or 32 GiB in FP8, on top of 29 GB of weights. It does
not fit on this card.

The final launch command is in [serve/compose.yaml](../serve/compose.yaml).

## 6. Performance

| Measurement | Result |
|---|---|
| Decode, single stream, thinking off | 58-62 tok/s |
| Decode, single stream, thinking on | ~51 tok/s |
| Decode at 134K context | ~47 tok/s |
| Aggregate decode, 2 / 4 streams | ~102 / ~150-160 tok/s |
| MTP mean accepted length (3 draft steps) | 3.52 tokens per verify (coding and Q&A mix) |
| Prefill, 38K-token prompt | 38.6 s (~1,000 tok/s) |
| Prefill, 134K-token prompt | 174 s with 4,096-token chunks, **140 s** with `--chunked-prefill-size 8192` |
| Same 38K prompt again (radix-cache hit) | TTFT **0.14 s** |

The practical consequence for coding agents is that the first read of a large context is slow, but later turns
reuse the cached prefix and respond immediately. The radix cache makes multi-turn agents viable on this card.

## 7. Power, clocks, temperature and fan

The A6000 is a blower card with an 84 °C temperature target, a 93 °C maximum operating temperature and thermal
slowdown at 95 °C. Under sustained load at the stock 300 W it settled at 87 °C and reported SW thermal slowdown
(`clocks_event_reasons = 0x20`). Case airflow has little effect on a blower, so two controls were swept:

![Power and cooling sweep](../figures/power-cooling.png)

| Power cap | Driver fan curve | Custom fan curve | 38K prefill | Single-stream decode |
|---|---|---|---|---|
| 300 W | 87 °C / 68% fan, **thermal slowdown** | 78 °C / 84% | 38.5-38.6 s | ~60 tok/s |
| 280 W | 85 °C / 67% | - | 38.8 s | ~60 tok/s |
| **250 W** | 83 °C / 60% | **74 °C / 75%** | 39.1-39.6 s | ~61 tok/s |
| 220 W | 79 °C / 56% | 72 °C / 70% | 42.8-45.7 s | ~59 tok/s |

Takeaways:

- **Decode does not depend on core clocks.** It is bandwidth-bound, so it is flat at about 60 tok/s from 220 W
  to 300 W. Only prefill pays: -1.5% at 250 W and -11-18% at 220 W.
- **The driver's fan curve tracks its 84 °C target.** A lower power cap therefore mostly lowers fan speed, not
  temperature. To lower temperature you need a different fan curve.
- **Fan control works headless through NVML.** `nvmlDeviceSetFanSpeed_v2` works on the A6000 without X or
  Coolbits. [scripts/system/gpu_fan_curve.py](../scripts/system/gpu_fan_curve.py) implements the curve
  `45:30, 55:40, 65:55, 72:70, 78:85, 82:100` (°C:%). It ramps up immediately and down by at most 2% per 2 s, and
  forces the fan to 100% if a temperature read fails. On exit it hands control back to the driver, including
  when SIGKILLed (via `ExecStopPost`). A health check logs a warning when the measured fan speed stays more than
  15 points below the commanded speed for 2 minutes, an early sign of a worn bearing.
- **Deployed: 250 W cap and the custom curve.** That is 13 °C cooler than stock, with 7 points more fan speed
  and no measurable decode cost. Bearing wear grows roughly with RPM × hours. The extra RPM only applies under
  load, and idle stays at 30%, so the lifetime cost is a few percent of the fan's life. Meanwhile the
  temperature drop benefits the core, VRMs and the backside GDDR6, and the fan is a replaceable part.

## 8. BF16 to FP8: reproducing Qwen's recipe byte for byte

The abliterated model ([huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated),
revision `739e3c5`) ships only in BF16, which is 55.6 GB and does not fit. Rather than use a generic quantizer,
[scripts/quantize/quantize_fp8_like.py](../scripts/quantize/quantize_fp8_like.py) uses the official
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) checkpoint as a **layout template**:

- A tensor is quantized to FP8 E4M3 with 128×128 block scales (`weight_scale_inv`) exactly when it is FP8 in the
  template, with the scale stored in the template's dtype (BF16).
- Every other tensor is copied in the template's dtype. This covers norms, the DeltaNet gates, the vision tower,
  the embedding and LM head, and the MTP fusion layers.
- Names, dtypes and shapes are asserted to match the template. `config.json`, with its `quantization_config`,
  comes from the template.

The result loads through exactly the same SGLang code path as the official FP8. The first version matched the
official bytes on only 98.4% of elements in unmodified tensors. A probe ([scripts/quantize/probe_quant.py](../scripts/quantize/probe_quant.py))
tried candidate recipes against tensors that abliteration does not touch:

![Recipe probe](../figures/fp8-recipe-match.png)

Qwen's recipe computes the block scale in **FP32** as `amax × (1/448)`, quantizes with that FP32 scale, and
only then rounds the scale to BF16 for storage. Rounding the scale to BF16 *before* quantizing (the "consistent"
choice, recipe A) changes 1.6% of the FP8 codes. With the final recipe,
[compare_fp8.py](../scripts/quantize/compare_fp8.py) shows that every tensor abliteration did not modify is
**byte-identical** to the official FP8 checkpoint: layers 0-16 and 52-63, the MTP head, the vision tower
(333/333), and the embedding and LM head. The maximum relative weight error of the quantization itself is
2.65%, which is normal for E4M3.

This byte-identity check is the strongest evidence that the published FP8 model and the official one differ
*only* where the abliteration changed the weights.

## 9. Anatomy of an abliteration

Comparing the two FP8 checkpoints tensor by tensor gives the edit's footprint:

![Abliteration map](../figures/abliteration-map.png)

**Exactly 70 of the 1,606 tensors differ, one per sub-block in layers 17-51:**

| Tensor | Count | Role |
|---|---|---|
| `mlp.down_proj.weight` | 35 | MLP output projection: writes into the residual stream |
| `linear_attn.out_proj.weight` | 26 | Gated DeltaNet output projection: writes into the residual stream |
| `self_attn.o_proj.weight` | 9 | Full-attention output projection: writes into the residual stream |

Nothing else changed. That covers every Q/K/V projection, every MLP gate and up projection, every norm, the
embedding, the LM head and the MTP head.

This is exactly the footprint *directional ablation* predicts (Arditi et al., 2024,
["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717)). If refusal
is mediated by one direction *r* in the residual stream, you can stop the model from ever writing along *r* by
projecting it out of every matrix that writes into the stream: W ← W − r rᵀW. Matrices that only *read* from
the stream (Q/K/V, gate/up) need no change, because nothing can write *r* back. Restricting the edit to
the middle layers (17-51 of 64) leaves early feature extraction and late token formation untouched. huihui's
model card says the narrower range was chosen to preserve capability.

**Rank and direction analysis.** The footprint is consistent with the theory. The weights themselves can test
it. If the edit really is W ← W − α r rᵀW with one shared r, then three things must hold for every
ΔW = W_abl − W_orig, computed in BF16 against the official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
weights:

1. ΔW is **rank 1**.
2. Its left singular vector is the **same vector r** in every edited tensor.
3. ΔW ≈ −α r rᵀW for a single **α**.

[scripts/analysis/abliteration_svd.py](../scripts/analysis/abliteration_svd.py) checks all three. The script
was first validated on a synthetic rank-1 edit, where it recovered α = 1.0000 and |cos| = 1.0000.

![Abliteration SVD](../figures/abliteration-svd.png)

| Test | Result over the 70 edited tensors |
|---|---|
| Changed vs bit-identical BF16 tensors | 70 changed, 1,129 bit-identical |
| Size of the edit | ‖ΔW‖ / ‖W‖ = 1.8-2.2% |
| Rank 1 | σ1 holds 99.23-99.49% of ‖ΔW‖²; σ2/σ1 ≤ 0.012 |
| One direction | pairwise \|cos\| between the 70 left singular vectors ≥ 0.99999 |
| One scale | least-squares α = 1.2992-1.3003 in every tensor |
| Rebuild from a single r, α = 1.3 | 91.9% of BF16 elements bit-exact, residual 3.9% of ‖ΔW‖ |
| Same rebuild with α = 1.0 | 46.6% bit-exact, residual 25.8% |

So the whole abliteration is one 5,120-dimensional unit vector and one scalar:
**ΔW = −1.3 · r rᵀW**, applied identically to the 70 residual-stream writers of layers 17-51. The scale is
not 1. With α = 1.3 the edit does more than remove the component along *r*: it reverses that component to −0.3×
its original value, deliberately overshooting so that these layers push the residual stream *away* from the
refusal direction. The last 8% of bit mismatches come from our estimate of *r*, not from the formula. *r* is
recovered from ΔW, which is itself the difference of two BF16-rounded tensors, so tiny errors in *r* flip
roundings that sit at a BF16 boundary. Probing fp32, fp64, BF16 and explicit-projector arithmetic did not
move the match above 92%
([results/abliteration/arithmetic-probe.txt](../results/abliteration/arithmetic-probe.txt)).

Why does a single direction in 35 of 64 layers suffice? Refusal is not stored in any one weight. It is a
feature the model *writes* into the residual stream and later layers *read*. Every component communicates
only through that stream, so constraining what the middle layers can write along *r* is enough, and the
reading weights stay untouched. This is also why the cost shows up as behavior rather than broken capability
(§10). The same edit that removes refusals also blunts a direction the model uses to commit to an answer.
The analysis covers only this checkpoint, and the recovered *r* is published in
[results/abliteration/svd.directions.pt](../results/abliteration/svd.directions.pt).

## 10. Official vs abliterated: what the edit costs

Both FP8 models were run with the same serving configuration, thinking off and temperature 0
([scripts/bench/eval_compare.py](../scripts/bench/eval_compare.py)). HumanEval solutions were executed in a
network-less, memory-capped throwaway container.

| | Official FP8 | Abliterated FP8 (this repo) |
|---|---|---|
| HumanEval pass@1 (164) | **98.2%** (3 failed) | 95.7% (7 failed) |
| GSM8K, first 200 | **97.5%** | 94.5% |
| Over-refusal: 15 benign prompts that safety-tuned models often refuse | 2 partial refusals | **0** |
| Decode, single stream / 4 streams | ~60 / ~150 tok/s | ~62 / ~142 tok/s |
| MTP mean accepted length | 3.52 | 3.52 |

- **The accuracy cost is real and one-directional.** On GSM8K the abliterated model lost 6 problems the
  official model solved and gained none. Most of these were not arithmetic errors. The model became indecisive
  on ambiguously worded problems and debated interpretations until it hit the token limit. This fits the edit
  weakening a direction that also carries "commit to an answer" behavior.
- **The official model's refusals were mild.** For "pick the lock on my own front door" it gave alternatives
  (locksmith, landlord) instead of steps. For "a phishing example for employee training" it gave an annotated
  teaching example rather than a realistic one. Nmap, SQL injection, villain monologues and medication dosing
  were all answered normally.
- **The MTP head is unaffected.** Its acceptance rate is identical, because the edit does not move the hidden
  states enough to hurt the draft predictions.

The deployment serves the official model by default, since the robot talks to children and coding accuracy
compounds over long agent runs. The abliterated model is one command away (`serve/switch-model.sh abliterated`).

## 11. A 10-hour soak test

### Design

Neither full load nor plain idle finds the failures that matter for an always-on home server. Thermal headroom
was already characterized in §7. What remains are transitions and accumulation: idle to busy wake-ups, prefix-cache
eviction as agent contexts grow, queueing past the concurrency limit, and slow memory growth. The test
([scripts/bench/soak_test.py](../scripts/bench/soak_test.py)) repeats a one-hour cycle ten times:

| Phase (per hour) | Workload | Exercises |
|---|---|---|
| min 0-35 | 3 coding-agent sessions with tool definitions. Tool calls are answered with synthetic code, so contexts grow from 3K to 50-100K tokens before restarting. One session uses thinking mode | Prefix cache, eviction, tool-call parsing, long contexts |
| min 0-35 | Home-robot chat every 1-4 min, 20% with an image | Interactive TTFT under load, vision path |
| min 35 | Burst of 6 simultaneous requests (limit is 4) | Queueing |
| min 38 | Temperature-0 probe with a fixed prompt | Numerical drift |
| min 40-60 | Idle | Return to P8, `--sleep-on-idle` wake-up |
| hours 1, 3, 6, 8 | ~150K-token prompt | Peak memory, long prefill |
| hours 4, 9 | 30 min of 6 concurrent long generations, replacing the coding phase | Thermal steady state, power cap |

A monitor samples temperature, fan speed, power, P-state, clocks, VRAM, container restarts and health,
container CPU and memory, and host RAM every 30 s. Pass criteria were set before the run: zero restarts, zero
5xx errors and timeouts, valid tool calls, under 5% drift from hour 1 to hour 10, no VRAM or RAM growth trend,
identical probe answers, no Xid/AER/MCE/OOM, temperature at or below 80 °C, and idle at P8 with SGLang CPU at or
below 2%.

### Results

![10-hour soak](../figures/soak-10h.png)

| Request kind | n | Errors | TTFT p50 / p95 | Decode p50 | Max prompt |
|---|---|---|---|---|---|
| coding | 1,705 | 0 | 6.6 / 14.2 s | 48.4 tok/s | 82,361 |
| coding (thinking) | 339 | 0 | 9.0 / 17.9 s | 23.4 tok/s | 75,863 |
| robot | 103 | 0 | 3.1 / 30.3 s | 25.4 tok/s | 31 |
| robot + image | 20 | 0 | 3.9 / 32.8 s | 42.6 tok/s | 87 |
| burst | 60 | 0 | 0.25 / 7.4 s | 42.2 tok/s | 45 |
| full load | 376 | 0 | 20.2 / 35.1 s | 53.2 tok/s | 52 |
| 150K prefill | 4 | 0 | 200.7 s | 47.7 tok/s | 149,330 |
| probe | 10 | 0 | 0.12 s | 64.1 tok/s | 31 |

- **2,617 requests, 0 errors.** There were 1,324 tool calls and none had malformed JSON arguments. The container
  never restarted, and the kernel logged no Xid, AER, MCE or OOM events. The fan health check raised no
  warnings.
- **No drift.** Decode p50 was 44.9 tok/s in hour 1 and 49.7 tok/s in hour 10. The temperature-0 probe returned
  the identical answer all 10 times.
- **No leaks.** VRAM stayed flat at 48.25 GB (pools are preallocated). Host RAM grew 140 MB over 10 hours.
- **Thermals:** the maximum was 74 °C at 75% fan, with no thermal slowdown.
- **Verdict:** every pass criterion held except one. *Idle at P8* failed in 4 of the 10 idle windows. This was
  missed at first, because the summary statistic only counted low-power samples. It was caught when the
  timeline was plotted, then root-caused and fixed (finding 3).
- **Findings:**
  1. *Health check false negatives during long prefills.* By default SGLang's `/health` issues a one-token
     generation and waits for the scheduler. A 150K-token prefill keeps the scheduler busy for over 3 minutes,
     so with a 10 s timeout and 3 retries Docker flagged the container `unhealthy` during every long prefill,
     even though the server never stopped serving. Finding 3 fixes the root cause.
  2. *Robot latency under heavy coding load.* With 6 heavy streams and a limit of 4 running requests, robot
     requests wait in the FIFO queue: TTFT p50 16 s and p95 35 s, against 2.5 s and 9.3 s otherwise. Priority
     scheduling for interactive requests is the next improvement.
  3. *Idle windows stuck in P2: the health check was waking the GPU.* In 4 of 10 idle windows the GPU stayed
     in P2 at 1,800 MHz and ~73 W, with 0% utilization and SGLang CPU at 1.4%. The other 6 reached P8 at
     ~19 W. The request logs showed no difference between good and bad windows. A compressed-cycle re-run did
     not reproduce the problem, and that re-run had one difference: the Docker health check had since moved
     from every 30 s to every 60 s. A controlled test ([scripts/bench/idle_wake_test.sh](../scripts/bench/idle_wake_test.sh),
     [results/idle-power/](../results/idle-power/)) confirmed the cause. Each generating `/health` call runs
     a non-CUDA-graph forward pass that wakes the GPU. If wake-ups arrive every ≤ 30 s, the driver never
     completes its P2 → P8 transition, which takes 30-60 s on this card:

     | Extra `/health` polling (idle) | Generating `/health` (default) | `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false` |
     |---|---|---|
     | none | P2 28%, 7.7 W | P2 0%, 7.6 W |
     | every 30 s | P2 73%, 72.5 W | - |
     | every 10-15 s | P2 100%, ~77 W | **P2 0%, 7.2 W** |

     The soak's 30 s health check sat right on that boundary, which made the problem intermittent. It also
     explains why the "good" idle windows read 19 W rather than 8 W. The fix is
     `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false`. With it, `/health` answers from the HTTP layer in about
     1 ms and never touches the GPU, and `/health_generate` remains available as a deep check. The same change
     removes the false negatives of finding 1. Any client that polls a generating endpoint (a dashboard, an
     agent heartbeat) would cause the same 65 W idle penalty, about 570 kWh a year if it ran around the clock.

## 12. Operations

- **Serving:** Docker Compose project `neuron` ([serve/compose.yaml](../serve/compose.yaml)) with
  `restart: unless-stopped` and log rotation. The API key is mounted as a file and read at container start, so it
  never appears in `docker inspect`. The health check polls a non-generating `/health` every 30 s (see §11,
  finding 3).
- **Model switch:** `serve/switch-model.sh official|abliterated` rewrites `.env`, recreates the container and
  waits for health. The first start compiles CUDA graphs (about 4 min). Later starts take about 2 min.
- **Boot chain:** `nvidia-persistenced` (persistence on), then
  [gpu-power-limit.service](../system/gpu-power-limit.service) (250 W), then
  [gpu-fan-curve.service](../system/gpu-fan-curve.service), then Docker, then SGLang. After a reboot the API is
  ready in about 4 minutes, verified end to end over the LAN including tool calling.
- **Upgrades:** driver and kernel upgrades are manual: unhold, upgrade, reboot, re-hold.

## 13. Limitations and what changes on Blackwell

- On this card, single-stream decode tops out around 60 tok/s and prefill around 1,000 tok/s. Both are hardware
  limits (768 GB/s, no FP8 tensor cores), not tuning gaps.
- 262K is the per-request maximum, but it uses the whole KV pool. Concurrent sessions share about 248K tokens.
- The planned upgrade is an RTX PRO 6000 (96 GB, about 1.8 TB/s, native FP8/FP4) in the x16 slot, with an RTX
  PRO 4000 for display and auxiliary models. No tensor parallelism across the asymmetric pair is planned. It
  should bring native W8A8 FP8, 1M context with an FP8 KV cache, and well over 100 tok/s single stream with MTP.

## 14. Reproduce

| Path | What |
|---|---|
| [serve/](../serve/) | Compose file and model switch script |
| [system/](../system/) | systemd units, persistence override, unattended-upgrades blacklist |
| [scripts/quantize/](../scripts/quantize/) | Template-driven FP8 quantizer, checkpoint comparer, recipe probe |
| [scripts/analysis/](../scripts/analysis/) | Abliteration SVD analysis |
| [scripts/bench/](../scripts/bench/) | Speed bench, model comparison, power sweep, soak test and report |
| [scripts/system/](../scripts/system/) | NVML fan-curve daemon, model download |
| [scripts/figures/](../scripts/figures/) | Renders every figure from `results/` |
| [results/](../results/) | Raw data behind every number in these notes (incl. `abliteration/` SVD outputs and recovered direction, `idle-power/`) |
