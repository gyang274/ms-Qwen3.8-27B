#!/usr/bin/env python3
"""Render the figures in figures/ from the raw data in results/.

usage: python scripts/figures/make_figures.py   (run from the repository root; needs matplotlib, numpy)
"""
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, FIG = ROOT / "results", ROOT / "figures"

# palette (brand scheme 'Muted': dark gold / deep blue / muted purple; see ~/.claude/preferences/colors.md):
# categorical order gold -> blue; a single series uses the primary blue; gold marks the highlight
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#2b2925", "#5c5955", "#736f6b", "#e7e3dd"
GOLD, BLUE = "#a4772e", "#03628b"
ACCENT, ACCENT_SOFT, ACCENT_FAINT = BLUE, "#9bc2db", "#e0edf6"
HIGHLIGHT, HIGHLIGHT_SOFT = GOLD, "#d1b895"
CONTRAST = GOLD
NEUTRAL, CELL, CELL_FAINT = "#bfbbb7", "#eeeae5", "#f6f3ef"
CAT = ["#a4772e", "#03628b", "#8f77bd", "#0d8b74", "#853e24", "#7a9c55", "#7f3a5d"]  # categorical, fixed order

FONT = next((f for f in ("Inter", "Noto Sans") if any(f == x.name for x in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams.update({
    "font.family": FONT, "font.size": 10.5, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": NEUTRAL, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "lines.linewidth": 2.2, "axes.unicode_minus": False, "lines.solid_capstyle": "round", "legend.frameon": False,
})


def header(fig, title, subtitle):
    """Title and subtitle at fixed pixel offsets from the top-left corner."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.22 / h, title, fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.56 / h, subtitle, fontsize=11, color=INK2, va="top")


def footer(fig, text):
    fig.text(0.012, 0.14 / fig.get_figheight(), text, fontsize=8.5, color=MUTED, va="bottom")


def panel_title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=INK, pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)
    print("wrote", FIG / name)


# --------------------------------------------------------------------------- power / cooling

def parse_sweep(path):
    out, cur = {}, None
    for line in open(path):
        if m := re.match(r"=== power limit (\d+) W", line):
            cur = int(m.group(1))
            out[cur] = {}
        elif cur and line.startswith("prefill"):
            out[cur]["prefill_s"] = float(re.search(r"ttft\s+([\d.]+)s", line).group(1))
        elif cur and (m := re.search(r"temp (\d+) C, fan (\d+) %", line)):
            out[cur].update(temp=int(m.group(1)), fan=int(m.group(2)))
    return out


def fig_power_cooling():
    drv = parse_sweep(RES / "power-sweep/driver-fan-curve/summary.log")
    cus = parse_sweep(RES / "power-sweep/custom-fan-curve/summary.log")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw={"wspace": 0.34})
    fig.subplots_adjust(left=0.035, right=0.925, top=0.71, bottom=0.15)
    header(fig, "250 W and a custom fan curve: 13 °C cooler, same decode speed",
           "RTX A6000 under sustained 4-stream load. Decode stayed at ~60 tok/s at every setting; only prefill feels the cap.")
    panels = [("temp", "GPU temperature", "°C, steady state", "{:g}°"), ("fan", "Fan speed", "% of max", "{:g}%"),
              ("prefill_s", "Prefill time", "seconds for a 38K-token prompt", "{:.1f}s")]
    for ax, (key, title, sub, fmt) in zip(axes, panels):
        ax.axvspan(240, 260, color=ACCENT_FAINT, zorder=0, linewidth=0)
        for data, color, name in [(drv, CONTRAST, "driver curve"), (cus, ACCENT, "custom curve")]:
            xs = sorted(data)
            ys = [data[x][key] for x in xs]
            ax.plot(xs, ys, color=color, zorder=3)
            ax.scatter(xs, ys, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
            if key == "prefill_s":  # lines converge at 300 W: label the 220 W end and offset the 250 W values
                up = color == CONTRAST
                ax.annotate(fmt.format(data[250][key]), (250, data[250][key]), xytext=(8, 9 if up else -11),
                            textcoords="offset points", ha="left", va="center", fontsize=9.5, color=color, fontweight="bold")
                ax.annotate(name, (xs[0], ys[0]), xytext=(9, 0), textcoords="offset points", va="center",
                            fontsize=9.5, color=color, fontweight="bold")
            else:
                ax.annotate(fmt.format(data[250][key]), (250, data[250][key]), xytext=(-9, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=9.5, color=color, fontweight="bold")
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(9, 0), textcoords="offset points", va="center",
                            fontsize=9.5, color=color, fontweight="bold")
        panel_title(ax, title, sub)
        ax.set_xticks([220, 250, 280, 300], ["220 W", "250 W", "280 W", "300 W"])
        ax.set_xlim(212, 304)
        ax.margins(y=0.2)
    axes[0].axhline(84, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    axes[0].text(213, 84.3, "driver target 84°", fontsize=8.5, color=MUTED)
    lo = axes[1].get_ylim()[0]
    axes[1].text(250, lo + 0.8, "deployed", ha="center", fontsize=9, color=ACCENT, fontweight="bold")
    footer(fig, "Data: results/power-sweep/ · 4 min of sustained load per setting · stock 300 W with the driver curve hit SW thermal slowdown at 87 °C")
    save(fig, "power-cooling.png")


# --------------------------------------------------------------------------- soak

def fig_soak():
    rows = [r for r in csv.DictReader(open(RES / "soak-10h/monitor.csv")) if r["temp_c"].strip().isdigit()]
    t0 = int(rows[0]["ts"])
    h = np.array([(int(r["ts"]) - t0) / 3600 for r in rows])
    temp = np.array([float(r["temp_c"]) for r in rows])
    power = np.array([float(r["power_w"]) for r in rows])
    p2_idle = np.array([r["pstate"] == "P2" and r["gpu_util"] == "0" for r in rows])
    reqs = [json.loads(l) for l in open(RES / "soak-10h/requests.jsonl")]
    events = [l.rstrip("\n").split(" ", 2) for l in open(RES / "soak-10h/events.log")]

    def ev_hours(word):
        return [(dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S").timestamp() - t0) / 3600 for d, t, m in events if word in m]

    full_load, long_pf = ev_hours("full load"), ev_hours("long prefill")
    bins = defaultdict(int)
    for r in reqs:
        if r["ok"]:
            bins[int((r["ts"] + r.get("total", 0) - t0) // 300)] += r.get("completion_tokens", 0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9.4), sharex=True, gridspec_kw={"hspace": 0.5, "height_ratios": [1, 1, 0.9]})
    fig.subplots_adjust(left=0.05, right=0.965, top=0.855, bottom=0.075)
    header(fig, "10 hours, 2,617 requests, 0 errors",
           "Mixed home-robot + coding-agent workload with hourly idle windows. VRAM held flat at 48.25 GB; host RAM grew 140 MB.")

    def shade(ax):
        for s in full_load:
            ax.axvspan(s, s + 0.5, color=ACCENT_FAINT, linewidth=0, zorder=0)
        for s in long_pf:
            ax.axvline(s + 0.02, color=ACCENT_SOFT, linewidth=1.4, zorder=1)

    ax = axes[0]
    shade(ax)
    ax.fill_between(h, 40, temp, color=ACCENT, alpha=0.10, linewidth=0)
    ax.plot(h, temp, color=ACCENT, linewidth=1.3)
    ax.set_ylim(40, 84)
    ax.axhline(80, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.text(10.05, 80, "pass\nlimit", fontsize=8, color=MUTED, va="center", linespacing=0.9)
    panel_title(ax, "GPU temperature", "°C · peak 74 °C at 75% fan, never thermally limited")
    ax.text(full_load[0] + 0.25, 77.5, "full load", ha="center", fontsize=8.5, color=ACCENT)
    ax.text(long_pf[1] + 0.06, 77.5, "150K-token prefill", fontsize=8.5, color=ACCENT)

    ax = axes[1]
    shade(ax)
    ax.plot(h, power, color=ACCENT, linewidth=1.1)
    ax.plot(h, np.where(p2_idle, power, np.nan), color=HIGHLIGHT, linewidth=2.6)
    ax.set_ylim(0, 275)
    panel_title(ax, "Board power", "W · cap 250 W · gold: idle windows where the GPU stayed in P2 (~73 W) instead of dropping to P8 (~19 W)")

    ax = axes[2]
    shade(ax)
    bx = sorted(bins)
    ax.bar([(b + 0.5) * 300 / 3600 for b in bx], [bins[b] / 300 for b in bx], width=300 / 3600 * 0.78, color=ACCENT, linewidth=0)
    panel_title(ax, "Generated tokens per second", "5-minute averages across all requests")
    ax.set_xticks(range(0, 11), [f"{i} h" for i in range(11)])
    ax.set_xlim(-0.1, 10.35)
    footer(fig, "Data: results/soak-10h/ · each hour: coding agents with tool calls, robot chat incl. images, a 6-request burst, a temperature-0 probe, then 20 min idle")
    save(fig, "soak-10h.png")


# --------------------------------------------------------------------------- FP8 recipe

def fig_recipe():
    d = json.load(open(RES / "quantization/recipe-probe.json"))
    recs = sorted(d["recipes"], key=lambda r: r["match"][0])
    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.subplots_adjust(left=0.29, right=0.93, top=0.76, bottom=0.14)
    header(fig, "Reproducing Qwen's FP8 recipe, byte for byte",
           "Share of FP8 codes identical to the official checkpoint when re-quantizing an unmodified tensor (layer 0 down_proj)")
    labels = {"A": "BF16-rounded scale, divide", "B": "FP32 scale amax/448, divide", "C": "FP32 multiply by 448/amax",
              "D": "BF16 scale, multiply by reciprocal", "E": "All-BF16 arithmetic", "F": "BF16 scale rounded up",
              "final": "FP32 scale amax × (1/448), divide"}
    for y, r in enumerate(recs):
        v = 100 * r["match"][0]
        hi = r["id"] == "final"
        ax.hlines(y, 96.8, v, color=HIGHLIGHT_SOFT if hi else CELL, linewidth=3, zorder=1)
        ax.scatter([v], [y], s=130 if hi else 70, color=HIGHLIGHT if hi else NEUTRAL, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(v + 0.07, y, f"{v:.2f}%", va="center", fontsize=10, color=HIGHLIGHT if hi else INK2, fontweight="bold" if hi else "normal")
    ax.set_yticks(range(len(recs)), [labels[r["id"]] for r in recs], fontsize=10)
    for t, r in zip(ax.get_yticklabels(), recs):
        t.set_color(HIGHLIGHT if r["id"] == "final" else INK2)
        t.set_fontweight("bold" if r["id"] == "final" else "normal")
    ax.set_xlim(96.8, 100.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    footer(fig, "Data: results/quantization/recipe-probe.json · the 'consistent' choice (round the scale to BF16 first) changes 1.6% of codes")
    save(fig, "fp8-recipe-match.png")


# --------------------------------------------------------------------------- abliteration map

def fig_abliteration_map():
    changed = set()
    for line in open(RES / "quantization/compare-huihui-vs-official-fp8.log"):
        m = re.match(r"layer(\d+)\s+identical\s+(\d+)/(\d+)", line)
        if m and m.group(2) != m.group(3):
            changed.add(int(m.group(1)))
    n = 64
    rows = [("Attention / DeltaNet input projections", "read", lambda i: False),
            ("Attention / DeltaNet output projection", "write", lambda i: i in changed),
            ("MLP gate and up projections", "read", lambda i: False),
            ("MLP down projection", "write", lambda i: i in changed),
            ("Norms, gates, conv", "-", lambda i: False)]

    fig, ax = plt.subplots(figsize=(13, 4.3))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.73, bottom=0.2)
    header(fig, "Where the abliteration lives: 70 of 1,606 tensors",
           "Abliterated FP8 vs official FP8, every tensor compared. Only residual-stream writers in layers 17-51 differ; the rest is byte-identical.")
    gap = 0.16
    for r, (_, role, fn) in enumerate(rows):
        for i in range(n):
            face = HIGHLIGHT if fn(i) else (CELL_FAINT if role == "-" else CELL)
            ax.add_patch(FancyBboxPatch((i + gap / 2, -r - 1 + gap / 2), 1 - gap, 1 - gap,
                                        boxstyle="round,pad=0,rounding_size=0.18", facecolor=face, linewidth=0))
    for i in range(n):
        if i % 4 == 3:  # every 4th layer is full attention, the rest Gated DeltaNet
            ax.plot(i + 0.5, 0.4, marker="v", markersize=4.5, color=MUTED, clip_on=False)
    ax.text(-0.3, 0.4, "full-attention layers", fontsize=8.5, color=MUTED, ha="right", va="center")
    ax.set_xlim(0, n)
    ax.set_ylim(-len(rows) - 0.1, 0.75)
    ax.set_yticks([-r - 0.5 for r in range(len(rows))], [name for name, _, _ in rows], fontsize=10)
    for t, (_, role, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(INK if role == "write" else MUTED)
        t.set_fontweight("bold" if role == "write" else "normal")
    ax.set_xticks([0.5, 17.5, 51.5, 63.5], ["layer 0", "17", "51", "63"])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    footer(fig, "Data: results/quantization/compare-huihui-vs-official-fp8.log · bold rows write into the residual stream "
                "(DeltaNet out_proj ×26, attention o_proj ×9, MLP down_proj ×35)")
    save(fig, "abliteration-map.png")


# --------------------------------------------------------------------------- abliteration SVD

def fig_abliteration_svd():
    svd = json.load(open(RES / "abliteration/svd.json"))
    rec = {r["tensor"]: r for r in json.load(open(RES / "abliteration/reconstruct.json"))["tensors"]}
    ts = svd["tensors"]
    cos = np.array(svd["abs_cos"])
    min_cos = cos[~np.eye(len(cos), dtype=bool)].min()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.15, 1.15]})
    fig.subplots_adjust(left=0.045, right=0.985, top=0.70, bottom=0.19)
    header(fig, "One direction, one scale: every edit is ΔW = -1.3 · r rᵀW",
           f"All 70 edited tensors (35 layers × residual writers) share a single unit vector r: pairwise |cos| of at least {min_cos:.5f}.")

    ax = axes[0]
    k = np.arange(1, 9)
    for t in ts:
        s = np.array(t["singular_values"])
        ax.plot(k, s / s[0], color=ACCENT, alpha=0.18, linewidth=1.2)
    ax.scatter([1], [1], s=60, color=ACCENT, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 1.6)
    ax.set_xticks(k, [f"σ{i}" for i in k])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    worst = max(t["singular_values"][1] / t["singular_values"][0] for t in ts)
    ax.text(2.2, worst * 1.6, f"σ2/σ1 at most {worst:.3f}", fontsize=9.5, color=ACCENT, fontweight="bold")
    panel_title(ax, "Rank 1", "singular values of ΔW relative to σ1, all 70 tensors")

    ax = axes[1]
    kinds = {"mlp.down_proj": ("MLP down_proj", "o"), "linear_attn.out_proj": ("DeltaNet out_proj", "s"), "self_attn.o_proj": ("attention o_proj", "D")}
    for kind, (label, marker) in kinds.items():
        xs = [t["layer"] for t in ts if t["kind"] == kind]
        ys = [t["alpha"] for t in ts if t["kind"] == kind]
        ax.scatter(xs, ys, s=30, marker=marker, color=HIGHLIGHT, edgecolor=SURFACE, linewidth=1, zorder=3, label=label)
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.text(17, 1.015, "α = 1: pure projection (remove r)", fontsize=8.5, color=MUTED)
    ax.text(17, 1.325, "α = 1.30 in every tensor: r is removed and pushed 30% past zero", fontsize=8.5, color=HIGHLIGHT, fontweight="bold")
    ax.set_ylim(0.9, 1.4)
    ax.set_xlim(15, 53)
    ax.set_xlabel("layer")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8.5, handletextpad=0.2, borderaxespad=0.2, labelcolor=INK2)
    panel_title(ax, "Same scale everywhere", "least-squares fit of α in ΔW = -α · r rᵀW, per tensor")

    ax = axes[2]
    layers = np.array([t["layer"] for t in ts])
    b13 = np.array([100 * rec[t["tensor"]]["alpha_1.3"]["bit_exact"] for t in ts])
    b10 = np.array([100 * rec[t["tensor"]]["alpha_1"]["bit_exact"] for t in ts])
    ax.scatter(layers, b10, s=22, color=NEUTRAL, edgecolor=SURFACE, linewidth=0.8, zorder=2)
    ax.scatter(layers, b13, s=26, color=ACCENT, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.text(52.5, b13.mean(), f"α = 1.3\n{b13.mean():.1f}%", fontsize=9.5, color=ACCENT, fontweight="bold", va="center")
    ax.text(52.5, b10.mean(), f"α = 1.0\n{b10.mean():.1f}%", fontsize=9.5, color=MUTED, va="center")
    ax.set_ylim(0, 100)
    ax.set_xlim(15, 58)
    ax.set_xlabel("layer")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}%"))
    panel_title(ax, "Rebuilt from one r", "BF16 elements bit-identical to the published weights")
    footer(fig, "Data: results/abliteration/ · ΔW = W_abliterated - W_official (BF16) · the remaining ~8% of bit flips come from estimating r from rounded weights, not from the formula")
    save(fig, "abliteration-svd.png")


# --------------------------------------------------------------------------- two GPUs: agent capacity

def agent_rows(path):
    """Per-level summary of an agent_bench.py requests.jsonl (same definitions as agent_report.py)."""
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["ok"]:
            by[r["level"]].append(r)
    out = {}
    for n, rs in sorted(by.items()):
        prompt = sum(r.get("prompt_tokens", 0) for r in rs)
        ttft = sorted(r["ttft"] for r in rs)
        out[n] = {"turns": len(rs) / n / 5, "ttft90": ttft[min(len(ttft) - 1, int(0.9 * len(ttft)))],
                  "cache": 100 * sum(r.get("cached_tokens", 0) for r in rs) / prompt}
    return out


def fig_agent_capacity():
    d = RES / "dual-gpu"
    series = [("1 GPU", d / "m1/single/requests.jsonl", CAT[0], "-"),
              ("DP=2, tuned router", d / "m3/dp2-tuned/requests.jsonl", CAT[1], "-"),
              ("TP=2", d / "m1/tp2/requests.jsonl", CAT[2], "-"),
              ("PP=2 (no MTP)", d / "m1/pp2/requests.jsonl", CAT[3], "-"),
              ("DP=2, default router", d / "m2/dp2/requests.jsonl", CAT[4], (0, (2, 2)))]
    data = [(name, agent_rows(path), color, ls) for name, path, color, ls in series]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.045, right=0.985, top=0.69, bottom=0.15)
    header(fig, "Coding agents are prefill-bound: capacity follows the KV pool and the router",
           "Closed-loop coding agents (shared 9K prompt, ~3K new tokens per turn, contexts 40-80K) on one or two RTX A6000s.")
    panels = [("turns", "Agent progress", "turns per agent per minute", False),
              ("ttft90", "Time to first token", "p90 seconds, log scale", True),
              ("cache", "Prefix-cache hit", "% of prompt tokens served from cache", False)]
    for ax, (key, title, sub, log) in zip(axes, panels):
        for name, rows, color, ls in data:
            xs = sorted(rows)
            ys = [rows[x][key] for x in xs]
            ax.plot(xs, ys, color=color, linestyle=ls, zorder=3, label=name)
            ax.scatter(xs, ys, s=38, color=color, edgecolor=SURFACE, linewidth=1.8, zorder=4)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        if key == "cache":
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}%"))
        ax.set_xticks([1, 2, 4, 8, 12, 16])
        ax.set_xlabel("concurrent agents")
        panel_title(ax, title, sub)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.008, 0.865), ncol=5,
               fontsize=9.5, handlelength=2.2, columnspacing=1.8, labelcolor=INK2)
    footer(fig, "Data: results/dual-gpu/ · Qwen3.8-27B-FP8, SGLang v0.5.21, FP8 KV · 5 min scored per level after 1 min warm-up · "
                "KV pool: 1 GPU 250K, DP=2 2x250K, TP=2 821K, PP=2 974K tokens")
    save(fig, "agent-capacity.png")


if __name__ == "__main__":
    fig_power_cooling()
    fig_soak()
    fig_recipe()
    fig_abliteration_map()
    fig_abliteration_svd()
    fig_agent_capacity()
