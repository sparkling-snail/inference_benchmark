"""
Charts for the tail-latency post, from the JSON files in results/tail/.

    python -m experiments.tail.plot                      # light theme
    python -m experiments.tail.plot --theme dark         # for a dark site
    python -m experiments.tail.plot --results results/tail --out results/tail/figs

Makes whichever figures it has data for:
  exp1_load_cliff.png    p50 vs p99 TTFT and ITL as offered load rises
  exp2_prefill_stall.png background streams' inter-token gaps over time, one panel per config
  exp3_kv_pressure.png   p99 E2E and preemptions as KV-cache capacity shrinks
  exp4_hol_blocking.png  p99 TTFT for short vs long requests under each admission policy
  exp4_ttft_cdf.png      distribution of short requests' TTFT per policy (where the tail lives)

Style: one hue per series in a fixed order, 2px lines, light grid, one
y-axis per panel (different units/scales get their own panel).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3de",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33322f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}


def style(theme: str) -> dict:
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["muted"], "axes.edgecolor": t["grid"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
    })
    return t


def load(pattern: str) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]


def ms(x):
    return None if x is None else x * 1000


def plot_exp1(runs, t, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "ttft", "Time to first token"), (axes[1], "itl", "Inter-token latency")):
        for run in runs:
            pts = run["points"]
            x = [p["offered_rate"] for p in pts]
            for pct, color, ls in (("p50", t["series"][0], "-"), ("p99", t["series"][1], "-")):
                label = f"{pct}" + (f" ({run['label']})" if len(runs) > 1 else "")
                ax.plot(x, [ms(p[metric][pct]) for p in pts], ls, marker="o", color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("offered load (requests/s)")
        ax.set_ylabel("ms")
        ax.legend()
    fig.suptitle("p50 stays flat; p99 bends first", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "exp1_load_cliff.png", dpi=160)
    plt.close(fig)


def plot_exp2(runs, t, out):
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.8 * len(runs)), sharex=True, sharey=True, squeeze=False)
    for ax, run, color in zip(axes[:, 0], runs, t["series"]):
        gaps = run["gaps"]
        ax.scatter([g[0] for g in gaps], [ms(g[1]) for g in gaps], s=6, color=color, alpha=0.7, linewidths=0)
        for r in run["requests"]:
            if r["tag"] == "long_prompt" and r["sent_at_s"] is not None:
                ax.axvline(r["sent_at_s"], color=t["muted"], lw=1, ls="--")
        s = run["summary"]
        ax.set_title(f"{run['label']}  -  worst gap {ms(s['worst_gap_s']):.0f} ms, long-prompt TTFT {ms(max(x for x in s['long_prompt_ttft_s'] if x is not None)):.0f} ms")
        ax.set_ylabel("gap between tokens (ms)")
    axes[-1, 0].set_xlabel("time (s)   dashed lines = a long prompt arrives")
    fig.tight_layout()
    fig.savefig(out / "exp2_prefill_stall.png", dpi=160)
    plt.close(fig)


def plot_exp3(runs, t, out):
    runs = sorted(runs, key=lambda r: (r.get("kv_blocks") or 0), reverse=True)
    labels = [r["label"] for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(runs))
    axes[0].plot(x, [ms(r["summary"]["e2e"]["p50"]) for r in runs], marker="o", color=t["series"][0], label="p50")
    axes[0].plot(x, [ms(r["summary"]["e2e"]["p99"]) for r in runs], marker="o", color=t["series"][1], label="p99")
    axes[0].set_title("End-to-end latency")
    axes[0].set_ylabel("ms")
    axes[0].legend()
    axes[1].bar(x, [r["summary"]["preemptions"] or 0 for r in runs], color=t["series"][0], width=0.6)
    axes[1].grid(axis="x", visible=False)
    axes[1].set_title("Preemptions during the run")
    axes[1].set_ylabel("count")
    for ax in axes:
        ax.set_xticks(list(x), labels)
        ax.set_xlabel("KV-cache capacity (largest to smallest)")
    fig.tight_layout()
    fig.savefig(out / "exp3_kv_pressure.png", dpi=160)
    plt.close(fig)


def plot_exp4(run, t, out):
    policies = list(run["results"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, cls in zip(axes, ("short", "long")):
        vals = [ms(run["results"][p]["summary"][cls]["ttft"]["p99"]) for p in policies]
        bars = ax.bar(policies, vals, color=t["series"][: len(policies)], width=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9,
                        color=t["text"], xytext=(0, 3), textcoords="offset points")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"p99 TTFT, {cls} requests")
        ax.set_ylabel("ms")
    fig.suptitle("SJF moves the tail; it doesn't delete it", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "exp4_hol_blocking.png", dpi=160)
    plt.close(fig)


def plot_exp4_cdf(run, t, out):
    fig, ax = plt.subplots(figsize=(11, 4))
    for (policy, res), color in zip(run["results"].items(), t["series"]):
        xs = sorted(r["ttft_s"] * 1000 for r in res["requests"] if r["cls"] == "short" and r["ttft_s"] is not None)
        ys = [(i + 1) / len(xs) * 100 for i in range(len(xs))]
        ax.step(xs, ys, where="post", color=color, label=policy)
    ax.set_xscale("log")
    ax.axhline(99, color=t["muted"], lw=1, ls="--")
    ax.annotate("p99", (ax.get_xlim()[0], 99), xytext=(4, -12), textcoords="offset points", color=t["muted"], fontsize=9)
    ax.set_xlabel("time to first token, ms (log scale)")
    ax.set_ylabel("% of short requests")
    ax.set_title("Short requests: same median, very different tail")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "exp4_ttft_cdf.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/tail")
    ap.add_argument("--out", default=None)
    ap.add_argument("--theme", choices=list(THEMES), default="light")
    args = ap.parse_args()
    out = Path(args.out or f"{args.results}/figs-{args.theme}")
    out.mkdir(parents=True, exist_ok=True)
    t = style(args.theme)
    made = []
    if runs := load(f"{args.results}/exp1_*.json"):
        plot_exp1(runs, t, out); made.append("exp1")
    if runs := load(f"{args.results}/exp2_*.json"):
        plot_exp2(runs, t, out); made.append("exp2")
    if runs := load(f"{args.results}/exp3_*.json"):
        plot_exp3(runs, t, out); made.append("exp3")
    main4 = Path(f"{args.results}/exp4_hol_blocking.json")
    runs = [json.loads(main4.read_text())] if main4.exists() else load(f"{args.results}/exp4_*.json")
    if runs:
        plot_exp4(runs[-1], t, out); plot_exp4_cdf(runs[-1], t, out); made.append("exp4")
    print(f"Wrote {', '.join(made) or 'nothing (no results found)'} to {out}/")


if __name__ == "__main__":
    main()
