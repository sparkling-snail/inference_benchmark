"""
Shared helpers for the recipe pipeline and the tail-latency experiments:
percentiles, Poisson arrivals, synthetic prompts, and result files.

Every result file records the git commit, the exact config, and raw
per-request numbers (not just summaries), so any chart can be
regenerated -- or re-cut differently -- without rerunning anything.
"""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

FILLER_WORDS = (
    "the quick brown fox jumps over the lazy dog while a large language model "
    "decodes one token at a time and every token waits for the one before it"
).split()


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (same convention as numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    """p50/p90/p99/max/mean of a list of seconds, plus the p99/p50 ratio.

    The ratio is the single most useful "is there a tail problem" number:
    ~1-2x is healthy, 5x+ means something structural is going on.
    """
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None, "p99_over_p50": None}
    p50, p99 = percentile(values, 50), percentile(values, 99)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": p50,
        "p90": percentile(values, 90),
        "p99": p99,
        "max": max(values),
        "p99_over_p50": (p99 / p50) if p50 else None,
    }


def poisson_offsets(n: int, rate_per_s: float, seed: int) -> list[float]:
    """Open-loop arrival times (seconds from t0) for a Poisson process.

    Open-loop matters: a closed-loop benchmark (N clients that each wait
    for a reply before sending again) slows its own arrivals down when the
    server slows down, which hides exactly the queueing tail we want to see.
    """
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate_per_s)
        out.append(t)
    return out


def make_prompt(approx_words: int, seed: int) -> str:
    """Deterministic filler prompt of roughly `approx_words` words.

    Seeded per request so prompts differ (no accidental prefix-cache hits
    between requests) but the whole workload is reproducible.
    """
    rng = random.Random(seed)
    words = [f"[{seed}]"] + [rng.choice(FILLER_WORDS) for _ in range(max(1, approx_words - 1))]
    return " ".join(words)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def save_result(path: str | Path, experiment: str, config: dict[str, Any], data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "timestamp_unix": time.time(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "config": config,
        **data,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def fmt_ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:8.1f}"
