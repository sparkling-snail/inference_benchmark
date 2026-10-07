"""
SLO goodput: the highest request rate a deployment sustains while
p99 TTFT, p99 inter-token latency and error rate all stay inside the SLO.

This is the number a recipe is built on, rather than peak throughput:
peak throughput is measured with the queue overflowing and p99 already
blown, so it overstates what an endpoint can actually be sold at.

Each probe replays an open-loop Poisson trace at one offered rate (see
stats.poisson_offsets for why open-loop). The search doubles the rate
until a probe fails, then bisects between the last pass and the first
fail. Probes are expensive (tens of seconds each), so the search stops
at a relative tolerance rather than an exact edge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .loadgen import RequestSpec, run_open_loop, scrape_metrics, pick
from .matrix import Search, Slo, Workload
from .stats import make_prompt, poisson_offsets, summarize


@dataclass
class Probe:
    offered_rate: float
    passed: bool
    n: int = 0
    errors: int = 0
    achieved_rate: float | None = None
    output_tokens_per_s: float | None = None
    ttft: dict = field(default_factory=dict)
    itl: dict = field(default_factory=dict)
    e2e: dict = field(default_factory=dict)
    preemptions: float | None = None
    failed_on: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    best: Probe | None              # highest passing probe; None if even min_rate fails
    probes: list[Probe]
    bracket: tuple[float | None, float | None]  # (highest pass, lowest fail)


Measure = Callable[[float], Awaitable[Probe]]


async def search_goodput(measure: Measure, cfg: Search) -> SearchResult:
    probes: list[Probe] = []
    lo: Probe | None = None
    hi: float | None = None

    async def probe(rate: float) -> Probe:
        p = await measure(rate)
        probes.append(p)
        return p

    # 1. bracket: double up from start_rate until a failure (or max_rate),
    #    or halve down until a pass (or min_rate) if start_rate already fails
    rate = cfg.start_rate
    while len(probes) < cfg.max_probes:
        p = await probe(rate)
        if p.passed:
            lo = p
            if hi is not None or rate >= cfg.max_rate:
                break
            rate = min(rate * 2, cfg.max_rate)
        else:
            hi = rate
            if lo is not None or rate <= cfg.min_rate:
                break
            rate = max(rate / 2, cfg.min_rate)

    # 2. bisect (geometric midpoint: rates span orders of magnitude)
    while lo is not None and hi is not None and len(probes) < cfg.max_probes:
        if hi / lo.offered_rate <= 1 + cfg.rel_tol:
            break
        mid = (lo.offered_rate * hi) ** 0.5
        p = await probe(mid)
        if p.passed:
            lo = p
        else:
            hi = mid

    return SearchResult(best=lo, probes=probes, bracket=(lo.offered_rate if lo else None, hi))


def check_slo(p: Probe, slo: Slo) -> list[str]:
    failed = []
    if p.n == 0 or p.errors / p.n > slo.max_error_rate:
        failed.append("errors")
    if p.ttft.get("p99") is None or p.ttft["p99"] * 1000 > slo.ttft_p99_ms:
        failed.append("ttft_p99")
    if p.itl.get("p99") is not None and p.itl["p99"] * 1000 > slo.itl_p99_ms:
        failed.append("itl_p99")
    return failed


def build_trace(rate: float, wl: Workload, seed: int) -> list[RequestSpec]:
    n = max(10, int(rate * wl.duration_s))
    rng = random.Random(seed)
    return [
        RequestSpec(
            offset_s=off,
            prompt=make_prompt(rng.randint(*wl.prompt_words), seed=seed * 1_000_003 + i),
            max_tokens=rng.randint(*wl.output_tokens),
            tag="goodput",
        )
        for i, off in enumerate(poisson_offsets(n, rate, seed))
    ]


def http_measure(base_url: str, model: str, wl: Workload, slo: Slo, log: Callable[[Probe], None] | None = None) -> Measure:
    """A Measure that probes a live OpenAI-compatible server."""
    calls = 0

    async def measure(rate: float) -> Probe:
        nonlocal calls
        calls += 1
        before = await scrape_metrics(base_url)
        results, wall = await run_open_loop(base_url, model, build_trace(rate, wl, seed=wl.seed + calls))
        after = await scrape_metrics(base_url)
        ok = [r for r in results if r.ok]
        out_tokens = sum((r.completion_tokens or len(r.chunk_times_s)) for r in ok)
        p = Probe(
            offered_rate=rate,
            passed=False,
            n=len(results),
            errors=len(results) - len(ok),
            achieved_rate=len(ok) / wall if wall else None,
            output_tokens_per_s=out_tokens / wall if wall else None,
            ttft=summarize([r.ttft_s for r in ok if r.ttft_s is not None]),
            itl=summarize([x for r in ok for x in r.itls_s]),
            e2e=summarize([r.e2e_s for r in ok if r.e2e_s is not None]),
        )
        if pick(after, "preempt") is not None:
            p.preemptions = (pick(after, "preempt") or 0) - (pick(before, "preempt") or 0)
        p.failed_on = check_slo(p, slo)
        p.passed = not p.failed_on
        if log:
            log(p)
        return p

    return measure
