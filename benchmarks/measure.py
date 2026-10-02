from __future__ import annotations

from statistics import mean

from .schema import BenchmarkSummary, RequestMetrics


def summarize_requests(requests: list[RequestMetrics], wall_time_sec: float) -> BenchmarkSummary:
    successes = [req for req in requests if req.success]
    latencies = [req.total_latency_sec for req in successes]
    ttfts = [req.ttft_sec for req in successes if req.ttft_sec is not None]
    tpots = [req.tpot_sec for req in successes if req.tpot_sec is not None]
    total_output_tokens = sum(req.output_tokens for req in successes)

    return BenchmarkSummary(
        request_count=len(requests),
        success_count=len(successes),
        error_count=len(requests) - len(successes),
        throughput_tokens_per_sec=(total_output_tokens / wall_time_sec) if wall_time_sec else 0.0,
        latency_sec=_stats(latencies),
        ttft_sec=_stats(ttfts),
        tpot_sec=_stats(tpots),
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[idx]
