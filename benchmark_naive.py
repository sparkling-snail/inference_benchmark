"""
Run this to get your Phase 1 baseline numbers.

Usage:
    python benchmark_naive.py
    python benchmark_naive.py --num-requests 16 --max-new-tokens 50

These are the numbers Phase 3 (batched KV cache) and Phase 4
(continuous batching scheduler) need to beat. Results are written as
JSON to --output so Phase 5's comparison plot can load them directly
instead of hand-copying terminal output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch

from src.model_wrapper import ModelWrapper
from src.request import Request
from src.scheduler_naive import run_naive_batch

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a small village,",
    "The best way to learn a new programming language is",
    "In the year 2050, cities will",
    "My favorite recipe for a quick dinner is",
    "The most important lesson I learned from my first job was",
    "Climate change is affecting the way we",
    "The history of the internet begins with",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 naive static-batching baseline benchmark."
    )
    parser.add_argument(
        "--num-requests", type=int, default=len(PROMPTS),
        help="Requests to benchmark. Cycles through PROMPTS if more than are available.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=30,
        help="Tokens to generate per request.",
    )
    parser.add_argument(
        "--warmup-requests", type=int, default=1,
        help="Requests run and discarded before timing, to absorb one-time "
             "model/CUDA warm-up cost that would otherwise skew throughput.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Torch random seed, for reproducibility."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/phase1_naive.json"),
        help="Where to write the JSON results report.",
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. `pct` in [0, 100]."""
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[idx]


def build_requests(count: int, max_new_tokens: int) -> list[Request]:
    prompts = itertools.islice(itertools.cycle(PROMPTS), count)
    return [Request(prompt=p, max_new_tokens=max_new_tokens) for p in prompts]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("Loading model...")
    model = ModelWrapper()
    print(f"Device: {model.device}")

    if args.warmup_requests > 0:
        print(f"Warming up ({args.warmup_requests} request(s), untimed)...")
        run_naive_batch(model, build_requests(args.warmup_requests, args.max_new_tokens))

    requests = build_requests(args.num_requests, args.max_new_tokens)
    print(f"\nRunning naive static batch of {len(requests)} requests...")
    start = time.perf_counter()
    finished = run_naive_batch(model, requests)
    wall_time = time.perf_counter() - start

    total_tokens = sum(len(r.generated_token_ids) for r in finished)
    throughput = total_tokens / wall_time
    latencies = [r.total_latency for r in finished if r.total_latency is not None]
    ttfts = [r.ttft for r in finished if r.ttft is not None]

    report = {
        "phase": "phase1_naive_static_batching",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "device": model.device,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "config": {
            "num_requests": len(finished),
            "max_new_tokens": args.max_new_tokens,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
        },
        "results": {
            "total_tokens_generated": total_tokens,
            "wall_time_sec": wall_time,
            "throughput_tokens_per_sec": throughput,
            "latency_sec": {
                "mean": mean(latencies),
                "p50": _percentile(latencies, 50),
                "p90": _percentile(latencies, 90),
                "p99": _percentile(latencies, 99),
            },
            "ttft_sec": {
                "mean": mean(ttfts),
                "p50": _percentile(ttfts, 50),
                "p90": _percentile(ttfts, 90),
                "p99": _percentile(ttfts, 99),
            },
        },
    }

    lat, ttft = report["results"]["latency_sec"], report["results"]["ttft_sec"]
    print(f"\n{'='*60}")
    print("PHASE 1 BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Requests:             {report['config']['num_requests']}")
    print(f"Total tokens gen:     {total_tokens}")
    print(f"Wall time:            {wall_time:.3f}s")
    print(f"Throughput:           {throughput:.2f} tokens/sec")
    print(f"Latency  p50/p90/p99: {lat['p50']:.3f}s / {lat['p90']:.3f}s / {lat['p99']:.3f}s")
    print(f"TTFT     p50/p90/p99: {ttft['p50']:.3f}s / {ttft['p90']:.3f}s / {ttft['p99']:.3f}s")
    print(f"{'='*60}\n")

    print("Sample outputs:")
    for req in finished[:3]:
        text = model.decode(req.prompt_token_ids + req.generated_token_ids)
        print(f"\n  [{req.id}] {text}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
