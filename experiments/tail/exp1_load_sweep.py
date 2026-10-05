"""
Experiment 1 -- the load cliff.

Question: how do p50 and p99 behave as offered load approaches capacity?

Sweeps the arrival rate of an open-loop Poisson stream and records TTFT,
per-token ITL and end-to-end latency at each rate. Expect p50 to stay
nearly flat while p99 TTFT bends upward well before the server looks
"full" -- queueing theory's hockey stick, and the reason capacity should
be planned at the p99 knee, not at peak throughput.

Usage (server already running):
    python -m experiments.tail.exp1_load_sweep --base-url http://localhost:8000 \\
        --rates 1 2 4 6 8 10 12 --duration-s 60
"""

from __future__ import annotations

import argparse
import asyncio
import random

from .client import RequestSpec, resolve_model, run_open_loop, scrape_metrics, pick
from .common import fmt_ms, make_prompt, poisson_offsets, save_result, summarize


def workload(rate: float, duration_s: float, args, seed: int) -> list[RequestSpec]:
    n = max(5, int(rate * duration_s))
    rng = random.Random(seed)
    specs = []
    for i, off in enumerate(poisson_offsets(n, rate, seed)):
        specs.append(
            RequestSpec(
                offset_s=off,
                prompt=make_prompt(rng.randint(args.min_prompt_words, args.max_prompt_words), seed=seed * 100_000 + i),
                max_tokens=rng.randint(args.min_output_tokens, args.max_output_tokens),
                tag="sweep",
            )
        )
    return specs


async def main_async(args) -> None:
    model = await resolve_model(args.base_url, args.model)
    print(f"Server {args.base_url}, model {model}")

    # warm-up: first requests pay for CUDA graph / compile / allocator warm-up
    await run_open_loop(args.base_url, model, workload(2, 5, args, seed=999))

    points = []
    for i, rate in enumerate(args.rates):
        before = await scrape_metrics(args.base_url)
        specs = workload(rate, args.duration_s, args, seed=args.seed + i)
        results, wall = await run_open_loop(args.base_url, model, specs)
        after = await scrape_metrics(args.base_url)

        ok = [r for r in results if r.ok]
        out_tokens = sum((r.completion_tokens or len(r.chunk_times_s)) for r in ok)
        preempt = None
        if pick(after, "preempt") is not None:
            preempt = (pick(after, "preempt") or 0) - (pick(before, "preempt") or 0)
        point = {
            "offered_rate": rate,
            "n": len(results),
            "errors": len(results) - len(ok),
            "achieved_rate": len(ok) / wall if wall else None,
            "output_tokens_per_s": out_tokens / wall if wall else None,
            "preemptions": preempt,
            "ttft": summarize([r.ttft_s for r in ok if r.ttft_s is not None]),
            "itl": summarize([x for r in ok for x in r.itls_s]),
            "e2e": summarize([r.e2e_s for r in ok if r.e2e_s is not None]),
            "requests": [r.to_dict(keep_chunk_times=False) for r in results],
        }
        points.append(point)
        print(
            f"rate {rate:6.2f} req/s | TTFT p50 {fmt_ms(point['ttft']['p50'])} p99 {fmt_ms(point['ttft']['p99'])} ms"
            f" | ITL p99 {fmt_ms(point['itl']['p99'])} ms | {point['output_tokens_per_s'] or 0:7.0f} tok/s"
            f" | errors {point['errors']}"
        )

    path = save_result(args.out, "exp1_load_sweep", {**vars(args), "model": model}, {"label": args.label, "points": points})
    print(f"Saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="defaults to the first model the server lists")
    ap.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4, 6, 8, 10, 12])
    ap.add_argument("--duration-s", type=float, default=60.0, help="trace length per rate")
    ap.add_argument("--min-prompt-words", type=int, default=50)
    ap.add_argument("--max-prompt-words", type=int, default=600)
    ap.add_argument("--min-output-tokens", type=int, default=64)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--label", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tail/exp1_load_sweep.json")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
