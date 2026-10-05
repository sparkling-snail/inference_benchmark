"""
Experiment 3 -- KV-cache pressure and preemption.

Question: what happens to the tail when the KV cache is too small for
the work in flight?

Runs the same burst of long-output requests against servers configured
with progressively less KV-cache memory. When the cache fills, vLLM
preempts running requests (frees their cache and recomputes them later).
Every preempted request pays its prefill again and loses its place --
so p99 end-to-end latency jumps while throughput can still look fine.

Samples /metrics during the run for the preemption counter and KV-cache
usage. Run once per server config with a --label, e.g. vLLM started with
  --num-gpu-blocks-override 4096 / 2048 / 1024 / 512   (or lower
  --gpu-memory-utilization), then plot them together.

Usage:
    python -m experiments.tail.exp3_kv_pressure --label blocks-1024 --kv-blocks 1024
"""

from __future__ import annotations

import argparse
import asyncio

from .client import RequestSpec, pick, resolve_model, run_open_loop, scrape_metrics
from .common import fmt_ms, make_prompt, poisson_offsets, save_result, summarize


async def main_async(args) -> None:
    model = await resolve_model(args.base_url, args.model)
    print(f"Server {args.base_url}, model {model}, label {args.label!r}")
    await run_open_loop(args.base_url, model, [RequestSpec(0, make_prompt(50, 999), 32, "warmup")] * 4)

    specs = [
        RequestSpec(offset_s=off, prompt=make_prompt(args.prompt_words, seed=i), max_tokens=args.output_tokens, tag="load")
        for i, off in enumerate(poisson_offsets(args.n, args.rate, args.seed))
    ]

    samples: list[dict] = []

    async def sample(elapsed: float) -> None:
        m = await scrape_metrics(args.base_url)
        if m:
            samples.append(
                {
                    "t_s": elapsed,
                    "kv_cache_usage": pick(m, "cache_usage"),
                    "running": pick(m, "num_requests_running"),
                    "waiting": pick(m, "num_requests_waiting"),
                    "preemptions_total": pick(m, "preempt"),
                }
            )

    before = await scrape_metrics(args.base_url)
    results, wall = await run_open_loop(args.base_url, model, specs, on_tick=sample)
    after = await scrape_metrics(args.base_url)

    ok = [r for r in results if r.ok]
    preemptions = None
    if pick(after, "preempt") is not None:
        preemptions = (pick(after, "preempt") or 0) - (pick(before, "preempt") or 0)
    out_tokens = sum((r.completion_tokens or len(r.chunk_times_s)) for r in ok)
    summary = {
        "preemptions": preemptions,
        "max_kv_cache_usage": max((s["kv_cache_usage"] for s in samples if s["kv_cache_usage"] is not None), default=None),
        "output_tokens_per_s": out_tokens / wall if wall else None,
        "errors": len(results) - len(ok),
        "ttft": summarize([r.ttft_s for r in ok if r.ttft_s is not None]),
        "itl": summarize([x for r in ok for x in r.itls_s]),
        "e2e": summarize([r.e2e_s for r in ok if r.e2e_s is not None]),
    }
    print(
        f"preemptions {preemptions} | peak KV usage {summary['max_kv_cache_usage']} | "
        f"E2E p50 {fmt_ms(summary['e2e']['p50'])} p99 {fmt_ms(summary['e2e']['p99'])} ms | "
        f"{summary['output_tokens_per_s'] or 0:.0f} tok/s"
    )
    path = save_result(
        args.out,
        "exp3_kv_pressure",
        {**vars(args), "model": model},
        {"label": args.label, "kv_blocks": args.kv_blocks, "summary": summary, "samples": samples,
         "requests": [r.to_dict(keep_chunk_times=False) for r in results]},
    )
    print(f"Saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", required=True, help="server config name, e.g. blocks-1024")
    ap.add_argument("--kv-blocks", type=int, default=None, help="KV blocks the server was given (for the x-axis)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rate", type=float, default=20.0, help="req/s; high on purpose -- this is a burst")
    ap.add_argument("--prompt-words", type=int, default=400)
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/tail/exp3_{args.label}.json"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
