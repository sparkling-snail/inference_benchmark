"""
Experiment 2 -- the long-prompt stall (prefill/decode interference).

Question: what happens to users who are mid-stream when someone else
submits a very long prompt?

A set of background streams (short prompt, long output) decode steadily.
A few seconds in, one or more very long prompts arrive. Their prefill
competes with everyone else's decode steps for the same GPU iterations;
without chunking, a single big prefill can freeze every other stream for
as long as it takes to process. Records every background token's
timestamp so the stall shows up as a spike in the ITL timeline.

Run it once per server configuration and give each run a --label, e.g.:
  - vLLM, small token budget per step:  --max-num-batched-tokens 512
  - vLLM, budget large enough to take the whole long prompt in one step:
    --max-num-batched-tokens 16384
(vLLM V1 always chunks prefill; the per-step token budget is the knob
that decides how much prefill can land in one iteration.)

Usage:
    python -m experiments.tail.exp2_prefill_stall --label budget-512 \\
        --out results/tail/exp2_budget-512.json
"""

from __future__ import annotations

import argparse
import asyncio

from benchmarks.loadgen import RequestSpec, resolve_model, run_open_loop
from benchmarks.stats import fmt_ms, make_prompt, save_result, summarize


async def main_async(args) -> None:
    model = await resolve_model(args.base_url, args.model)
    print(f"Server {args.base_url}, model {model}, label {args.label!r}")
    await run_open_loop(args.base_url, model, [RequestSpec(0, make_prompt(50, 999), 32, "warmup")] * 4)

    specs = [
        RequestSpec(offset_s=i * 0.05, prompt=make_prompt(args.background_prompt_words, seed=i), max_tokens=args.background_tokens, tag="background")
        for i in range(args.background_streams)
    ]
    specs += [
        RequestSpec(offset_s=args.inject_at_s + j * args.inject_gap_s, prompt=make_prompt(args.long_prompt_words, seed=10_000 + j), max_tokens=args.long_prompt_output_tokens, tag="long_prompt")
        for j in range(args.long_prompts)
    ]
    results, wall = await run_open_loop(args.base_url, model, specs)

    bg = [r for r in results if r.tag == "background" and r.ok]
    inj = [r for r in results if r.tag == "long_prompt"]
    window_start = args.inject_at_s
    window_end = max((r.e2e_s or 0) + (r.sent_at_s or 0) for r in inj) + 0.5 if inj else window_start

    # every background inter-token gap, stamped with when it ended
    gaps = [(t1, t1 - t0) for r in bg for t0, t1 in zip(r.chunk_times_s, r.chunk_times_s[1:])]
    during = [g for t, g in gaps if window_start <= t <= window_end]
    before = [g for t, g in gaps if t < window_start]
    summary = {
        "itl_before_injection": summarize(before),
        "itl_during_injection": summarize(during),
        "worst_gap_s": max((g for _, g in gaps), default=None),
        "long_prompt_ttft_s": [r.ttft_s for r in inj],
        "long_prompt_tokens": [r.prompt_tokens for r in inj],
    }
    print(f"Background ITL before injection: p50 {fmt_ms(summary['itl_before_injection']['p50'])} p99 {fmt_ms(summary['itl_before_injection']['p99'])} ms")
    print(f"Background ITL during injection: p50 {fmt_ms(summary['itl_during_injection']['p50'])} p99 {fmt_ms(summary['itl_during_injection']['p99'])} ms")
    print(f"Worst single gap: {fmt_ms(summary['worst_gap_s'])} ms | long prompt TTFT: {[fmt_ms(x).strip() for x in summary['long_prompt_ttft_s']]} ms")

    path = save_result(
        args.out,
        "exp2_prefill_stall",
        {**vars(args), "model": model},
        {"label": args.label, "summary": summary, "gaps": gaps, "requests": [r.to_dict() for r in results]},
    )
    print(f"Saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", required=True, help="server config name, e.g. budget-512")
    ap.add_argument("--background-streams", type=int, default=16)
    ap.add_argument("--background-prompt-words", type=int, default=40)
    ap.add_argument("--background-tokens", type=int, default=600)
    ap.add_argument("--inject-at-s", type=float, default=4.0)
    ap.add_argument("--long-prompts", type=int, default=3)
    ap.add_argument("--inject-gap-s", type=float, default=2.0)
    ap.add_argument("--long-prompt-words", type=int, default=6000, help="~8k tokens for most tokenizers")
    ap.add_argument("--long-prompt-output-tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/tail/exp2_{args.label}.json"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
