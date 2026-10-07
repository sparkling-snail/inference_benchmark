"""
Experiment 4 -- head-of-line blocking in my own continuous-batching engine.

Question: when a few long requests are mixed into a stream of short
ones, how much of the short requests' tail latency is just *waiting
behind the long ones*? And how much does the admission policy (the one
line in scheduler.fill() that picks who gets a free slot) change that?

Setup: an open-loop Poisson stream at a fixed fraction of measured
capacity, ~10% long requests (big prompt, many output tokens) and ~90%
short ones. The exact same arrival trace is replayed against each
policy, so the only thing that changes between runs is the policy.

What to look for:
  - fcfs: short requests' p99 TTFT is dominated by queue wait -- they
    land behind long requests holding the batch slots.
  - sjf: short p99 drops sharply, but long requests' p99 gets worse
    (they get skipped) -- SJF moves the tail, it doesn't delete it.
  - sjf_aging: most of SJF's win for short requests, with the long
    requests' worst case bounded by max_wait_s.

Usage:
    python -m experiments.tail.exp4_hol_blocking                # gpt2, defaults
    python -m experiments.tail.exp4_hol_blocking --load 0.9 --n 120
    python -m experiments.tail.exp4_hol_blocking --model ./my-local-model
"""

from __future__ import annotations

import argparse
import time

from engine.model_wrapper import MODEL_NAME, ModelWrapper
from engine.request import Request
from engine.scheduler import POLICIES, make_sjf_aging, run_scheduler

from benchmarks.stats import fmt_ms, make_prompt, poisson_offsets, save_result, summarize


def build_workload(args) -> list[dict]:
    """The arrival trace + request shapes, shared by every policy run."""
    import random

    rng = random.Random(args.seed)
    offsets = poisson_offsets(args.n, args.rate, args.seed)
    # exactly round(n * long_frac) long requests at random positions, so
    # small runs don't end up with zero (or far too many) long requests
    long_idx = set(rng.sample(range(args.n), max(1, round(args.n * args.long_frac))))
    workload = []
    for i, offset in enumerate(offsets):
        is_long = i in long_idx
        workload.append(
            {
                "index": i,
                "cls": "long" if is_long else "short",
                "offset_s": offset,
                "prompt": make_prompt(args.long_prompt_words if is_long else args.short_prompt_words, seed=args.seed * 100_000 + i),
                "max_new_tokens": args.long_tokens if is_long else args.short_tokens,
            }
        )
    return workload


def calibrate(model: ModelWrapper, args) -> dict:
    """Measure decode step time at a full batch to estimate capacity.

    capacity (req/s) ~= tokens/s at full batch / mean output tokens per
    request. Ignores prefill cost, so it slightly overestimates capacity --
    which errs toward *less* load than requested, not more.
    """
    reqs = [Request(prompt=make_prompt(args.short_prompt_words, seed=900 + i), max_new_tokens=64) for i in range(args.max_batch_size)]
    t0 = time.perf_counter()
    _, stats = run_scheduler(model, reqs, max_batch_size=args.max_batch_size, ignore_eos=True)
    wall = time.perf_counter() - t0
    step_s = wall / max(1, len(stats.step_batch_sizes))
    mean_tokens = args.long_frac * args.long_tokens + (1 - args.long_frac) * args.short_tokens
    capacity = args.max_batch_size / (step_s * mean_tokens)
    return {"step_s_full_batch": step_s, "mean_output_tokens": mean_tokens, "capacity_req_per_s": capacity}


def run_policy(model: ModelWrapper, workload: list[dict], policy_name: str, args) -> list[dict]:
    policy = make_sjf_aging(args.max_wait_s) if policy_name == "sjf_aging" else POLICIES[policy_name]
    t0 = time.perf_counter() + 0.2
    reqs, cls_by_id, idx_by_id = [], {}, {}
    for w in workload:
        r = Request(prompt=w["prompt"], max_new_tokens=w["max_new_tokens"])
        r.arrival_time = t0 + w["offset_s"]
        reqs.append(r)
        cls_by_id[r.id], idx_by_id[r.id] = w["cls"], w["index"]

    finished, stats = run_scheduler(
        model, reqs, max_batch_size=args.max_batch_size, policy=policy, respect_arrivals=True, ignore_eos=True
    )
    rows = []
    for r in finished:
        rows.append(
            {
                "index": idx_by_id[r.id],
                "cls": cls_by_id[r.id],
                "max_new_tokens": r.max_new_tokens,
                "queue_wait_s": r.queue_wait,
                "ttft_s": r.ttft,
                "e2e_s": r.total_latency,
                "itl_p99_s": summarize(r.itls)["p99"],
                "itls_s": r.itls,
            }
        )
    rows.sort(key=lambda x: x["index"])
    print(f"  {policy_name}: done, mean batch occupancy {stats.mean_occupancy_pct:.0f}%")
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    out = {}
    for cls in ("short", "long", "all"):
        sel = [r for r in rows if cls == "all" or r["cls"] == cls]
        out[cls] = {
            metric: summarize([r[f"{metric}_s"] for r in sel if r[f"{metric}_s"] is not None])
            for metric in ("queue_wait", "ttft", "e2e")
        }
        out[cls]["itl"] = summarize([x for r in sel for x in r["itls_s"]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--n", type=int, default=80, help="requests per policy run")
    ap.add_argument("--long-frac", type=float, default=0.1)
    ap.add_argument("--short-tokens", type=int, default=16)
    ap.add_argument("--long-tokens", type=int, default=256)
    ap.add_argument("--short-prompt-words", type=int, default=20)
    ap.add_argument("--long-prompt-words", type=int, default=150)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--load", type=float, default=0.85, help="arrival rate as a fraction of measured capacity")
    ap.add_argument("--rate", type=float, default=None, help="fixed arrival rate (req/s); overrides --load")
    ap.add_argument("--max-wait-s", type=float, default=2.0, help="aging threshold for sjf_aging")
    ap.add_argument("--policies", nargs="+", default=["fcfs", "sjf", "sjf_aging"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tail/exp4_hol_blocking.json")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    model = ModelWrapper(args.model)
    print(f"Device: {model.device}")

    cal = calibrate(model, args)
    if args.rate is None:
        args.rate = args.load * cal["capacity_req_per_s"]
    print(
        f"Calibration: {cal['step_s_full_batch'] * 1000:.1f} ms/step at batch {args.max_batch_size} -> "
        f"capacity ~{cal['capacity_req_per_s']:.2f} req/s; running at {args.rate:.2f} req/s"
    )

    workload = build_workload(args)
    n_long = sum(w["cls"] == "long" for w in workload)
    print(f"Workload: {len(workload)} requests ({n_long} long), trace length {workload[-1]['offset_s']:.1f}s\n")

    run_policy(model, workload[: min(8, len(workload))], "fcfs", args)  # warm-up, discarded
    results = {}
    for name in args.policies:
        rows = run_policy(model, workload, name, args)
        results[name] = {"summary": summarize_rows(rows), "requests": rows}

    print(f"\n{'policy':10s} {'class':6s} {'p50 TTFT':>10s} {'p99 TTFT':>10s} {'p99 wait':>10s} {'p99 E2E':>10s}   (ms)")
    for name, res in results.items():
        for cls in ("short", "long"):
            s = res["summary"][cls]
            print(f"{name:10s} {cls:6s} {fmt_ms(s['ttft']['p50']):>10s} {fmt_ms(s['ttft']['p99']):>10s} "
                  f"{fmt_ms(s['queue_wait']['p99']):>10s} {fmt_ms(s['e2e']['p99']):>10s}")

    config = {k: v for k, v in vars(args).items()}
    path = save_result(args.out, "exp4_hol_blocking", config, {"calibration": cal, "device": model.device, "results": results})
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
