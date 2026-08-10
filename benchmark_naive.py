"""
Run this to get your Phase 1 baseline numbers.

Usage:
    python benchmark_naive.py

These are the numbers Phase 3 (batched KV cache) and Phase 4
(continuous batching scheduler) need to beat. Save the printed output --
you'll want it for the final comparison plot in Phase 5.
"""

import time

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


def main():
    print("Loading model...")
    model = ModelWrapper()
    print(f"Device: {model.device}")

    requests = [Request(prompt=p, max_new_tokens=30) for p in PROMPTS]

    print(f"\nRunning naive static batch of {len(requests)} requests...")
    start = time.perf_counter()
    finished = run_naive_batch(model, requests)
    wall_time = time.perf_counter() - start

    total_tokens = sum(len(r.generated_token_ids) for r in finished)
    throughput = total_tokens / wall_time

    print(f"\n{'='*60}")
    print("PHASE 1 BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Requests:          {len(finished)}")
    print(f"Total tokens gen:  {total_tokens}")
    print(f"Wall time:         {wall_time:.3f}s")
    print(f"Throughput:        {throughput:.2f} tokens/sec")
    print(f"Avg latency/req:   {wall_time / len(finished):.3f}s")
    print(f"{'='*60}\n")

    print("Sample outputs:")
    for req in finished[:3]:
        text = model.decode(req.prompt_token_ids + req.generated_token_ids)
        print(f"\n  [{req.id}] {text}")


if __name__ == "__main__":
    main()
