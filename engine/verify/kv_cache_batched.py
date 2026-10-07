"""
Batched KV cache verification.

Checkpoint A: confirms a FIXED batch of requests, run together with
per-sequence KV caching, produces token-for-token identical output to
running each request alone through the single-sequence cache
(kv_cache_single.run_single_with_cache). Deliberately includes "Hi" --
the exact short prompt that exposed the position_ids bug fixed
alongside this file -- as a regression check.

Checkpoint B: same idea, but exercises evict() and admit() mid-batch --
one request finishes and gets evicted, a short prompt gets admitted
(pads the new row), then a long prompt gets admitted (pads the existing
rows instead) -- and every request that ever passed through the batch
still has to match its own standalone reference run.

Usage:
    python -m engine.verify.kv_cache_batched
"""

from __future__ import annotations

from engine.kv_cache_batched import admit, evict, run_batch_to_completion, start_batch, step
from engine.kv_cache_single import run_single_with_cache
from engine.model_wrapper import ModelWrapper
from engine.request import Request

CHECKPOINT_A_PROMPTS = [
    ("Hi", 10),
    ("The future of artificial intelligence is", 20),
    ("Once upon a time in a small village,", 20),
]


def reference_tokens(model: ModelWrapper, prompt: str, max_new_tokens: int) -> list[int]:
    """Regenerates a prompt alone via the verified single-sequence cache."""
    ref_req = Request(prompt=prompt, max_new_tokens=max_new_tokens)
    ref_req = run_single_with_cache(model, ref_req)
    return ref_req.generated_token_ids


def check(label: str, req: Request, model: ModelWrapper) -> bool:
    reference = reference_tokens(model, req.prompt, req.max_new_tokens)
    match = req.generated_token_ids == reference
    print(f"[{'PASS' if match else 'FAIL'}] {label}: {req.prompt!r}")
    if not match:
        print(f"  ours:      {req.generated_token_ids}")
        print(f"  reference: {reference}")
    return match


def run_checkpoint_a(model: ModelWrapper) -> bool:
    print("--- Checkpoint A: fixed batch, per-sequence caching ---")
    requests = [Request(prompt=p, max_new_tokens=n) for p, n in CHECKPOINT_A_PROMPTS]
    finished = run_batch_to_completion(model, requests)

    all_passed = True
    for req in finished:
        all_passed &= check("checkpoint A", req, model)
    return all_passed


def run_checkpoint_b(model: ModelWrapper) -> bool:
    print("\n--- Checkpoint B: evict + admit mid-batch ---")
    all_seen: list[Request] = []
    all_passed = True

    a = Request(prompt="Hi", max_new_tokens=5)  # finishes first, gets evicted
    b = Request(prompt="The future of artificial intelligence is", max_new_tokens=12)
    c = Request(prompt="Once upon a time in a small village,", max_new_tokens=12)
    all_seen += [a, b, c]

    batch = start_batch(model, [a, b, c])
    for _ in range(4):
        step(model, batch, model.eos_token_id)

    assert a.is_finished(model.eos_token_id), "expected 'a' to be finished by now"
    evicted = evict(batch, [batch.requests.index(a)])
    print(f"  evicted: {[r.prompt for r in evicted]} (batch size now {batch.batch_size})")

    d = Request(prompt="Tell me a joke.", max_new_tokens=10)  # short -> pads new row
    all_seen.append(d)
    print(f"  admitting (short prompt, {len(model.encode(d.prompt))} tokens < batch seq_len {batch.seq_len})")
    admit(model, batch, d)

    e = Request(
        prompt="In a distant galaxy far beyond the reach of any known star system,",
        max_new_tokens=10,
    )  # long -> pads existing rows
    all_seen.append(e)
    print(f"  admitting (long prompt, {len(model.encode(e.prompt))} tokens vs batch seq_len {batch.seq_len})")
    admit(model, batch, e)

    while batch.batch_size > 0:
        step(model, batch, model.eos_token_id)
        done_idx = [i for i, r in enumerate(batch.requests) if r.is_finished(model.eos_token_id)]
        if done_idx:
            evicted = evict(batch, done_idx)
            print(f"  evicted: {[r.prompt for r in evicted]} (batch size now {batch.batch_size})")

    for req in all_seen:
        all_passed &= check("checkpoint B", req, model)
    return all_passed


def main():
    print("Loading model...")
    model = ModelWrapper()
    print(f"Device: {model.device}\n")

    passed_a = run_checkpoint_a(model)
    passed_b = run_checkpoint_b(model)

    print()
    if passed_a and passed_b:
        print("All checkpoints passed -- batched KV cache verified.")
    else:
        print("MISMATCH -- do not trust the batched KV cache yet.")


if __name__ == "__main__":
    main()
