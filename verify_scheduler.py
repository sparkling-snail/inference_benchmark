"""
Phase 4 verification.

Confirms the scheduler produces the same per-request output regardless
of scheduling order or batch pressure, by checking every request
against Phase 2's single-sequence reference -- same correctness oracle
used for Phase 3, since the scheduler introduces no new per-token math
(see scheduler.py's docstring).

Two checks:
  - Correctness: every request's generated tokens match running it
    alone through Phase 2's cache, regardless of how it got
    admitted/evicted along the way.
  - Continuous-batching actually happened: with more requests than
    max_batch_size and deliberately uneven max_new_tokens (so some
    finish early), assert at least one admit event fires *after* the
    initial fill -- i.e. a freed slot got reused mid-run instead of the
    scheduler idling until everyone in the first batch finished. This
    is the property Phase 3 Checkpoint A didn't have and this phase
    exists to add.

Both checks run once per admission policy (fcfs, sjf): a policy only
changes WHO gets a free slot, so per-request output must not change.

Usage:
    python verify_scheduler.py
"""

from __future__ import annotations

from src.kv_cache_single import run_single_with_cache
from src.model_wrapper import ModelWrapper
from src.request import Request
from src.scheduler import run_scheduler

MAX_BATCH_SIZE = 3

# Deliberately more requests than max_batch_size, with uneven
# max_new_tokens, so some rows finish well before others and a mid-run
# admit is forced to happen if continuous batching is actually working.
WORKLOAD = [
    ("Hi", 6),
    ("The future of artificial intelligence is", 24),
    ("Once upon a time in a small village,", 24),
    ("The best way to learn a new programming language is", 10),
    ("In the year 2050, cities will", 18),
    ("My favorite recipe for a quick dinner is", 8),
    ("The most important lesson I learned from my first job was", 20),
    ("Climate change is affecting the way we", 14),
]


def reference_tokens(model: ModelWrapper, prompt: str, max_new_tokens: int) -> list[int]:
    ref_req = Request(prompt=prompt, max_new_tokens=max_new_tokens)
    ref_req = run_single_with_cache(model, ref_req)
    return ref_req.generated_token_ids


def main():
    print("Loading model...")
    model = ModelWrapper()
    print(f"Device: {model.device}\n")

    results = {policy: check_policy(model, policy) for policy in ("fcfs", "sjf")}
    print()
    for policy, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] policy={policy}")
    if all(results.values()):
        print("\nAll policies verified. Phase 4 verified.")
    else:
        print("\nMISMATCH or no continuous batching observed -- do not proceed to Phase 5.")


def check_policy(model: ModelWrapper, policy: str) -> bool:
    print(f"=== policy: {policy} ===")

    requests = [Request(prompt=p, max_new_tokens=n) for p, n in WORKLOAD]
    prompt_by_id = {req.id: req.prompt for req in requests}
    max_new_tokens_by_id = {req.id: req.max_new_tokens for req in requests}

    events: list[tuple[int, str, int]] = []  # (step_index, kind, request_id)

    def on_event(step_index: int, kind: str, req: Request) -> None:
        events.append((step_index, kind, req.id))

    print(f"Running {len(requests)} requests through the scheduler (max_batch_size={MAX_BATCH_SIZE})...\n")
    finished, stats = run_scheduler(
        model, requests, max_batch_size=MAX_BATCH_SIZE, on_event=on_event, policy=policy
    )

    print("Scheduling timeline:")
    for step_index, kind, req_id in events:
        print(f"  step {step_index:3d}  {kind:5s}  req {req_id} ({prompt_by_id[req_id]!r})")

    # --- Check 1: a freed slot actually got reused mid-run ---------------
    initial_admit_steps = {s for s, k, _ in events if k == "admit"}
    first_fill_step = min(initial_admit_steps) if initial_admit_steps else 0
    late_admits = [e for e in events if e[1] == "admit" and e[0] > first_fill_step]
    continuous_batching_observed = len(late_admits) > 0

    print(f"\n{'[PASS]' if continuous_batching_observed else '[FAIL]'} "
          f"continuous batching observed: {len(late_admits)} admit(s) happened after the initial fill")

    # --- Check 2: every request matches its standalone reference ---------
    all_passed = continuous_batching_observed
    print()
    for req in sorted(finished, key=lambda r: r.id):
        reference = reference_tokens(model, req.prompt, max_new_tokens_by_id[req.id])
        match = req.generated_token_ids == reference
        all_passed &= match
        print(f"[{'PASS' if match else 'FAIL'}] req {req.id}: {req.prompt!r}")
        if not match:
            print(f"  ours:      {req.generated_token_ids}")
            print(f"  reference: {reference}")

    print(f"\nBatch occupancy: mean {stats.mean_occupancy:.2f}/{stats.max_batch_size} "
          f"({stats.mean_occupancy_pct:.1f}%) over {len(stats.step_batch_sizes)} steps")

    print()
    if all_passed:
        print(f"All {len(requests)} requests match their single-sequence reference, "
              f"and continuous batching was exercised (policy={policy}).\n")
    return all_passed


if __name__ == "__main__":
    main()
