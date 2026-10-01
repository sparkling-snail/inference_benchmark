"""
Phase 4 -- continuous batching scheduler.

Everything through Phase 3 operated on a FIXED set of requests: hand it
a list up front and it runs them as one locked batch (Phase 1, Phase 3
Checkpoint A), or exercises admit/evict manually in a test (Phase 3
Checkpoint B). This phase is what makes it a *server*: a queue of
requests, and a scheduler that keeps the GPU busy by refilling a freed
batch slot the instant a row finishes, instead of waiting for the whole
batch to drain before starting the next one.

Policy: FCFS, fixed max_batch_size. Admission order follows each
Request's arrival_time (set at construction), not list order, so
handing the scheduler an out-of-order list still behaves correctly. A
smarter policy (priority, deadline-aware, shortest-job-first, ...) is a
drop-in replacement of the admission choice in _fill() later -- nothing
else here is FCFS-specific.

Calling convention for admit()+step(), matching Phase 3's own
Checkpoint B test (verify_kv_cache_batched.py): admit() does the new
row's prefill and records its first token, so a newly admitted row is
already "caught up" to every other active row by the time the next
step() call runs -- no special-casing needed, every active row just
gets step()'d together each iteration regardless of when it joined.

Correctness note: Phase 3's admit()/evict() already guarantee a row's
generation is unaffected by its batch-mates (own cache slot, own
position ids per build_position_ids). This scheduler only changes
ORDERING/ADMISSION on top of that -- it introduces no new per-token
math, so Phase 2's single-sequence reference is still the right
correctness oracle regardless of scheduling order. See
verify_scheduler.py.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .kv_cache_batched import BatchedKVCache, admit, evict, start_batch, step
from .model_wrapper import ModelWrapper
from .request import Request

# Fired as on_event(step_index, kind, request) for kind in
# {"admit", "evict"} -- optional hook for logging/benchmarking the
# scheduler's own behavior (batch occupancy over time, admit/evict
# timeline). Not needed for correctness.
SchedulerEventHook = Callable[[int, str, Request], None]


@dataclass
class SchedulerStats:
    """
    Per-step batch occupancy, for the Phase 5 "how full was the GPU"
    story. One entry appended per scheduler step (the first batch's
    initial prefill counts as step 0).
    """

    step_batch_sizes: list[int] = field(default_factory=list)
    max_batch_size: int = 0

    @property
    def mean_occupancy(self) -> float:
        if not self.step_batch_sizes:
            return 0.0
        return sum(self.step_batch_sizes) / len(self.step_batch_sizes)

    @property
    def mean_occupancy_pct(self) -> float:
        if self.max_batch_size == 0:
            return 0.0
        return 100.0 * self.mean_occupancy / self.max_batch_size


def run_scheduler(
    model: ModelWrapper,
    requests: list[Request],
    max_batch_size: int,
    on_event: SchedulerEventHook | None = None,
) -> tuple[list[Request], SchedulerStats]:
    """
    Runs every request in `requests` to completion under continuous
    batching: at most `max_batch_size` requests active at once, FCFS by
    arrival_time, freed slots refilled from the queue before the next
    step rather than at the next full-batch boundary.

    Returns (finished_requests, stats). finished_requests is in FINISH
    order (not submission order) -- that order is itself a result worth
    keeping, since under FCFS a short request submitted late can finish
    before a long one submitted earlier.
    """
    queue: deque[Request] = deque(sorted(requests, key=lambda r: r.arrival_time))
    active: BatchedKVCache | None = None
    finished: list[Request] = []
    stats = SchedulerStats(max_batch_size=max_batch_size)
    eos_token_id = model.eos_token_id
    step_index = 0

    def fill() -> None:
        nonlocal active
        while queue and (active is None or active.batch_size < max_batch_size):
            req = queue.popleft()
            if active is None:
                active = start_batch(model, [req])
            else:
                admit(model, active, req)
            if on_event:
                on_event(step_index, "admit", req)

    while queue or (active is not None and active.batch_size > 0):
        fill()

        if active is None or active.batch_size == 0:
            break  # nothing admittable and nothing running; loop condition prevents reaching here

        stats.step_batch_sizes.append(active.batch_size)
        step(model, active, eos_token_id)

        finished_rows = [
            i for i, req in enumerate(active.requests) if req.is_finished(eos_token_id)
        ]
        if finished_rows:
            evicted = evict(active, finished_rows)
            finished.extend(evicted)
            if on_event:
                for req in evicted:
                    on_event(step_index, "evict", req)
            if active.batch_size == 0:
                active = None

        step_index += 1

    return finished, stats
