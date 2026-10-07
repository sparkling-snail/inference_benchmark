"""
Continuous batching scheduler.

The lower layers operate on a FIXED set of requests: hand them a list
up front and they run it as one locked batch (scheduler_naive.py,
kv_cache_batched.py Checkpoint A), or exercise admit/evict manually in
a test (Checkpoint B). This module is what makes it a *server*: a queue of
requests, and a scheduler that keeps the GPU busy by refilling a freed
batch slot the instant a row finishes, instead of waiting for the whole
batch to drain before starting the next one.

Policy: pluggable admission, fixed max_batch_size. The admission choice
in fill() is the only policy-specific code; everything else (prefill on
admit, step, evict) is shared. Built-in policies (see POLICIES):

  - "fcfs": first come, first served by arrival_time (the default, and
    what vLLM does out of the box). Order follows each Request's
    arrival_time, not list order, so handing the scheduler an
    out-of-order list still behaves correctly.
  - "sjf": shortest job first, by max_new_tokens. Uses the requested
    output budget as an oracle for job length -- a real server only has
    an estimate (see "Efficient LLM Scheduling by Learning to Rank"),
    so treat this as the best case. Cuts head-of-line blocking for short
    requests, but can starve long ones under sustained load.
  - "sjf_aging": SJF, except any request that has waited longer than
    max_wait_s jumps the line (FCFS among those). Bounds the starvation
    SJF introduces, at a small cost to short-request latency.

By default every request is treated as already arrived (the original
behavior). With respect_arrivals=True, a request is only
admittable once perf_counter() >= its arrival_time, which lets
experiments replay an open-loop arrival process (e.g. Poisson) against
the real engine -- that's what makes queueing, and so tail latency,
show up at all.

Calling convention for admit()+step(), matching the batched
cache's own Checkpoint B test (engine/verify/kv_cache_batched.py): admit() does the new
row's prefill and records its first token, so a newly admitted row is
already "caught up" to every other active row by the time the next
step() call runs -- no special-casing needed, every active row just
gets step()'d together each iteration regardless of when it joined.

Correctness note: kv_cache_batched's admit()/evict() already guarantee a row's
generation is unaffected by its batch-mates (own cache slot, own
position ids per build_position_ids). This scheduler only changes
ORDERING/ADMISSION on top of that -- it introduces no new per-token
math, so the single-sequence cache's output is still the right
correctness oracle regardless of scheduling order. See
engine/verify/scheduler.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

from .kv_cache_batched import BatchedKVCache, admit, evict, start_batch, step
from .model_wrapper import ModelWrapper
from .request import Request

# Fired as on_event(step_index, kind, request) for kind in
# {"admit", "evict"} -- optional hook for logging/benchmarking the
# scheduler's own behavior (batch occupancy over time, admit/evict
# timeline). Not needed for correctness.
SchedulerEventHook = Callable[[int, str, Request], None]

# An admission policy picks which of the arrived, waiting requests gets
# the next free batch slot: policy(waiting, now) -> index into waiting.
AdmissionPolicy = Callable[[list[Request], float], int]


def fcfs(waiting: list[Request], now: float) -> int:
    return min(range(len(waiting)), key=lambda i: (waiting[i].arrival_time, waiting[i].id))


def sjf(waiting: list[Request], now: float) -> int:
    return min(range(len(waiting)), key=lambda i: (waiting[i].max_new_tokens, waiting[i].arrival_time, waiting[i].id))


def make_sjf_aging(max_wait_s: float) -> AdmissionPolicy:
    def sjf_aging(waiting: list[Request], now: float) -> int:
        starving = [i for i, r in enumerate(waiting) if now - r.arrival_time >= max_wait_s]
        if starving:
            return min(starving, key=lambda i: (waiting[i].arrival_time, waiting[i].id))
        return sjf(waiting, now)

    return sjf_aging


POLICIES: dict[str, AdmissionPolicy] = {
    "fcfs": fcfs,
    "sjf": sjf,
    "sjf_aging": make_sjf_aging(max_wait_s=2.0),
}


@dataclass
class SchedulerStats:
    """
    Per-step batch occupancy, for the "how full was the batch"
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
    def mean_occupancy_pct(self) -> float:  #on average the batch was X% full
        if self.max_batch_size == 0:
            return 0.0
        return 100.0 * self.mean_occupancy / self.max_batch_size


def run_scheduler(
    model: ModelWrapper,
    requests: list[Request],
    max_batch_size: int,
    on_event: SchedulerEventHook | None = None,
    policy: str | AdmissionPolicy = "fcfs",
    respect_arrivals: bool = False,
    ignore_eos: bool = False,
) -> tuple[list[Request], SchedulerStats]:
    """
    Runs every request in `requests` to completion under continuous
    batching: at most `max_batch_size` requests active at once, freed
    slots refilled from the queue before the next step rather than at
    the next full-batch boundary. `policy` chooses who gets a freed slot
    (a name from POLICIES, or any AdmissionPolicy callable).

    respect_arrivals: only admit a request once its arrival_time has
    passed (for replaying arrival processes in experiments).
    ignore_eos: run every request to exactly max_new_tokens, so output
    length is controlled by the experiment, not by the model.

    Returns (finished_requests, stats). finished_requests is in FINISH
    order (not submission order) -- that order is itself a result worth
    keeping, since a short request submitted late can finish before a
    long one submitted earlier.
    """
    choose = POLICIES[policy] if isinstance(policy, str) else policy
    pending: list[Request] = sorted(requests, key=lambda r: (r.arrival_time, r.id))  # not yet arrived
    waiting: list[Request] = []  # arrived, not yet admitted
    active: BatchedKVCache | None = None
    finished: list[Request] = []
    stats = SchedulerStats(max_batch_size=max_batch_size)
    eos_token_id = None if ignore_eos else model.eos_token_id
    step_index = 0

    def collect_arrivals() -> None:
        now = time.perf_counter()
        while pending and (not respect_arrivals or pending[0].arrival_time <= now):
            waiting.append(pending.pop(0))

    def fill() -> None:
        nonlocal active
        collect_arrivals()
        while waiting and (active is None or active.batch_size < max_batch_size):
            req = waiting.pop(choose(waiting, time.perf_counter()))
            req.admit_time = time.perf_counter()
            if active is None:
                active = start_batch(model, [req])
            else:
                admit(model, active, req)
            if on_event:
                on_event(step_index, "admit", req)

    while pending or waiting or (active is not None and active.batch_size > 0):
        fill()

        if active is None or active.batch_size == 0:
            if pending:  # idle until the next request arrives
                time.sleep(max(0.0, pending[0].arrival_time - time.perf_counter()))
                continue
            break

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
