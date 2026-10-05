"""
Request abstraction used by the queue, scheduler, and executor.

Every phase of this project (naive static batching -> single-request KV
cache -> batched KV cache -> continuous batching scheduler) reuses this
same Request object, just adds more state to it over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools
import time


class RequestStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"


_id_counter = itertools.count()


@dataclass
class Request:
    prompt: str
    max_new_tokens: int = 32

    # assigned automatically
    id: int = field(default_factory=lambda: next(_id_counter))
    status: RequestStatus = RequestStatus.QUEUED

    # filled in as the request is processed
    prompt_token_ids: list[int] | None = None
    generated_token_ids: list[int] = field(default_factory=list)

    # timing, for benchmarking later phases (TTFT / TPOT)
    arrival_time: float = field(default_factory=time.perf_counter)
    admit_time: float | None = None  # when the scheduler let it into the batch (queue wait ends)
    first_token_time: float | None = None
    finish_time: float | None = None
    # one perf_counter() timestamp per generated token, so inter-token
    # latency (ITL) can be measured per token rather than averaged per request
    token_times: list[float] = field(default_factory=list)

    def is_finished(self, eos_token_id: int | None) -> bool:
        if len(self.generated_token_ids) >= self.max_new_tokens:
            return True
        if eos_token_id is not None and self.generated_token_ids:
            if self.generated_token_ids[-1] == eos_token_id:
                return True
        return False

    def record_token(self, token_id: int) -> None:
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
        self.generated_token_ids.append(token_id)

    def mark_finished(self) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_time = time.perf_counter()

    @property
    def ttft(self) -> float | None:
        """Time to first token, seconds."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def queue_wait(self) -> float | None:
        """Time spent waiting in the queue before admission, seconds."""
        if self.admit_time is None:
            return None
        return self.admit_time - self.arrival_time

    @property
    def itls(self) -> list[float]:
        """Inter-token latencies (gaps between consecutive tokens), seconds."""
        t = self.token_times
        return [b - a for a, b in zip(t, t[1:])]

    @property
    def total_latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        """Time per output token (excluding the first), seconds."""
        n = len(self.generated_token_ids)
        if n < 2 or self.finish_time is None or self.first_token_time is None:
            return None
        return (self.finish_time - self.first_token_time) / (n - 1)
