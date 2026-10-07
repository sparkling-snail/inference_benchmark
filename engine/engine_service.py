"""
Service layer that exposes the local engine over an OpenAI-compatible
HTTP interface (see server.py).

This module deliberately sits *above* the core engine code. It does
not change batching math; it wraps the existing execution paths in a
simple service API that can:

  - run single requests in a baseline "naive" mode
  - run queued requests through the continuous batching loop
  - stream token events back to an HTTP layer for fair benchmarking
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Iterator, Literal

from .kv_cache_batched import BatchedKVCache, admit, evict, start_batch, step
from .kv_cache_single import run_single_with_cache
from .model_wrapper import MODEL_NAME, ModelWrapper
from .request import Request
from .scheduler_naive import run_naive_batch

RuntimeMode = Literal["continuous", "naive", "single"]


@dataclass
class StreamEvent:
    kind: Literal["token", "done", "error"]
    text: str = ""
    token_id: int | None = None
    finish_reason: str | None = None
    metrics: dict[str, float | int | str | None] | None = None
    error: str | None = None


@dataclass
class GenerationResult:
    request_id: int
    text: str
    token_ids: list[int]
    finish_reason: str
    ttft_sec: float | None
    total_latency_sec: float | None
    tpot_sec: float | None


@dataclass
class _QueuedJob:
    request: Request
    events: queue.Queue[StreamEvent]
    emitted_tokens: int = 0


class EngineService:
    def __init__(
        self,
        runtime: RuntimeMode = "continuous",
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_batch_size: int = 4,
        scheduler_tick_ms: int = 5,
    ) -> None:
        self.runtime = runtime
        self.max_batch_size = max_batch_size
        self.scheduler_tick_s = max(scheduler_tick_ms, 0) / 1000.0
        self.model = ModelWrapper(model_name=model_name, device=device)

        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._pending: deque[_QueuedJob] = deque()
        self._jobs: dict[int, _QueuedJob] = {}
        self._worker: threading.Thread | None = None

        if self.runtime == "continuous":
            self._worker = threading.Thread(
                target=self._continuous_loop,
                name="engine-continuous-worker",
                daemon=True,
            )
            self._worker.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        final_event: StreamEvent | None = None
        for event in self.stream_generate(prompt, max_new_tokens):
            if event.kind == "done":
                final_event = event
            elif event.kind == "error":
                raise RuntimeError(event.error or "generation failed")

        if final_event is None or final_event.metrics is None:
            raise RuntimeError("generation completed without a final event")

        metrics = final_event.metrics
        return GenerationResult(
            request_id=int(metrics["request_id"]),
            text=final_event.text,
            token_ids=list(metrics["token_ids"]),
            finish_reason=str(final_event.finish_reason or "stop"),
            ttft_sec=self._as_float(metrics.get("ttft_sec")),
            total_latency_sec=self._as_float(metrics.get("total_latency_sec")),
            tpot_sec=self._as_float(metrics.get("tpot_sec")),
        )

    def stream_generate(self, prompt: str, max_new_tokens: int) -> Iterator[StreamEvent]:
        request = Request(prompt=prompt, max_new_tokens=max_new_tokens)
        events: queue.Queue[StreamEvent] = queue.Queue()

        if self.runtime == "continuous":
            job = _QueuedJob(request=request, events=events)
            with self._condition:
                self._jobs[request.id] = job
                self._pending.append(job)
                self._condition.notify()
        else:
            worker = threading.Thread(
                target=self._run_sync_job,
                args=(request, events),
                name=f"engine-{self.runtime}-job-{request.id}",
                daemon=True,
            )
            worker.start()

        while True:
            event = events.get()
            yield event
            if event.kind in {"done", "error"}:
                break

    def _run_sync_job(self, request: Request, events: queue.Queue[StreamEvent]) -> None:
        try:
            if self.runtime == "naive":
                finished = run_naive_batch(self.model, [request])[0]
            elif self.runtime == "single":
                finished = run_single_with_cache(self.model, request)
            else:
                raise ValueError(f"unsupported runtime mode: {self.runtime}")

            for token_id in finished.generated_token_ids:
                events.put(
                    StreamEvent(
                        kind="token",
                        text=self._decode_token(token_id),
                        token_id=token_id,
                    )
                )
            events.put(self._build_done_event(finished))
        except Exception as exc:  # pragma: no cover - best effort error surfacing
            events.put(StreamEvent(kind="error", error=str(exc)))

    def _continuous_loop(self) -> None:
        active: BatchedKVCache | None = None
        eos_token_id = self.model.eos_token_id

        while True:
            with self._condition:
                while (
                    not self._stop_event.is_set()
                    and active is None
                    and not self._pending
                ):
                    self._condition.wait(timeout=0.1)

                if self._stop_event.is_set() and active is None and not self._pending:
                    return

                jobs_to_admit: list[_QueuedJob] = []
                capacity = self.max_batch_size if active is None else self.max_batch_size - active.batch_size
                while self._pending and capacity > 0:
                    jobs_to_admit.append(self._pending.popleft())
                    capacity -= 1

            for job in jobs_to_admit:
                if active is None:
                    active = start_batch(self.model, [job.request])
                else:
                    admit(self.model, active, job.request)
                self._emit_new_tokens(job)

            if active is None or active.batch_size == 0:
                time.sleep(self.scheduler_tick_s or 0.001)
                continue

            step(self.model, active, eos_token_id)

            for req in active.requests:
                job = self._jobs.get(req.id)
                if job is not None:
                    self._emit_new_tokens(job)

            finished_rows = [
                i for i, req in enumerate(active.requests) if req.is_finished(eos_token_id)
            ]
            if finished_rows:
                finished_requests = [active.requests[i] for i in finished_rows]
                evict(active, finished_rows)
                for req in finished_requests:
                    job = self._jobs.pop(req.id, None)
                    if job is not None:
                        job.events.put(self._build_done_event(req))
                if active.batch_size == 0:
                    active = None

            if self.scheduler_tick_s:
                time.sleep(self.scheduler_tick_s)

    def _emit_new_tokens(self, job: _QueuedJob) -> None:
        while job.emitted_tokens < len(job.request.generated_token_ids):
            token_id = job.request.generated_token_ids[job.emitted_tokens]
            job.emitted_tokens += 1
            job.events.put(
                StreamEvent(
                    kind="token",
                    text=self._decode_token(token_id),
                    token_id=token_id,
                )
            )

    def _build_done_event(self, request: Request) -> StreamEvent:
        return StreamEvent(
            kind="done",
            text=self.model.decode(request.generated_token_ids),
            finish_reason=self._finish_reason(request),
            metrics={
                "request_id": request.id,
                "token_ids": list(request.generated_token_ids),
                "ttft_sec": request.ttft,
                "total_latency_sec": request.total_latency,
                "tpot_sec": request.tpot,
            },
        )

    def _finish_reason(self, request: Request) -> str:
        if (
            self.model.eos_token_id is not None
            and request.generated_token_ids
            and request.generated_token_ids[-1] == self.model.eos_token_id
        ):
            return "stop"
        return "length"

    def _decode_token(self, token_id: int) -> str:
        return self.model.tokenizer.decode([token_id], skip_special_tokens=False)

    @staticmethod
    def _as_float(value: float | int | str | None) -> float | None:
        if value is None:
            return None
        return float(value)
