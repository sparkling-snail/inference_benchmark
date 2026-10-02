from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TrafficPattern = Literal["steady", "bursty"]


@dataclass
class RuntimeTarget:
    name: str
    kind: str
    base_url: str
    model: str


@dataclass
class WorkloadCase:
    name: str
    prompt_chars: int
    max_new_tokens: int
    concurrency: int
    num_requests: int
    traffic_pattern: TrafficPattern
    request_interval_ms: int = 0
    burst_size: int | None = None


@dataclass
class RequestMetrics:
    request_index: int
    output_text: str
    output_tokens: int
    ttft_sec: float | None
    total_latency_sec: float
    tpot_sec: float | None
    success: bool
    error: str | None = None


@dataclass
class BenchmarkSummary:
    request_count: int
    success_count: int
    error_count: int
    throughput_tokens_per_sec: float
    latency_sec: dict[str, float]
    ttft_sec: dict[str, float]
    tpot_sec: dict[str, float]


@dataclass
class BenchmarkResult:
    runtime: RuntimeTarget
    workload: WorkloadCase
    repeat_index: int
    started_at_unix: float
    completed_at_unix: float
    requests: list[RequestMetrics] = field(default_factory=list)
    summary: BenchmarkSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
