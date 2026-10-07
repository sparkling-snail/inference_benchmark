"""
Streaming load generator for any OpenAI-compatible server (vLLM, SGLang,
TGI's OpenAI route, or this repo's engine/server.py).

Records a timestamp for EVERY streamed chunk, not just first/last, so the
experiments can look at inter-token latency (ITL) per token. A request's
average TPOT hides stalls: one 2-second pause spread over 500 tokens
looks like +4 ms/token, but the user saw the stream freeze for 2 seconds.

Note: servers may coalesce several tokens into one chunk under load; the
gaps here are per *chunk*, which is what the user actually experiences.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field

import httpx


@dataclass
class RequestSpec:
    offset_s: float        # when to send, relative to the start of the run
    prompt: str
    max_tokens: int
    tag: str = ""          # free-form label, e.g. "background" / "long_prompt"


@dataclass
class RequestResult:
    tag: str
    offset_s: float
    max_tokens: int
    sent_at_s: float | None = None          # relative to run start
    chunk_times_s: list[float] = field(default_factory=list)  # relative to run start
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    ok: bool = False
    error: str | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.sent_at_s is None or not self.chunk_times_s:
            return None
        return self.chunk_times_s[0] - self.sent_at_s

    @property
    def e2e_s(self) -> float | None:
        if self.sent_at_s is None or not self.chunk_times_s:
            return None
        return self.chunk_times_s[-1] - self.sent_at_s

    @property
    def itls_s(self) -> list[float]:
        t = self.chunk_times_s
        return [b - a for a, b in zip(t, t[1:])]

    def to_dict(self, keep_chunk_times: bool = True) -> dict:
        d = asdict(self)
        if not keep_chunk_times:
            d.pop("chunk_times_s")
        d.update(ttft_s=self.ttft_s, e2e_s=self.e2e_s)
        return d


async def _one(client: httpx.AsyncClient, base_url: str, model: str, spec: RequestSpec, t0: float, ignore_eos: bool) -> RequestResult:
    res = RequestResult(tag=spec.tag, offset_s=spec.offset_s, max_tokens=spec.max_tokens)
    delay = t0 + spec.offset_s - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    body = {
        "model": model,
        "prompt": spec.prompt,
        "max_tokens": spec.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True  # vLLM extension: always generate exactly max_tokens
    res.sent_at_s = time.perf_counter() - t0
    try:
        async with client.stream("POST", f"{base_url.rstrip('/')}/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    res.completion_tokens = chunk["usage"].get("completion_tokens")
                    res.prompt_tokens = chunk["usage"].get("prompt_tokens")
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    res.chunk_times_s.append(time.perf_counter() - t0)
        res.ok = bool(res.chunk_times_s)
        if not res.ok:
            res.error = "no tokens streamed"
    except Exception as exc:  # keep going; failures are part of the result
        res.error = f"{type(exc).__name__}: {exc}"
    return res


async def run_open_loop(
    base_url: str,
    model: str,
    specs: list[RequestSpec],
    ignore_eos: bool = True,
    timeout_s: float = 600.0,
    on_tick=None,
    tick_s: float = 0.5,
) -> tuple[list[RequestResult], float]:
    """Fire every spec at its offset (open loop) and wait for all of them.

    on_tick: optional async callback(elapsed_s) invoked every tick_s while
    the run is in flight -- used to sample /metrics during the run.
    Returns (results in spec order, wall time in seconds).
    """
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        t0 = time.perf_counter() + 0.1
        tasks = [asyncio.create_task(_one(client, base_url, model, s, t0, ignore_eos)) for s in specs]
        if on_tick is not None:
            while not all(t.done() for t in tasks):
                await on_tick(time.perf_counter() - t0)
                await asyncio.sleep(tick_s)
        results = await asyncio.gather(*tasks)
        return list(results), time.perf_counter() - t0


async def wait_healthy(base_url: str, timeout_s: float = 900.0) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=5) as client:
        while time.time() < deadline:
            for path in ("/health", "/healthz"):
                try:
                    r = await client.get(base_url.rstrip("/") + path)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
            await asyncio.sleep(2)
    raise TimeoutError(f"{base_url} not healthy after {timeout_s}s")


async def resolve_model(base_url: str, model: str | None) -> str:
    """Use --model if given, otherwise the first model the server lists."""
    if model:
        return model
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(base_url.rstrip("/") + "/v1/models")
        r.raise_for_status()
        return r.json()["data"][0]["id"]


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


async def scrape_metrics(base_url: str) -> dict[str, float]:
    """Prometheus /metrics -> {metric_name: value summed over label sets}.

    Returns {} if the server has no /metrics endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(base_url.rstrip("/") + "/metrics")
            if r.status_code != 200:
                return {}
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def pick(metrics: dict[str, float], *needles: str) -> float | None:
    """Sum of every metric whose name contains all needles (version-tolerant:
    vLLM has renamed e.g. gpu_cache_usage_perc -> kv_cache_usage_perc)."""
    vals = [v for k, v in metrics.items() if all(n in k for n in needles) and not k.endswith("_created")]
    return sum(vals) if vals else None
