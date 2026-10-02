from __future__ import annotations

import json
import time

import httpx

from .schema import RequestMetrics, RuntimeTarget


class OpenAICompatClient:
    def __init__(self, runtime: RuntimeTarget, timeout_sec: float = 120.0) -> None:
        self.runtime = runtime
        self.timeout_sec = timeout_sec

    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        request_index: int,
    ) -> RequestMetrics:
        start = time.perf_counter()
        first_token_at: float | None = None
        output_chunks: list[str] = []
        output_tokens = 0
        finish_reason = "unknown"
        final_metrics: dict[str, float] = {}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                async with client.stream(
                    "POST",
                    f"{self.runtime.base_url.rstrip('/')}/v1/completions",
                    json={
                        "model": self.runtime.model,
                        "prompt": prompt,
                        "max_tokens": max_new_tokens,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break

                        chunk = json.loads(payload)
                        choice = chunk["choices"][0]
                        text = choice.get("text", "")
                        token_id = chunk.get("token_id")
                        if text:
                            output_chunks.append(text)
                            output_tokens += 1 if token_id is not None else 0
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                        if "metrics" in chunk:
                            final_metrics = chunk["metrics"]
        except Exception as exc:
            return RequestMetrics(
                request_index=request_index,
                output_text="",
                output_tokens=0,
                ttft_sec=None,
                total_latency_sec=time.perf_counter() - start,
                tpot_sec=None,
                success=False,
                error=str(exc),
            )

        total_latency = time.perf_counter() - start
        ttft = (
            float(final_metrics["ttft_sec"])
            if final_metrics.get("ttft_sec") is not None
            else (first_token_at - start if first_token_at is not None else None)
        )
        tpot = (
            float(final_metrics["tpot_sec"])
            if final_metrics.get("tpot_sec") is not None
            else ((total_latency - ttft) / (output_tokens - 1) if ttft is not None and output_tokens > 1 else None)
        )
        return RequestMetrics(
            request_index=request_index,
            output_text="".join(output_chunks),
            output_tokens=output_tokens,
            ttft_sec=ttft,
            total_latency_sec=total_latency,
            tpot_sec=tpot,
            success=True,
            error=None if finish_reason else "missing finish reason",
        )
