"""
A fake, vLLM-shaped OpenAI-compatible server for testing the tail-latency
harness without a GPU. NOT a benchmark target -- its numbers mean nothing.

It simulates just enough engine behavior for each experiment to produce
the right *shape* of result:
  - iteration-level continuous batching (one token per running request per step)
  - a per-step token budget (--budget): decodes first, then prefill chunks
    from the FCFS queue, so a small budget spreads a long prefill over
    many steps and a large one does it in one slow step
  - step time = base + cost per decode + cost per prefill token
  - a KV capacity in tokens (--kv-tokens); when exceeded, the newest
    running request is preempted and re-queued for recompute
  - /metrics with vLLM-style names, /health, /v1/models

Usage:
    python tests/fake_vllm_server.py --port 8000 --budget 512 --kv-tokens 200000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request as HTTPRequest
from fastapi.responses import PlainTextResponse, StreamingResponse


@dataclass
class Seq:
    prompt_tokens: int
    max_tokens: int
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    prefilled: int = 0
    to_prefill: int = 0
    generated: int = 0

    def __post_init__(self):
        self.to_prefill = self.prompt_tokens


class Engine:
    def __init__(self, args):
        self.a = args
        self.waiting: list[Seq] = []
        self.running: list[Seq] = []
        self.preemptions = 0
        self.wake = asyncio.Event()

    def kv_used(self) -> int:
        return sum(s.prompt_tokens + s.generated for s in self.running) + sum(s.prefilled for s in self.waiting)

    async def loop(self):
        while True:
            if not self.waiting and not self.running:
                self.wake.clear()
                await self.wake.wait()
            budget = self.a.budget - len(self.running)
            prefill_tokens = 0
            finished_prefill = []
            for s in list(self.waiting):
                if budget <= 0 or len(self.running) + len(finished_prefill) >= self.a.max_num_seqs:
                    break
                chunk = min(budget, s.to_prefill - s.prefilled)
                s.prefilled += chunk
                budget -= chunk
                prefill_tokens += chunk
                if s.prefilled >= s.to_prefill:
                    finished_prefill.append(s)
                else:
                    break  # FCFS: don't let later requests jump an unfinished prefill
            step_s = (self.a.step_ms + 0.05 * len(self.running) + prefill_tokens * self.a.prefill_us_per_token / 1000) / 1000
            await asyncio.sleep(step_s)
            for s in self.running:
                s.generated += 1
                await s.out.put(" tok")
            for s in finished_prefill:
                self.waiting.remove(s)
                self.running.append(s)
                if s.generated == 0:
                    s.generated = 1
                    await s.out.put(" tok")
            for s in [s for s in self.running if s.generated >= s.max_tokens]:
                self.running.remove(s)
                await s.out.put(None)
            while self.kv_used() > self.a.kv_tokens and len(self.running) > 1:
                victim = self.running.pop()  # newest
                victim.to_prefill = victim.prompt_tokens + victim.generated  # recompute
                victim.prefilled = 0
                self.waiting.insert(0, victim)
                self.preemptions += 1


def create_app(args) -> FastAPI:
    app = FastAPI()
    engine = Engine(args)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(engine.loop())

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "fake-model"}]}

    @app.get("/metrics")
    async def metrics():
        lines = [
            f"vllm:num_preemptions_total{{model_name=\"fake\"}} {engine.preemptions}",
            f"vllm:kv_cache_usage_perc{{model_name=\"fake\"}} {min(1.0, engine.kv_used() / args.kv_tokens)}",
            f"vllm:num_requests_running{{model_name=\"fake\"}} {len(engine.running)}",
            f"vllm:num_requests_waiting{{model_name=\"fake\"}} {len(engine.waiting)}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/completions")
    async def completions(req: HTTPRequest):
        body = await req.json()
        seq = Seq(prompt_tokens=max(1, len(body["prompt"].split())), max_tokens=int(body.get("max_tokens", 16)))
        engine.waiting.append(seq)
        engine.wake.set()

        async def stream():
            n = 0
            while True:
                item = await seq.out.get()
                if item is None:
                    break
                n += 1
                yield "data: " + json.dumps({"choices": [{"index": 0, "text": item, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": seq.prompt_tokens, "completion_tokens": n}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--budget", type=int, default=2048, help="max tokens (decode + prefill) per step")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--kv-tokens", type=int, default=1_000_000)
    ap.add_argument("--step-ms", type=float, default=15.0)
    ap.add_argument("--prefill-us-per-token", type=float, default=60.0)
    a = ap.parse_args()
    uvicorn.run(create_app(a), host="127.0.0.1", port=a.port, log_level="warning")
