"""
Minimal OpenAI-compatible HTTP wrapper for the local benchmark runtime.

Supported endpoints:
  - GET  /healthz
  - POST /v1/completions
  - POST /v1/chat/completions

This is intentionally small: it only implements the request/response
shape needed by the Phase 5 benchmark harness.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from src.engine_service import EngineService, GenerationResult, RuntimeMode


class CompletionRequest(BaseModel):
    model: str = "local"
    prompt: str
    max_tokens: int = Field(default=32, ge=1)
    stream: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=32, ge=1)
    stream: bool = False


def create_app(service: EngineService, served_model_name: str) -> FastAPI:
    app = FastAPI(title="mini-inference-server phase5")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        service.close()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model": served_model_name, "runtime": service.runtime}

    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        if request.stream:
            return StreamingResponse(
                _completion_event_stream(service, served_model_name, request.prompt, request.max_tokens),
                media_type="text/event-stream",
            )
        result = service.generate(prompt=request.prompt, max_new_tokens=request.max_tokens)
        return JSONResponse(_completion_response(served_model_name, request.prompt, result))

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        prompt = _messages_to_prompt(request.messages)
        if request.stream:
            return StreamingResponse(
                _chat_event_stream(service, served_model_name, prompt, request.max_tokens),
                media_type="text/event-stream",
            )
        result = service.generate(prompt=prompt, max_new_tokens=request.max_tokens)
        return JSONResponse(_chat_response(served_model_name, result))

    return app


def _completion_response(model_name: str, prompt: str, result: GenerationResult) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"cmpl-{result.request_id}",
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": result.text,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(result.token_ids),
            "total_tokens": len(result.token_ids),
        },
        "metrics": {
            "ttft_sec": result.ttft_sec,
            "total_latency_sec": result.total_latency_sec,
            "tpot_sec": result.tpot_sec,
        },
        "prompt": prompt,
    }


def _chat_response(model_name: str, result: GenerationResult) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{result.request_id}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(result.token_ids),
            "total_tokens": len(result.token_ids),
        },
        "metrics": {
            "ttft_sec": result.ttft_sec,
            "total_latency_sec": result.total_latency_sec,
            "tpot_sec": result.tpot_sec,
        },
    }


def _completion_event_stream(
    service: EngineService,
    model_name: str,
    prompt: str,
    max_tokens: int,
):
    created = int(time.time())
    for event in service.stream_generate(prompt=prompt, max_new_tokens=max_tokens):
        if event.kind == "token":
            payload = {
                "id": "cmpl-stream",
                "object": "text_completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": event.text,
                        "finish_reason": None,
                    }
                ],
                "token_id": event.token_id,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        elif event.kind == "done":
            payload = {
                "id": "cmpl-stream",
                "object": "text_completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": event.finish_reason,
                    }
                ],
                "metrics": event.metrics,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            raise HTTPException(status_code=500, detail=event.error or "generation failed")


def _chat_event_stream(
    service: EngineService,
    model_name: str,
    prompt: str,
    max_tokens: int,
):
    created = int(time.time())
    for event in service.stream_generate(prompt=prompt, max_new_tokens=max_tokens):
        if event.kind == "token":
            payload = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event.text},
                        "finish_reason": None,
                    }
                ],
                "token_id": event.token_id,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        elif event.kind == "done":
            payload = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": event.finish_reason,
                    }
                ],
                "metrics": event.metrics,
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        else:
            raise HTTPException(status_code=500, detail=event.error or "generation failed")


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Phase 5 benchmark server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--runtime",
        choices=["continuous", "naive", "single"],
        default="continuous",
    )
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--scheduler-tick-ms", type=int, default=5)
    args = parser.parse_args()

    service = EngineService(
        runtime=args.runtime,
        model_name=args.model_name,
        max_batch_size=args.max_batch_size,
        scheduler_tick_ms=args.scheduler_tick_ms,
    )
    app = create_app(service, served_model_name=args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
