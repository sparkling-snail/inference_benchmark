"""
Start an inference server for one Deployment, wait until it is healthy,
record the environment it ran in, and tear it down afterwards.

    async with launch(dep, port=8000, log_path=...) as server:
        ... benchmark server.base_url ...

Engine flags drift between releases (vLLM especially). Everything that
maps the matrix's vocabulary (tp, precision, spec_decode, ...) onto real
CLI flags lives in build_command(), and extra_args is the escape hatch
for anything it doesn't cover. Check `vllm serve --help` /
`python -m sglang.launch_server --help` against the pinned version.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from .loadgen import wait_healthy
from .matrix import Deployment

VLLM_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp8": "bfloat16", "auto": "auto"}


def build_command(dep: Deployment, port: int) -> list[str] | None:
    """The server command for a deployment, or None if nothing should be launched."""
    if dep.engine == "external":
        return None
    if dep.engine == "vllm":
        cmd = ["vllm", "serve", dep.model, "--port", str(port),
               "--tensor-parallel-size", str(dep.tp),
               "--dtype", VLLM_DTYPE[dep.precision]]
        if dep.pp > 1:
            cmd += ["--pipeline-parallel-size", str(dep.pp)]
        if dep.ep:
            cmd += ["--enable-expert-parallel"]
        if dep.precision == "fp8":
            cmd += ["--quantization", "fp8"]  # dynamic per-tensor FP8 weights, no calibration
        if dep.max_model_len:
            cmd += ["--max-model-len", str(dep.max_model_len)]
        if dep.spec_decode:
            spec = {"method": dep.spec_decode.method, "num_speculative_tokens": dep.spec_decode.num_tokens}
            if dep.spec_decode.method == "ngram":
                spec["prompt_lookup_max"] = 4
            if dep.spec_decode.draft_model:
                spec["model"] = dep.spec_decode.draft_model
            cmd += ["--speculative-config", json.dumps(spec)]
    elif dep.engine == "sglang":
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", dep.model,
               "--port", str(port), "--tp-size", str(dep.tp)]
        if dep.precision in ("bf16", "fp16"):
            cmd += ["--dtype", VLLM_DTYPE[dep.precision]]
        if dep.pp > 1:
            cmd += ["--pp-size", str(dep.pp)]
        if dep.ep:
            cmd += ["--ep-size", str(dep.tp)]
        if dep.precision == "fp8":
            cmd += ["--quantization", "fp8"]
        if dep.max_model_len:
            cmd += ["--context-length", str(dep.max_model_len)]
        if dep.spec_decode:
            cmd += ["--speculative-algorithm", dep.spec_decode.method.upper(),
                    "--speculative-num-draft-tokens", str(dep.spec_decode.num_tokens)]
            if dep.spec_decode.draft_model:
                cmd += ["--speculative-draft-model-path", dep.spec_decode.draft_model]
    elif dep.engine == "local":
        cmd = [sys.executable, "-m", "engine.server", "--port", str(port), "--model-name", dep.model]
    elif dep.engine == "fake":
        cmd = [sys.executable, "tests/fake_vllm_server.py",
               "--port", str(port)]
    else:
        raise ValueError(f"unknown engine {dep.engine!r}")
    return cmd + list(dep.extra_args)


@dataclass
class Server:
    base_url: str
    command: list[str] | None
    environment: dict = field(default_factory=dict)


@contextlib.asynccontextmanager
async def launch(dep: Deployment, port: int, log_path: Path, health_timeout_s: float = 1800.0) -> AsyncIterator[Server]:
    cmd = build_command(dep, port)
    if cmd is None:
        await wait_healthy(dep.base_url, timeout_s=60)
        yield Server(base_url=dep.base_url, command=None, environment=capture_environment(dep))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        # own process group, so TP workers the engine forks die with it
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            base_url = f"http://127.0.0.1:{port}"
            await _wait_or_crash(proc, base_url, health_timeout_s, log_path)
            yield Server(base_url=base_url, command=cmd, environment=capture_environment(dep))
        finally:
            _stop(proc)


async def _wait_or_crash(proc: subprocess.Popen, base_url: str, timeout_s: float, log_path: Path) -> None:
    health = asyncio.create_task(wait_healthy(base_url, timeout_s=timeout_s))
    while not health.done():
        if proc.poll() is not None:
            health.cancel()
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-20:])
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy:\n{tail}")
        await asyncio.sleep(1)
    health.result()


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def capture_environment(dep: Deployment) -> dict:
    """GPU, driver, topology and engine version: what a recipe is only valid for."""
    env: dict = {}
    gpus = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if gpus:
        env["gpus"] = [g.strip() for g in gpus.splitlines()]
        env["topology"] = _run(["nvidia-smi", "topo", "-m"])
        env["cuda"] = _run([sys.executable, "-c", "import torch; print(torch.version.cuda)"])
    module = {"vllm": "vllm", "sglang": "sglang", "local": "torch"}.get(dep.engine)
    if module:
        env["engine_version"] = _run([sys.executable, "-c", f"import {module}; print({module}.__version__)"])
    return {k: v for k, v in env.items() if v}


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    except Exception:
        return None
