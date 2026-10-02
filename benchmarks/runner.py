from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

import yaml

from .clients import OpenAICompatClient
from .measure import summarize_requests
from .plot import plot_latency_and_throughput
from .report import flatten_results, load_raw_results, write_markdown_summary
from .schema import BenchmarkResult, RuntimeTarget, WorkloadCase
from .workloads import build_prompts


async def run_case(
    client: OpenAICompatClient,
    workload: WorkloadCase,
    repeat_index: int,
    warmup_requests: int,
    output_dir: Path,
) -> BenchmarkResult:
    prompts = build_prompts(workload)

    for prompt in prompts[:warmup_requests]:
        await client.stream_generate(prompt, workload.max_new_tokens, request_index=-1)

    semaphore = asyncio.Semaphore(workload.concurrency)
    measured_prompts = prompts[warmup_requests:]
    started_at = time.time()

    async def one_request(request_index: int, prompt: str):
        async with semaphore:
            return await client.stream_generate(prompt, workload.max_new_tokens, request_index=request_index)

    tasks = []
    if workload.traffic_pattern == "steady":
        for i, prompt in enumerate(measured_prompts):
            tasks.append(asyncio.create_task(one_request(i, prompt)))
            if workload.request_interval_ms > 0:
                await asyncio.sleep(workload.request_interval_ms / 1000.0)
    else:
        burst_size = workload.burst_size or workload.concurrency
        for offset in range(0, len(measured_prompts), burst_size):
            for i, prompt in enumerate(measured_prompts[offset : offset + burst_size], start=offset):
                tasks.append(asyncio.create_task(one_request(i, prompt)))
            await asyncio.sleep(workload.request_interval_ms / 1000.0)

    request_metrics = await asyncio.gather(*tasks)
    completed_at = time.time()
    result = BenchmarkResult(
        runtime=client.runtime,
        workload=workload,
        repeat_index=repeat_index,
        started_at_unix=started_at,
        completed_at_unix=completed_at,
        requests=request_metrics,
    )
    result.summary = summarize_requests(request_metrics, wall_time_sec=completed_at - started_at)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{client.runtime.name}_{workload.name}_repeat{repeat_index}.json"
        .replace("/", "_")
        .replace(" ", "_")
    )
    (output_dir / filename).write_text(json.dumps(result.to_dict(), indent=2))
    return result


async def main_async(config_path: Path, runtime_name: str | None = None) -> None:
    config = yaml.safe_load(config_path.read_text())
    runtimes = [
        RuntimeTarget(**runtime_cfg)
        for runtime_cfg in config["runtimes"]
        if runtime_name is None or runtime_cfg["name"] == runtime_name
    ]
    workloads = [WorkloadCase(**workload_cfg) for workload_cfg in config["workloads"]]
    repeats = int(config.get("repeats", 1))
    warmup_requests = int(config.get("warmup_requests", 0))
    output_dir = Path(config.get("output_dir", "results/phase5/raw"))

    for runtime in runtimes:
        client = OpenAICompatClient(runtime)
        for workload in workloads:
            for repeat_index in range(repeats):
                await run_case(client, workload, repeat_index, warmup_requests, output_dir)

    raw_results = load_raw_results(output_dir)
    summary_rows = flatten_results(raw_results)
    summary_dir = output_dir.parent
    write_markdown_summary(summary_rows, summary_dir / "summary.md")
    plot_latency_and_throughput(summary_rows, summary_dir / "plots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 5 benchmark matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_matrix.yaml"))
    parser.add_argument("--runtime", default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args.config, args.runtime))


if __name__ == "__main__":
    main()
