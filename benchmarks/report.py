from __future__ import annotations

import json
from pathlib import Path


def write_markdown_summary(summary_rows: list[dict], output_path: Path) -> None:
    lines = [
        "| runtime | workload | p50 latency | p95 latency | p50 ttft | throughput tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {runtime} | {workload} | {latency_p50:.3f} | {latency_p95:.3f} | {ttft_p50:.3f} | {throughput:.2f} |".format(
                runtime=row["runtime"],
                workload=row["workload"],
                latency_p50=row["latency_p50"],
                latency_p95=row["latency_p95"],
                ttft_p50=row["ttft_p50"],
                throughput=row["throughput"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def load_raw_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def flatten_results(raw_results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in raw_results:
        summary = result["summary"]
        rows.append(
            {
                "runtime": result["runtime"]["name"],
                "workload": result["workload"]["name"],
                "latency_p50": summary["latency_sec"]["p50"],
                "latency_p95": summary["latency_sec"]["p95"],
                "ttft_p50": summary["ttft_sec"]["p50"],
                "throughput": summary["throughput_tokens_per_sec"],
            }
        )
    return rows
