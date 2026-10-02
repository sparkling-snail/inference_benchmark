from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_latency_and_throughput(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    runtimes = [row["runtime"] for row in rows]
    latency = [row["latency_p50"] for row in rows]
    throughput = [row["throughput"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(runtimes, latency)
    axes[0].set_title("p50 latency (sec)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(runtimes, throughput)
    axes[1].set_title("throughput (tok/s)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_dir / "phase5_summary.png")
    plt.close(fig)
