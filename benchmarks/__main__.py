"""
Inference recipe pipeline.

    python -m benchmarks plan   configs/l4x4.yaml           # list deployments + server commands
    python -m benchmarks run    configs/l4x4.yaml           # launch, search goodput, write recipes
    python -m benchmarks run    configs/l4x4.yaml --only fp8 --skip-existing
    python -m benchmarks gate   --max-drop-pts 1.0          # re-apply the accuracy gate to stored scores
    python -m benchmarks report --recipes-dir recipes       # regenerate the recipe table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from .goodput import Probe, build_trace, http_measure, search_goodput
from .launcher import build_command, launch
from .loadgen import run_open_loop
from .matrix import Deployment, Matrix, load_matrix
from .quality import QualityCache, gate_recipes_dir, measure_quality
from .recipe import build_recipe, recipe_relpath, write_recipe
from .report import write_report
from .stats import fmt_ms


def cmd_plan(m: Matrix) -> None:
    print(f"matrix {m.name}: {len(m.deployments)} deployments on {m.hardware.gpus}x {m.hardware.gpu}")
    print(f"SLO: p99 TTFT <= {m.slo.ttft_p99_ms:g} ms, p99 ITL <= {m.slo.itl_p99_ms:g} ms\n")
    for dep in m.deployments:
        cmd = build_command(dep, m.port)
        print(f"  {dep.name}  [{dep.gpus} GPU]")
        print(f"    $ {' '.join(cmd) if cmd else dep.base_url}")


async def run_one(m: Matrix, dep: Deployment, cache: QualityCache) -> Path:
    raw_dir = m.results_dir / m.name
    print(f"\n=== {dep.name} ({dep.gpus} GPU)")
    async with launch(dep, m.port, log_path=raw_dir / "logs" / f"{dep.name}.log") as server:
        model = dep.model

        # warm-up: CUDA graphs, allocator, compile caches -- never part of a probe
        await run_open_loop(server.base_url, model, build_trace(m.search.start_rate, m.workload, seed=10_000)[:10])

        def log(p: Probe) -> None:
            mark = "PASS" if p.passed else f"fail ({', '.join(p.failed_on)})"
            print(f"  {p.offered_rate:7.2f} req/s | TTFT p99 {fmt_ms(p.ttft.get('p99'))} ms"
                  f" | ITL p99 {fmt_ms(p.itl.get('p99'))} ms | {p.output_tokens_per_s or 0:7.0f} tok/s | {mark}")

        result = await search_goodput(http_measure(server.base_url, model, m.workload, m.slo, log), m.search)
        recipe = build_recipe(m, dep, result, server.environment, server.command)

        # only worth scoring a deployment that can actually meet the SLO
        if m.quality and result.best:
            quality = cache.get(dep)
            if quality is None:
                print(f"  quality: {', '.join(m.quality.tasks)} (limit {m.quality.limit})")
                try:
                    quality = await measure_quality(server.base_url, model, m.quality, raw_dir / "logs" / f"{dep.name}.quality.log")
                    cache.put(dep, quality)
                except Exception as exc:  # keep the performance measurement; scoring can be redone
                    print(f"  !! quality failed: {exc}")
                    quality = {"status": "error", "error": str(exc)}
            else:
                print("  quality: reusing scores from a deployment with the same precision")
            recipe["quality"] = quality

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{dep.name}.json").write_text(json.dumps(
        {"deployment": dep.to_dict(), "environment": server.environment,
         "probes": [asdict(p) for p in result.probes], "recipe": recipe}, indent=2))
    path = write_recipe(recipe, m.recipes_dir, dep)
    env = recipe["envelope"]
    summary = (f"goodput {env['goodput_rps']} req/s, ${env['usd_per_1m_output_tokens']}/1M tok"
               if env else "no rate met the SLO")
    print(f"  -> {path}: {summary}")
    return path


async def cmd_run(m: Matrix, only: list[str], skip_existing: bool) -> int:
    deps = [d for d in m.deployments if not only or any(o in d.name for o in only)]
    failures = []
    cache = QualityCache()
    for dep in deps:
        if skip_existing and (m.recipes_dir / recipe_relpath(dep.model, dep.name)).exists():
            print(f"skip {dep.name} (recipe exists)")
            continue
        try:
            await run_one(m, dep, cache)
        except Exception as exc:  # one broken config must not cost the rest of the GPU rental
            failures.append(dep.name)
            print(f"  !! {dep.name} failed: {exc}")
            traceback.print_exc(limit=2)
    if m.quality:
        print_gate(gate_recipes_dir(m.recipes_dir, m.quality.max_drop_pts))
    report = write_report(m.recipes_dir)
    print(f"\n{len(deps) - len(failures)}/{len(deps)} deployments done. Table: {report}")
    if failures:
        print("failed:", *failures, sep="\n  ")
    return 1 if failures else 0


def print_gate(statuses: dict[str, str]) -> None:
    if statuses:
        print("\nquality gate:")
        for name, status in sorted(statuses.items()):
            print(f"  {status:12s} {name}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # show progress live even when piped into tee
    ap = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="list deployments and the server command for each")
    p_plan.add_argument("matrix", type=Path)
    p_run = sub.add_parser("run", help="benchmark every deployment and write recipes")
    p_run.add_argument("matrix", type=Path)
    p_run.add_argument("--only", nargs="*", default=[], help="substrings of deployment names to run")
    p_run.add_argument("--skip-existing", action="store_true", help="resume: skip deployments with a recipe")
    p_gate = sub.add_parser("gate", help="re-apply the accuracy gate to scores already stored in recipes")
    p_gate.add_argument("--recipes-dir", type=Path, default=Path("recipes"))
    p_gate.add_argument("--max-drop-pts", type=float, default=1.0)
    p_rep = sub.add_parser("report", help="regenerate the recipe table")
    p_rep.add_argument("--recipes-dir", type=Path, default=Path("recipes"))
    args = ap.parse_args()

    if args.cmd == "gate":
        print_gate(gate_recipes_dir(args.recipes_dir, args.max_drop_pts))
        print(write_report(args.recipes_dir))
        return
    if args.cmd == "report":
        print(write_report(args.recipes_dir))
        return
    m = load_matrix(args.matrix)
    if args.cmd == "plan":
        cmd_plan(m)
    else:
        sys.exit(asyncio.run(cmd_run(m, args.only, args.skip_existing)))


if __name__ == "__main__":
    main()
