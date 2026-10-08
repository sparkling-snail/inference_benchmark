"""
Accuracy gate: a faster recipe only counts if its quality is acceptable.

Two steps, deliberately separate:

  1. measure  -- run lm-evaluation-harness against the live server right after
                 its goodput search, and store the scores in the recipe.
  2. gate     -- once every deployment has scores, compare each non-BF16 recipe
                 with the BF16 recipe of the same model and mark it pass/fail.

The gate is a separate pass because FP8 and BF16 recipes are measured in
different server launches, in no particular order. It is also re-runnable on
its own (`python -m benchmarks gate`), so changing the threshold doesn't cost
another GPU session.

Quality does not depend on tensor parallelism, so scores are cached per
(model, engine, precision, speculative decoding) within a run.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .matrix import Deployment, Quality
from .recipe import load_recipes, recipe_relpath

BASELINE_PRECISIONS = ("bf16", "fp16")

# (metric, stderr) keys in lm-eval's results JSON, per task
METRICS = {
    "gsm8k": ("exact_match,strict-match", "exact_match_stderr,strict-match"),
    "mmlu": ("acc,none", "acc_stderr,none"),
}
DEFAULT_METRIC = ("acc,none", "acc_stderr,none")


# --- measuring ---------------------------------------------------------------

def parse_lm_eval(raw: dict[str, Any], tasks: tuple[str, ...]) -> dict[str, dict[str, float | None]]:
    """lm-eval results JSON -> {task: {score, stderr, n}} with scores in [0, 1]."""
    results = raw["results"]
    n_samples = raw.get("n-samples", {})
    out: dict[str, dict[str, float | None]] = {}
    for task in tasks:
        if task not in results:
            raise KeyError(f"task {task!r} missing from lm-eval results (has: {sorted(results)})")
        metric, stderr_key = METRICS.get(task, DEFAULT_METRIC)
        row = results[task]
        if metric not in row:
            raise KeyError(f"{task}: metric {metric!r} missing (has: {sorted(k for k in row if k != 'alias')})")
        se = row.get(stderr_key)
        out[task] = {
            "score": float(row[metric]),
            "stderr": float(se) if isinstance(se, (int, float)) else None,
            "n": (n_samples.get(task) or {}).get("effective"),
        }
    return out


def lm_eval_command(base_url: str, model: str, cfg: Quality, task: str, output_dir: Path) -> list[str]:
    model_args = (f"model={model},base_url={base_url.rstrip('/')}/v1/completions,"
                  f"num_concurrent={cfg.num_concurrent},max_retries=3,tokenized_requests=False")
    cmd = [sys.executable, "-m", "lm_eval", "--model", "local-completions", "--model_args", model_args,
           "--tasks", task, "--output_path", str(output_dir), "--seed", "0"]
    if cfg.limit:
        cmd += ["--limit", str(cfg.limit)]
    if task in cfg.num_fewshot:
        cmd += ["--num_fewshot", str(cfg.num_fewshot[task])]
    return cmd


async def measure_quality(base_url: str, model: str, cfg: Quality, log_path: Path) -> dict[str, Any]:
    """Run each task through lm-eval against the live server."""
    scores: dict[str, dict[str, float | None]] = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log, tempfile.TemporaryDirectory() as tmp:
        for task in cfg.tasks:
            out_dir = Path(tmp) / task
            cmd = lm_eval_command(base_url, model, cfg, task, out_dir)
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=log, stderr=asyncio.subprocess.STDOUT)
            if await proc.wait() != 0:
                raise RuntimeError(f"lm_eval failed on {task} (is it installed? pip install 'lm-eval[api]'); see {log_path}")
            files = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
            if not files:
                raise RuntimeError(f"lm_eval wrote no results for {task}; see {log_path}")
            scores.update(parse_lm_eval(json.loads(files[-1].read_text()), (task,)))
    return {"status": "measured", "evaluator": "lm-eval", "limit": cfg.limit, "tasks": scores}


class QualityCache:
    """Reuse scores across deployments that differ only in parallelism."""

    def __init__(self) -> None:
        self._cache: dict[tuple, dict[str, Any]] = {}

    @staticmethod
    def key(dep: Deployment) -> tuple:
        return (dep.model, dep.engine, dep.precision, dep.spec_decode)

    def get(self, dep: Deployment) -> dict[str, Any] | None:
        return self._cache.get(self.key(dep))

    def put(self, dep: Deployment, quality: dict[str, Any]) -> None:
        self._cache[self.key(dep)] = quality


# --- gating ------------------------------------------------------------------

def apply_gate(recipes: list[dict[str, Any]], max_drop_pts: float) -> dict[str, dict[str, Any]]:
    """
    Decide pass/fail for every recipe that has scores. Pure: returns
    {recipe name: new quality dict}, only for recipes whose status changes.

    - A BF16/FP16 recipe is a baseline: status "baseline".
    - Anything else is compared with the BF16 recipe of the same model
      (same engine preferred, otherwise the highest-scoring one) and fails if
      any task drops by more than max_drop_pts percentage points.
    - No baseline in the set: "no_baseline" (not a pass).
    - noise_warning: the drop is within 2 standard errors of the combined
      noise, so a pass/fail at this threshold isn't statistically meaningful;
      raise `limit` or rerun on the full task.
    """
    scored = [r for r in recipes if _tasks(r)]
    updates: dict[str, dict[str, Any]] = {}
    for r in scored:
        q = {k: v for k, v in r["quality"].items() if k not in ("baseline", "drop_pts", "max_drop_pts", "noise_warning")}
        if r["precision"] in BASELINE_PRECISIONS:
            q["status"] = "baseline"
            updates[r["name"]] = q
            continue
        base = _pick_baseline(r, scored)
        if base is None:
            q["status"] = "no_baseline"
            updates[r["name"]] = q
            continue
        drops, noisy = {}, False
        for task, cur in _tasks(r).items():
            ref = _tasks(base).get(task)
            if ref is None:
                continue
            drops[task] = round((ref["score"] - cur["score"]) * 100, 2)
            se = math.hypot(cur.get("stderr") or 0.0, ref.get("stderr") or 0.0) * 100
            noisy = noisy or 2 * se > max_drop_pts
        q.update(
            status="fail" if any(d > max_drop_pts for d in drops.values()) else "pass",
            baseline=base["name"],
            drop_pts=drops,
            max_drop_pts=max_drop_pts,
        )
        if noisy:
            q["noise_warning"] = True
        updates[r["name"]] = q
    return updates


def gate_recipes_dir(recipes_dir: Path, max_drop_pts: float) -> dict[str, str]:
    """Load recipes, apply the gate, rewrite the changed files. Returns {name: status}."""
    recipes = load_recipes(recipes_dir)
    updates = apply_gate(recipes, max_drop_pts)
    for r in recipes:
        if r["name"] in updates:
            r["quality"] = updates[r["name"]]
            path = recipes_dir / recipe_relpath(r["model"]["id"], r["name"])
            path.write_text(yaml.safe_dump(r, sort_keys=False, width=100))
    return {name: q["status"] for name, q in updates.items()}


def _tasks(r: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return (r.get("quality") or {}).get("tasks") or {}


def _pick_baseline(r: dict[str, Any], scored: list[dict[str, Any]]) -> dict[str, Any] | None:
    bases = [b for b in scored if b["model"]["id"] == r["model"]["id"] and b["precision"] in BASELINE_PRECISIONS]
    if not bases:
        return None
    same_engine = [b for b in bases if b["runtime"]["engine"] == r["runtime"]["engine"]]
    pool = same_engine or bases
    return max(pool, key=lambda b: sum(t["score"] for t in _tasks(b).values()))
