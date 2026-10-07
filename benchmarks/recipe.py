"""
An inference recipe: everything needed to reproduce a deployment and
the performance envelope it was measured to deliver.

    model / runtime / precision / topology   -> how to run it
    slo / workload                           -> what it was measured against
    envelope                                 -> what it delivers at that SLO
    quality                                  -> whether the precision is acceptable
    k8s_profile                              -> what a scheduler needs to place it
    provenance                               -> where the numbers came from

One YAML file per deployment, so a recipe can be diffed, reviewed, and
re-qualified on its own when a runtime or driver changes.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .goodput import SearchResult
from .matrix import Deployment, Hardware, Matrix, model_slug
from .stats import git_commit


def build_recipe(m: Matrix, dep: Deployment, result: SearchResult, environment: dict, command: list[str] | None) -> dict[str, Any]:
    best = result.best
    return {
        "name": dep.name,
        "model": {"id": dep.model},
        "runtime": {"engine": dep.engine, **_pick(environment, "engine_version", "cuda"), "command": _portable(command)},
        "precision": dep.precision,
        "topology": {
            "gpu": m.hardware.gpu,
            "gpus": dep.gpus,
            "tp": dep.tp,
            "pp": dep.pp,
            "ep": dep.ep,
            "interconnect": m.hardware.interconnect,
        },
        "spec_decode": asdict(dep.spec_decode) if dep.spec_decode else None,
        "slo": asdict(m.slo),
        "workload": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(m.workload).items()},
        "envelope": envelope(result, dep, m.hardware),
        "quality": {"status": "not_run"},
        "k8s_profile": k8s_profile(dep, m.hardware),
        "provenance": {
            "matrix": m.name,
            "git_commit": git_commit(),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probes": len(result.probes),
            "gpus_seen": environment.get("gpus"),
        },
        "status": "pass" if best else "no_rate_meets_slo",
    }


def envelope(result: SearchResult, dep: Deployment, hw: Hardware) -> dict[str, Any] | None:
    best = result.best
    if best is None:
        return None
    tok_s = best.output_tokens_per_s or 0.0
    usd_per_hour = hw.usd_per_gpu_hour * dep.gpus
    lo, hi = result.bracket
    return {
        "goodput_rps": round(best.offered_rate, 3),
        "goodput_bracket_rps": [round(lo, 3) if lo else None, round(hi, 3) if hi else None],
        "output_tokens_per_s": round(tok_s, 1),
        "output_tokens_per_s_per_gpu": round(tok_s / dep.gpus, 1),
        "ttft_ms": _ms(best.ttft),
        "itl_ms": _ms(best.itl),
        "e2e_ms": _ms(best.e2e),
        "usd_per_1m_output_tokens": round(usd_per_hour / (tok_s * 3600) * 1e6, 4) if tok_s else None,
        "capped_at_max_rate": hi is None,
    }


def k8s_profile(dep: Deployment, hw: Hardware) -> dict[str, Any]:
    """A placement hint for a scheduler, not a full manifest."""
    profile: dict[str, Any] = {
        "resources": {"limits": {"nvidia.com/gpu": dep.gpus}},
        "nodeSelector": {"nvidia.com/gpu.product": hw.gpu},
        "placement": "single-node",
    }
    if dep.gpus > 1:
        # TP all-reduces every layer: keep the GPUs on the fastest shared link
        profile["topology"] = "same-nvlink-domain" if hw.interconnect != "pcie" else "same-pcie-root-complex"
        profile["shm"] = "required (NCCL / worker IPC over /dev/shm)"
    return profile


def recipe_relpath(model: str, name: str) -> Path:
    """recipes/<model-slug>/<config>.yaml, where <config> drops the model prefix from the name."""
    return Path(model_slug(model)) / f"{name.split('__', 1)[-1]}.yaml"


def write_recipe(recipe: dict[str, Any], recipes_dir: Path, dep: Deployment) -> Path:
    path = recipes_dir / recipe_relpath(dep.model, dep.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(recipe, sort_keys=False, width=100))
    return path


def load_recipes(recipes_dir: Path) -> list[dict[str, Any]]:
    return [yaml.safe_load(p.read_text()) for p in sorted(recipes_dir.glob("*/*.yaml"))]


def _portable(command: list[str] | None) -> list[str] | None:
    return [("python" if c == sys.executable else c) for c in command] if command else None


def _ms(stats: dict) -> dict[str, float | None]:
    return {k: (round(stats[k] * 1000, 1) if stats.get(k) is not None else None) for k in ("p50", "p99")}


def _pick(d: dict, *keys: str) -> dict:
    return {k: d[k] for k in keys if d.get(k)}
