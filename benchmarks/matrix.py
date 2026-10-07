"""
Matrix config: which deployments to benchmark, under which SLO and workload.

A matrix YAML lists deployment *axes*; every list-valued field expands
into a cartesian product, so

    deployments:
      - model: meta-llama/Llama-3.1-8B-Instruct
        engine: [vllm, sglang]
        precision: [bf16, fp8]
        tp: [1, 2]

is eight deployments. Combinations that need more GPUs than the box has
(tp * pp > hardware.gpus) are dropped, not errored, so one matrix can
describe "everything that fits".
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ENGINES = ("vllm", "sglang", "local", "fake", "external")
PRECISIONS = ("bf16", "fp16", "fp8", "auto")


@dataclass(frozen=True)
class SpecDecode:
    method: str                     # "ngram", "eagle", "eagle3", ...
    num_tokens: int = 4             # draft tokens proposed per step
    draft_model: str | None = None  # required for eagle-style methods


@dataclass(frozen=True)
class Deployment:
    model: str
    engine: str
    precision: str = "bf16"
    tp: int = 1
    pp: int = 1
    ep: bool = False                # expert parallelism (MoE models only)
    spec_decode: SpecDecode | None = None
    max_model_len: int | None = None
    extra_args: tuple[str, ...] = ()
    base_url: str | None = None     # engine=external: benchmark a server you started yourself
    name: str = ""

    @property
    def gpus(self) -> int:
        return self.tp * self.pp

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extra_args"] = list(self.extra_args)
        return d


@dataclass(frozen=True)
class Hardware:
    gpu: str
    gpus: int
    usd_per_gpu_hour: float
    interconnect: str = "pcie"      # pcie | nvlink | nvswitch


@dataclass(frozen=True)
class Slo:
    ttft_p99_ms: float
    itl_p99_ms: float
    max_error_rate: float = 0.01


@dataclass(frozen=True)
class Workload:
    prompt_words: tuple[int, int] = (200, 1000)
    output_tokens: tuple[int, int] = (64, 256)
    duration_s: float = 60.0
    seed: int = 0


@dataclass(frozen=True)
class Search:
    start_rate: float = 1.0
    max_rate: float = 64.0
    min_rate: float = 0.05
    rel_tol: float = 0.1            # stop bisecting once hi/lo <= 1 + rel_tol
    max_probes: int = 10


@dataclass
class Matrix:
    name: str
    hardware: Hardware
    slo: Slo
    workload: Workload
    search: Search
    deployments: list[Deployment] = field(default_factory=list)
    port: int = 8000
    results_dir: Path = Path("results/recipes")
    recipes_dir: Path = Path("recipes")


def load_matrix(path: str | Path) -> Matrix:
    raw = yaml.safe_load(Path(path).read_text())
    hardware = Hardware(**raw["hardware"])
    defaults = raw.get("defaults", {})
    deployments: list[Deployment] = []
    for block in raw["deployments"]:
        for combo in expand({**defaults, **block}):
            dep = _deployment(combo)
            if dep.gpus <= hardware.gpus:
                deployments.append(dep)
    names = [d.name for d in deployments]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate deployment names (add an explicit name:): {sorted(dupes)}")
    workload = raw.get("workload", {})
    return Matrix(
        name=raw.get("name", Path(path).stem),
        hardware=hardware,
        slo=Slo(**raw["slo"]),
        workload=Workload(**{k: tuple(v) if isinstance(v, list) else v for k, v in workload.items()}),
        search=Search(**raw.get("search", {})),
        deployments=deployments,
        port=int(raw.get("port", 8000)),
        results_dir=Path(raw.get("results_dir", "results/recipes")),
        recipes_dir=Path(raw.get("recipes_dir", "recipes")),
    )


def expand(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product over every list-valued key (except extra_args)."""
    axes = {k: v for k, v in block.items() if isinstance(v, list) and k != "extra_args"}
    fixed = {k: v for k, v in block.items() if k not in axes}
    if not axes:
        return [dict(fixed)]
    keys = list(axes)
    return [{**fixed, **dict(zip(keys, values))} for values in itertools.product(*(axes[k] for k in keys))]


def _deployment(d: dict[str, Any]) -> Deployment:
    known = {f.name for f in fields(Deployment)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown deployment keys: {sorted(unknown)}")
    d = dict(d)
    spec = d.get("spec_decode")
    if isinstance(spec, dict):
        d["spec_decode"] = SpecDecode(**spec)
    elif spec in (None, False, "off"):
        d["spec_decode"] = None
    d["extra_args"] = tuple(str(a) for a in d.get("extra_args", ()))
    if d["engine"] not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {d['engine']!r}")
    if d.get("precision", "bf16") not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {d['precision']!r}")
    if d["engine"] == "external" and not d.get("base_url"):
        raise ValueError("engine=external needs base_url")
    dep = Deployment(**d)
    if not dep.name:
        dep = Deployment(**{**d, "name": default_name(dep)})
    return dep


def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower()).strip("-")


def default_name(dep: Deployment) -> str:
    """e.g. llama-3.1-8b-instruct__vllm-fp8-tp2-spec-ngram"""
    parts = [dep.engine, dep.precision, f"tp{dep.tp}"]
    if dep.pp > 1:
        parts.append(f"pp{dep.pp}")
    if dep.ep:
        parts.append("ep")
    if dep.spec_decode:
        parts.append(f"spec-{dep.spec_decode.method}")
    return f"{model_slug(dep.model)}__{'-'.join(parts)}"
