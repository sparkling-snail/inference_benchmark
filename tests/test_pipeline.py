"""Unit tests for the recipe pipeline. No server, no GPU: python -m pytest tests/"""

from __future__ import annotations

import asyncio
import json
import textwrap

import pytest

from benchmarks.goodput import Probe, check_slo, search_goodput
from benchmarks.launcher import build_command
from benchmarks.matrix import Deployment, Hardware, Search, Slo, SpecDecode, load_matrix
from benchmarks.recipe import envelope, k8s_profile
from benchmarks.goodput import SearchResult


def _matrix(tmp_path, deployments: str):
    path = tmp_path / "m.yaml"
    path.write_text(textwrap.dedent("""
        name: t
        hardware: {gpu: L4, gpus: 4, usd_per_gpu_hour: 1.0}
        slo: {ttft_p99_ms: 1000, itl_p99_ms: 100}
        defaults: {max_model_len: 4096}
        deployments:
    """) + textwrap.indent(textwrap.dedent(deployments), "  "))
    return load_matrix(path)


# --- matrix -----------------------------------------------------------------

def test_matrix_expands_axes_and_drops_what_does_not_fit(tmp_path):
    m = _matrix(tmp_path, """
        - model: org/Model-8B
          engine: [vllm, sglang]
          precision: [bf16, fp8]
          tp: [1, 2, 4, 8]
    """)
    assert len(m.deployments) == 2 * 2 * 3  # tp=8 needs 8 GPUs, box has 4
    assert {d.max_model_len for d in m.deployments} == {4096}
    assert "model-8b__vllm-fp8-tp2" in {d.name for d in m.deployments}


def test_matrix_spec_decode_and_duplicate_names(tmp_path):
    m = _matrix(tmp_path, """
        - model: org/M
          engine: vllm
          spec_decode: [null, {method: ngram, num_tokens: 3}]
    """)
    assert [d.spec_decode for d in m.deployments] == [None, SpecDecode("ngram", 3)]
    assert m.deployments[1].name.endswith("-spec-ngram")
    with pytest.raises(ValueError, match="duplicate"):
        _matrix(tmp_path, """
            - {model: org/M, engine: vllm}
            - {model: org/M, engine: vllm}
        """)


def test_repo_matrices_load():
    assert len(load_matrix("configs/l4x4.yaml").deployments) == 30
    load_matrix("configs/smoke.yaml")
    load_matrix("configs/engine_local.yaml")


# --- launcher ---------------------------------------------------------------

def test_vllm_command():
    dep = Deployment(model="org/M", engine="vllm", precision="fp8", tp=2, pp=2, ep=True,
                     spec_decode=SpecDecode("ngram", 4), extra_args=("--enforce-eager",))
    cmd = build_command(dep, 9000)
    assert cmd[:3] == ["vllm", "serve", "org/M"]
    joined = " ".join(cmd)
    for flag in ("--tensor-parallel-size 2", "--pipeline-parallel-size 2", "--enable-expert-parallel",
                 "--quantization fp8", "--port 9000"):
        assert flag in joined
    spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert spec["method"] == "ngram" and spec["num_speculative_tokens"] == 4
    assert cmd[-1] == "--enforce-eager"


def test_sglang_and_external_commands():
    cmd = " ".join(build_command(Deployment(model="org/M", engine="sglang", precision="fp8", tp=4, ep=True), 9000))
    assert "--tp-size 4" in cmd and "--ep-size 4" in cmd and "--quantization fp8" in cmd
    assert build_command(Deployment(model="m", engine="external", base_url="http://x"), 9000) is None


# --- goodput search ---------------------------------------------------------

def _fake_measure(capacity: float):
    """A server that meets the SLO at any rate <= capacity."""
    async def measure(rate: float) -> Probe:
        return Probe(offered_rate=rate, passed=rate <= capacity, output_tokens_per_s=rate * 100)
    return measure


@pytest.mark.parametrize("capacity", [0.3, 1.0, 5.0, 13.7, 40.0])
def test_search_brackets_capacity(capacity):
    cfg = Search(start_rate=1, max_rate=64, rel_tol=0.1, max_probes=20)
    res = asyncio.run(search_goodput(_fake_measure(capacity), cfg))
    lo, hi = res.bracket
    assert res.best is not None and lo <= capacity < hi
    assert hi / lo <= 1 + cfg.rel_tol


def test_search_edges():
    cfg = Search(start_rate=1, max_rate=8, min_rate=0.1, max_probes=20)
    capped = asyncio.run(search_goodput(_fake_measure(100), cfg))
    assert capped.best.offered_rate == 8 and capped.bracket[1] is None
    never = asyncio.run(search_goodput(_fake_measure(0.01), cfg))
    assert never.best is None and never.probes[-1].offered_rate == pytest.approx(0.1)


def test_search_respects_probe_budget():
    cfg = Search(start_rate=1, max_rate=1e6, rel_tol=0.001, max_probes=5)
    assert len(asyncio.run(search_goodput(_fake_measure(1e5), cfg)).probes) == 5


def test_check_slo():
    slo = Slo(ttft_p99_ms=500, itl_p99_ms=50, max_error_rate=0.05)
    ok = Probe(offered_rate=1, passed=False, n=100, errors=1, ttft={"p99": 0.4}, itl={"p99": 0.04})
    assert check_slo(ok, slo) == []
    bad = Probe(offered_rate=1, passed=False, n=100, errors=10, ttft={"p99": 0.6}, itl={"p99": 0.06})
    assert check_slo(bad, slo) == ["errors", "ttft_p99", "itl_p99"]


# --- recipe -----------------------------------------------------------------

def test_envelope_cost_math():
    dep = Deployment(model="m", engine="vllm", tp=2)
    best = Probe(offered_rate=4, passed=True, output_tokens_per_s=1000.0,
                 ttft={"p50": 0.1, "p99": 0.5}, itl={"p50": 0.02, "p99": 0.05}, e2e={})
    env = envelope(SearchResult(best=best, probes=[best], bracket=(4, 5)), dep, Hardware("L4", 4, 1.8))
    # 2 GPUs * $1.8/h = $3.6/h for 3.6M tokens/h -> $1 per 1M tokens
    assert env["usd_per_1m_output_tokens"] == pytest.approx(1.0)
    assert env["output_tokens_per_s_per_gpu"] == 500
    assert env["ttft_ms"] == {"p50": 100.0, "p99": 500.0}


def test_k8s_profile_topology():
    hw = Hardware("L4", 4, 1.0, interconnect="pcie")
    assert "topology" not in k8s_profile(Deployment(model="m", engine="vllm"), hw)
    multi = k8s_profile(Deployment(model="m", engine="vllm", tp=2), hw)
    assert multi["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert multi["topology"] == "same-pcie-root-complex"
