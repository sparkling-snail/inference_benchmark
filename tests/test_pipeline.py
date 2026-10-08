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
    assert len(load_matrix("configs/a100x8.yaml").deployments) == 25
    assert len(load_matrix("configs/a100x8_quick.yaml").deployments) == 6
    assert load_matrix("configs/smoke_quality.yaml").quality.max_drop_pts == 1.0
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


# --- accuracy gate ----------------------------------------------------------

from benchmarks.matrix import Quality  # noqa: E402
from benchmarks.quality import QualityCache, apply_gate, gate_recipes_dir, lm_eval_command, parse_lm_eval  # noqa: E402
from benchmarks.recipe import recipe_relpath  # noqa: E402
from benchmarks.report import render  # noqa: E402

LM_EVAL_JSON = {
    "results": {
        "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.80, "exact_match_stderr,strict-match": 0.025,
                  "exact_match,flexible-extract": 0.82, "exact_match_stderr,flexible-extract": 0.024},
        "mmlu": {"alias": "mmlu", "acc,none": 0.65, "acc_stderr,none": 0.004},
    },
    "n-samples": {"gsm8k": {"original": 1319, "effective": 250}},
}


def _recipe(name, precision, scores, engine="vllm", model="org/M", se=0.001):
    tasks = {t: {"score": s, "stderr": se, "n": 250} for t, s in scores.items()} if scores else {}
    return {"name": name, "model": {"id": model}, "precision": precision, "runtime": {"engine": engine},
            "quality": {"status": "measured", "tasks": tasks} if scores else {"status": "not_run"}}


def test_parse_lm_eval_picks_strict_match_and_stderr():
    out = parse_lm_eval(LM_EVAL_JSON, ("gsm8k", "mmlu"))
    assert out["gsm8k"] == {"score": 0.80, "stderr": 0.025, "n": 250}
    assert out["mmlu"]["score"] == 0.65 and out["mmlu"]["n"] is None
    with pytest.raises(KeyError, match="missing"):
        parse_lm_eval(LM_EVAL_JSON, ("hellaswag",))


def test_lm_eval_command():
    cmd = lm_eval_command("http://127.0.0.1:8000/", "org/M", Quality(limit=100), "gsm8k", "/tmp/out")
    assert cmd[cmd.index("--limit") + 1] == "100" and cmd[cmd.index("--num_fewshot") + 1] == "5"
    args = cmd[cmd.index("--model_args") + 1]
    assert "model=org/M" in args and "base_url=http://127.0.0.1:8000/v1/completions" in args
    assert "--limit" not in lm_eval_command("http://x", "m", Quality(limit=None), "gsm8k", "/tmp/o")


def test_gate_pass_fail_and_baseline():
    recipes = [
        _recipe("m__vllm-bf16-tp1", "bf16", {"gsm8k": 0.80, "mmlu": 0.65}),
        _recipe("m__vllm-fp8-tp1", "fp8", {"gsm8k": 0.795, "mmlu": 0.648}),    # -0.5, -0.2 pts: pass
        _recipe("m__vllm-fp8-tp2", "fp8", {"gsm8k": 0.77, "mmlu": 0.65}),      # -3.0 pts on gsm8k: fail
        _recipe("m__vllm-fp8-tp4", "fp8", {"gsm8k": 0.82, "mmlu": 0.66}),      # better than baseline: pass
    ]
    u = apply_gate(recipes, max_drop_pts=1.0)
    assert u["m__vllm-bf16-tp1"]["status"] == "baseline"
    assert u["m__vllm-fp8-tp1"]["status"] == "pass"
    assert u["m__vllm-fp8-tp2"]["status"] == "fail"
    assert u["m__vllm-fp8-tp2"]["drop_pts"]["gsm8k"] == pytest.approx(3.0)
    assert u["m__vllm-fp8-tp2"]["baseline"] == "m__vllm-bf16-tp1"
    assert u["m__vllm-fp8-tp4"]["status"] == "pass" and u["m__vllm-fp8-tp4"]["drop_pts"]["gsm8k"] < 0


def test_gate_baseline_selection_and_missing():
    other = _recipe("m__sglang-bf16-tp1", "bf16", {"gsm8k": 0.90}, engine="sglang")
    same = _recipe("m__vllm-bf16-tp1", "bf16", {"gsm8k": 0.80}, engine="vllm")
    fp8 = _recipe("m__vllm-fp8-tp1", "fp8", {"gsm8k": 0.795})
    assert apply_gate([other, same, fp8], 1.0)["m__vllm-fp8-tp1"]["baseline"] == "m__vllm-bf16-tp1"  # same engine wins
    assert apply_gate([other, fp8], 1.0)["m__vllm-fp8-tp1"]["baseline"] == "m__sglang-bf16-tp1"      # else any BF16
    lone = apply_gate([fp8, _recipe("x__vllm-bf16", "bf16", {"gsm8k": 0.8}, model="org/Other")], 1.0)
    assert lone["m__vllm-fp8-tp1"]["status"] == "no_baseline"
    assert "m__unscored" not in apply_gate([_recipe("m__unscored", "fp8", None)], 1.0)


def test_gate_flags_noise_and_is_rerunnable():
    noisy = [_recipe("b", "bf16", {"gsm8k": 0.80}, se=0.025), _recipe("f", "fp8", {"gsm8k": 0.795}, se=0.025)]
    assert apply_gate(noisy, 1.0)["f"]["noise_warning"] is True        # +-7 pts of noise can't resolve a 1 pt gate
    tight = [_recipe("b", "bf16", {"gsm8k": 0.80}, se=0.001), _recipe("f", "fp8", {"gsm8k": 0.795}, se=0.001)]
    assert "noise_warning" not in apply_gate(tight, 1.0)["f"]
    # tightening the threshold flips the verdict without remeasuring
    assert apply_gate(tight, 1.0)["f"]["status"] == "pass"
    assert apply_gate(tight, 0.1)["f"]["status"] == "fail"


def test_gate_recipes_dir_roundtrip_and_report(tmp_path):
    import yaml
    recipes = [
        {**_recipe("m__vllm-bf16-tp1", "bf16", {"gsm8k": 0.80}), "slo": {"ttft_p99_ms": 1, "itl_p99_ms": 1, "max_error_rate": 0.01},
         "topology": {"gpus": 1}, "envelope": None},
        {**_recipe("m__vllm-fp8-tp1", "fp8", {"gsm8k": 0.70}), "slo": {"ttft_p99_ms": 1, "itl_p99_ms": 1, "max_error_rate": 0.01},
         "topology": {"gpus": 1}, "envelope": None},
    ]
    for r in recipes:
        path = tmp_path / recipe_relpath("org/M", r["name"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(r))
    assert gate_recipes_dir(tmp_path, 1.0) == {"m__vllm-bf16-tp1": "baseline", "m__vllm-fp8-tp1": "fail"}
    from benchmarks.recipe import load_recipes
    md = render(load_recipes(tmp_path))
    assert "Excluded by the accuracy gate" in md and "gsm8k -10.0 pts" in md
    ranked = md.split("Excluded")[0]
    assert "vllm-bf16-tp1" in ranked and "vllm-fp8-tp1" not in ranked


def test_quality_cache_ignores_parallelism():
    cache = QualityCache()
    cache.put(Deployment(model="m", engine="vllm", precision="fp8", tp=1), {"status": "measured"})
    assert cache.get(Deployment(model="m", engine="vllm", precision="fp8", tp=4)) is not None
    assert cache.get(Deployment(model="m", engine="vllm", precision="bf16", tp=1)) is None
    assert cache.get(Deployment(model="m", engine="sglang", precision="fp8", tp=1)) is None


def test_measure_quality_runs_harness_subprocess(tmp_path, monkeypatch):
    """measure -> subprocess -> lm-eval results file -> parsed scores, using the stub harness."""
    from benchmarks.quality import measure_quality
    monkeypatch.setenv("PYTHONPATH", "tests/stubs")
    q = asyncio.run(measure_quality("http://127.0.0.1:1", "org/M", Quality(limit=100), tmp_path / "q.log"))
    assert q["status"] == "measured" and q["limit"] == 100
    assert q["tasks"]["gsm8k"] == {"score": 0.80, "stderr": 0.001, "n": 100}
    assert q["tasks"]["mmlu"]["score"] == 0.65
    assert "--tasks gsm8k" in (tmp_path / "q.log").read_text()

    monkeypatch.setenv("STUB_LM_EVAL_FAIL", "1")
    with pytest.raises(RuntimeError, match="lm_eval failed on gsm8k"):
        asyncio.run(measure_quality("http://127.0.0.1:1", "org/M", Quality(), tmp_path / "q2.log"))


def test_engine_binaries_overridable(monkeypatch):
    monkeypatch.setenv("VLLM_BIN", "/opt/vllm-env/bin/vllm")
    monkeypatch.setenv("SGLANG_PYTHON", "/opt/sglang-env/bin/python")
    assert build_command(Deployment(model="m", engine="vllm"), 1)[0] == "/opt/vllm-env/bin/vllm"
    assert build_command(Deployment(model="m", engine="sglang"), 1)[0] == "/opt/sglang-env/bin/python"
