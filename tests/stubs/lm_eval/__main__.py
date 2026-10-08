"""
Stand-in for `python -m lm_eval`, for tests and CI only. NOT an evaluator.

Accepts the flags benchmarks/quality.py passes, makes no requests, and writes a
results file in lm-eval's layout (<output_path>/<model>/results_<ts>.json)
with fixed scores, so the measure -> store -> gate -> report path can be
exercised without GPUs or the real harness.

Put tests/stubs on PYTHONPATH to use it. Set STUB_LM_EVAL_FAIL=1 to simulate a crash.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCORES = {"gsm8k": ("exact_match,strict-match", "exact_match_stderr,strict-match", 0.80),
          "mmlu": ("acc,none", "acc_stderr,none", 0.65)}

ap = argparse.ArgumentParser()
for flag in ("--model", "--model_args", "--tasks", "--output_path", "--limit", "--num_fewshot", "--seed"):
    ap.add_argument(flag)
args = ap.parse_args()

if os.environ.get("STUB_LM_EVAL_FAIL"):
    sys.exit("stub lm_eval: simulated failure")

results = {}
for task in args.tasks.split(","):
    metric, stderr, score = SCORES[task]
    results[task] = {"alias": task, metric: score, stderr: 0.001}
out = Path(args.output_path) / "stub-model"
out.mkdir(parents=True, exist_ok=True)
(out / f"results_{int(time.time())}.json").write_text(json.dumps(
    {"results": results, "n-samples": {t: {"original": 100, "effective": int(args.limit or 100)} for t in results}}))
