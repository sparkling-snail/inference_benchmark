# Tail-latency experiments

Four experiments behind the post *"Where p99 comes from in LLM serving"*.
Each one isolates one mechanism that creates tail latency, so its
fingerprint (TTFT vs inter-token latency vs end-to-end) is visible on
its own.

| # | Mechanism | Runs against | Script |
|---|---|---|---|
| 1 | Queueing near saturation (the load cliff) | vLLM | `exp1_load_sweep.py` |
| 2 | Prefill/decode interference (long-prompt stall) | vLLM, two token budgets | `exp2_prefill_stall.py` |
| 3 | KV-cache pressure and preemption | vLLM, shrinking KV cache | `exp3_kv_pressure.py` |
| 4 | Head-of-line blocking vs admission policy | **this repo's engine** | `exp4_hol_blocking.py` |

All scripts write raw per-request (and for exp 2, per-token) data to
`results/tail/*.json` with the git commit and config, so charts can be
regenerated or re-cut without rerunning.

## Exp 4: on your laptop, no GPU needed

Uses the engine's continuous-batching scheduler with its pluggable admission
policy (`fcfs`, `sjf`, `sjf_aging` in `engine/scheduler.py`). The same Poisson
arrival trace is replayed for each policy.

```bash
python -m engine.verify.scheduler               # correctness, for fcfs AND sjf
python -m experiments.tail.exp4_hol_blocking    # ~10-20 min on CPU with gpt2
python -m experiments.tail.plot
```

The script calibrates capacity first and runs at `--load 0.85` of it.
If the "mean batch occupancy" it prints is low (well under ~80%), the
engine isn't saturated and there's no queue to reorder: raise `--load`
(e.g. 1.0-1.1) until it is. That's also a finding worth one sentence in
the post: scheduling policy only matters when there's a queue.

## Exp 1-3: on one rented NVIDIA GPU

An L4 or A10G for an afternoon is plenty with the default 1.5B model.

```bash
pip install -r requirements.txt vllm
bash experiments/tail/run_vllm_experiments.sh
```

The runner restarts vLLM with the right flags for each config, waits
for `/health`, runs the experiment, and draws the charts into
`results/tail/figs-light/` and `figs-dark/`. Server logs are in
`results/tail/logs/` -- for exp 3, grep them for "preempt".

You can also point any single script at an already-running server:

```bash
python -m experiments.tail.exp1_load_sweep --base-url http://localhost:8000 --rates 2 4 8 16
```

They work against any OpenAI-compatible server, including this repo's
`engine/server.py`.

## Testing the harness without a GPU

`tests/fake_vllm_server.py` is a toy server that mimics vLLM's scheduling
(token budget, chunked prefill, KV capacity, preemption, `/metrics`).
Its numbers are meaningless; it only proves the harness works end to end:

```bash
python tests/fake_vllm_server.py --port 8000 --budget 512 &
python -m experiments.tail.exp2_prefill_stall --label fake-512 --background-tokens 200
```

## What each chart should show

- **exp1_load_cliff.png**: p50 TTFT nearly flat while p99 bends upward
  well before throughput peaks. Plan capacity at the p99 knee.
- **exp2_prefill_stall.png**: with a large per-step budget, background
  streams freeze for one long gap when a long prompt arrives; with a
  small budget the gap shrinks but the long prompt's own TTFT grows.
  That's the tradeoff, not a free win.
- **exp3_kv_pressure.png**: preemptions appear once the cache is too
  small, and p99 end-to-end climbs much faster than p50.
- **exp4_hol_blocking.png**: SJF cuts short requests' p99 TTFT but
  raises long requests' p99; aging bounds the damage.
