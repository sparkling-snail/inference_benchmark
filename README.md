# mini-inference-server

Continuous batching and KV-cache management built from scratch on top
of a small HF model, mostly so I'd actually understand what vLLM,
SGLang, and TGI are doing under the hood instead of just knowing the
vocabulary.

## why

A lot of portfolio projects in this space stop at "I deployed vLLM."
I wanted to hit the actual problems those servers solve myself:
padding waste, cache eviction, deciding who gets into the next batch.
Small model, real tradeoffs.

## how it's laid out

```
Requests -> Queue -> Scheduler -> Batch Executor -> KV Cache Store -> Response
```

The scheduler decides who's in the next batch step. The batch executor
runs one forward pass over whoever's currently active. The KV cache
store holds each sequence's key/value tensors — allocated when a
request is admitted, freed when it finishes.

## build order

**Part A — the engine**

- [x] Phase 1: naive static batching (`src/scheduler_naive.py`). No
  cache, so every step recomputes the whole sequence from scratch, and
  the batch is locked once formed — a request that finishes early just
  sits idle waiting for the slowest one in its batch. This is the
  number everything after it has to beat.
- [x] Phase 2: single-request KV cache (`src/kv_cache_single.py`).
  Verified against HF's own cached `generate()` — token-for-token
  match on all test prompts via `verify_kv_cache.py`.
- [x] Phase 3: batched KV cache (`src/kv_cache_batched.py`). Per-sequence
  cache slots so requests can join or leave a running batch without
  restarting everyone else — this is the part that's actually
  "continuous batching." Verified in two checkpoints via
  `verify_kv_cache_batched.py`: (A) a fixed batch of mixed-length
  prompts decoded together with correct per-row caching, (B) evict and
  admit exercised mid-batch, including both padding directions admit
  needs (new row shorter than the batch, new row longer than the
  batch). Every request checked token-for-token against Phase 2's
  single-sequence reference.
- [ ] Phase 4: scheduler loop. Wrap Phase 3 in something that runs
  continuously against a queue (FCFS to start, can get smarter later).
- [ ] Phase 5: benchmark against vLLM, SGLang, and TGI. Wanted this to
  be a real experiment, not just "run it twice and eyeball the
  numbers":
  - test matrix across model, runtime, concurrency, prompt/output
    length, and traffic pattern (steady-state vs bursty)
  - warm-up kept separate from steady-state measurement, each cell run
    a few times
  - a small OpenAI-compatible streaming endpoint in front of my own
    server, so it gets hit the same way as the others instead of
    calling mine directly as a function while the rest go over HTTP
  - report p50/p95/p99 TTFT, inter-token latency, tokens/sec, error
    rate, and a rough cost-per-1M-output-tokens across naive /
    continuous / vLLM / SGLang / TGI
- [ ] Phase 6: evaluation harness. Speed doesn't mean much if
  correctness quietly breaks to get there, so this runs a small eval
  set alongside every Phase 5 sweep — streaming shouldn't drop or
  reorder tokens, long context shouldn't truncate, tool-call
  formatting shouldn't fall apart when batching changes. Pass/fail
  gates, not a vibe check.

**Part B — making it production, on AWS**

Once the engine's fast and provably still correct, this part is about
not leaving it as a laptop script.

- [ ] Phase 7: Dockerize it, deploy to a single GPU EC2 box
  (g4dn.xlarge to start) behind a health-checked endpoint, same
  streaming API from Phase 5.
- [ ] Phase 8: Prometheus metrics (queue depth, batch size, TTFT/TPOT
  histograms, GPU utilization, error rate) plus a Grafana dashboard
  with real alerts on it.
- [ ] Phase 9: Terraform for the instance, security group, IAM role,
  ECR repo. Reproducible, not click-ops.
- [ ] Phase 10: move to EKS, GPU node group, autoscale off the
  queue-depth metric.
- [ ] Phase 11: CI/CD — build, push to ECR, run the Phase 2/6 gates,
  deploy.
- [ ] Phase 12: rerun the Phase 5 matrix against the live deployment
  under real load, turn it into a cost-per-GPU-hour /
  cost-per-1M-tokens story, with and without autoscaling.

Heads up: phases 7, 10, and 12 cost real money (no free tier for GPU
instances). Plan is to spin things up via Terraform, benchmark, tear
back down — not leave anything running.

## setup

```bash
pip install -r requirements.txt
```

Needs internet access to pull GPT-2 from Hugging Face. (This got
scaffolded somewhere without HF access, so the scheduler logic was
smoke-tested against a tiny untrained model with matching shapes —
logic's verified, but run it for real before trusting any numbers.)

## running phase 1

```bash
python benchmark_naive.py
```

Loads GPT-2, runs 8 prompts through the naive batcher, prints
throughput, average latency, and a few sample outputs. Worth saving
the output — it's the baseline Phase 3/4 need to beat, and I'll want
it for the Phase 5 comparison later.

## running phase 2

```bash
python verify_kv_cache.py
```

Runs the same prompts through the cached implementation and checks the
output matches HF's `generate(use_cache=True)` exactly. Needs to print
all PASS before moving on — a cache bug here (misaligned position ids,
that kind of thing) would otherwise surface much later as subtly wrong
output once requests are being admitted and evicted mid-batch, which
is a far worse place to have to debug it.

## running phase 3

```bash
python verify_kv_cache_batched.py
```

Checkpoint A: a fixed batch of mixed-length prompts, checked against
Phase 2's reference. Checkpoint B: evict + admit exercised mid-batch
(short-prompt admit pads the new row, long-prompt admit pads the
existing rows), every request still checked against its own standalone
reference.

## running phase 4

```bash
python verify_scheduler.py
```

Runs more requests than `max_batch_size` with deliberately uneven
`max_new_tokens`, so some finish early and force a mid-run admit if
continuous batching is actually working. Prints the admit/evict
timeline, batch occupancy, and checks every request against Phase 2's
reference. Needs both the "continuous batching observed" check and
every per-request PASS before trusting Phase 5 numbers against this.

## layout

```
mini-inference-server/
├── README.md
├── requirements.txt
├── benchmark_naive.py            # phase 1 entry point
├── verify_kv_cache.py            # phase 2 correctness check
├── verify_kv_cache_batched.py    # phase 3 correctness check
├── verify_scheduler.py           # phase 4 correctness check
└── src/
    ├── request.py              # request dataclass — id, status, timing
    ├── model_wrapper.py        # HF model/tokenizer wrapper
    ├── scheduler_naive.py      # phase 1
    ├── kv_cache_single.py      # phase 2
    ├── kv_cache_batched.py     # phase 3
    └── scheduler.py            # phase 4
```

## next

Phase 5: benchmark matrix against vLLM/SGLang/TGI — same model, same
workload, p50/p95/p99 TTFT, inter-token latency, throughput, and a
rough cost-per-1M-output-tokens across naive / continuous / vLLM /
SGLang / TGI.
