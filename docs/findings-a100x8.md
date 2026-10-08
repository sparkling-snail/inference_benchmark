# Findings: first GPU run on 8× A100 40GB

A one-hour cut of the recipe study ([`configs/a100x8_quick.yaml`](../configs/a100x8_quick.yaml)),
run on 2026-10-08. Six vLLM deployments of Qwen2.5-7B and Qwen2.5-72B, each
measured as **SLO goodput**: the highest Poisson arrival rate at which p99
time-to-first-token (TTFT) stayed under 1 s, p99 inter-token latency (ITL)
under 100 ms, and errors under 1%.

Recipe files: [`recipes/`](../recipes/README.md). Raw per-probe data, server
logs and the exact package versions: [`results/recipes/a100x8_quick/`](../results/recipes/a100x8_quick/).

## Setup

| | |
|---|---|
| Hardware | 8× NVIDIA A100-SXM4-40GB, NVSwitch (driver 580.105) |
| Software | vLLM 0.31.0, torch 2.13.0, lm-eval 0.4.13 ([full list](../results/recipes/a100x8_quick/environment.txt)) |
| Models | Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct (vLLM defaults, `max_model_len` 4096) |
| Workload | Synthetic prompts of ~250–1,300 tokens, 64–256 output tokens, open-loop Poisson arrivals |
| Search | 30 s probes; rate doubled until the SLO breaks, then bisected to within ~20% |
| Quality | GSM8K, 250 examples, 5-shot, through the same live server |

## Results

| Deployment | GPUs | Goodput (req/s) | Output tok/s per GPU | p50 / p99 ITL at goodput (ms) | GSM8K |
|---|---:|---:|---:|---|---|
| 7B BF16, TP 2 | 2 | **22.6** | **1,666** | 14 / 90 | 81.2% |
| 7B BF16, TP 1 | 1 | 9.5 | 1,286 | 14 / 97 | 78.4% |
| 7B BF16 + n-gram spec decode, TP 1 | 1 | 8.0 | 1,148 | 22 / 100 | 77.6% |
| 7B weight-only FP8, TP 1 | 1 | 6.7 | 939 | 8 / 82 | 76.0% (inconclusive vs BF16) |
| 72B BF16, TP 8 | 8 | **0.84** | **16.3** | 21 / 96 | 92.8% |
| 72B weight-only FP8, TP 4 | 4 | 0.35 | 12.7 | 20 / 22 | 93.2% (inconclusive vs BF16) |

## What the data says

### 1. The p99 inter-token latency target sets the limit, not TTFT

In five of the six deployments, the probe just above the goodput edge failed on
ITL alone, with TTFT well inside its target. The exception, 7B TP 2, broke both.
The ITL tail is 5–7× the median:

- 7B BF16 TP 1 at 9.5 req/s: ITL p50 14 ms, p99 97 ms.
- 72B BF16 TP 8 at 1 req/s: ITL p50 21 ms, p99 102 ms (failed), with TTFT p99 only 310 ms.

A tail that far from the median usually means some decode steps share an
iteration with a long prefill, so every running stream stalls for that step.
That's the mechanism isolated in [tail experiment 2](../experiments/tail/README.md).
This run didn't test it directly, but it points at the next lever: vLLM's
per-step token budget (`--max-num-batched-tokens`), not more GPUs.

**Why the SLO choice matters for capacity:** 72B on TP 8 delivered 575 output
tok/s at 4 req/s with ITL p99 of 140 ms, but only 130 tok/s at the 0.84 req/s
that meets a 100 ms target. The ITL target costs about 4× in throughput here, a
bigger effect than any configuration change in this run.

### 2. For the 7B model, TP 2 was more efficient per GPU than TP 1

TP 2 sustained 2.4× the goodput of TP 1 on 2× the GPUs, so about 30% more
output per GPU. This holds even comparing TP 2's lowest bracket (22.6 req/s, so
11.3 per GPU) with TP 1's highest (under 11.3 req/s).

Two visible reasons:
- **Faster decode steps:** median ITL was 8–9 ms at TP 2 against 13–14 ms at
  TP 1, at the same load.
- **More KV cache:** splitting the weights over two GPUs left room for 1.06M
  cached tokens, 2.7× TP 1's 395k, so larger batches fit before requests queue.

With a tight ITL target, the latency headroom matters more than the cost of
the extra all-reduce. A looser target, or a throughput-only goal, could reverse this.

### 3. Weight-only FP8 on A100: faster median, worse tail, lower goodput

The A100 has no FP8 tensor cores, so vLLM stores the weights in FP8 and
converts them to BF16 for each matrix multiply.

- **7B, 1 GPU:** median ITL dropped about 40% (8 ms against 13–14 ms) and the
  KV cache grew 25% (494k against 395k tokens). But TTFT rose and the ITL tail
  broke the target sooner: 121 ms at 8 req/s, where BF16 was at 88 ms. Goodput
  fell 29% (6.7 against 9.5 req/s).
- **72B:** FP8 fits on 4 GPUs instead of 8, with the same median ITL (20–21 ms).
  But its ITL tail broke the target at 0.5 req/s (146 ms), so goodput per GPU
  is 22% lower (12.7 against 16.3 tok/s).

This fits weight-only kernels helping memory-bound decode but adding overhead to
compute-bound prefill, where the long prompts land. That's an interpretation,
not something this run isolated. **On Ampere, weight-only FP8 bought memory
(the same model on fewer GPUs), not efficiency at this SLO.** Native FP8 on
Hopper or Ada is a different question and still open.

### 4. 250 GSM8K examples can't resolve a 2-point quality gate

Three BF16 runs of the same 7B weights scored 77.6%, 78.4% and 81.2%, a
3.6-point spread from noise alone. The run with n-gram speculative decoding,
which shouldn't change the outputs at all, still differed by 0.8 points.

So both FP8 verdicts are **inconclusive**, not pass or fail:
- 7B FP8 against 7B BF16 TP 1: −2.4 ± 7.5 points.
- 72B FP8 against 72B BF16 TP 8: +0.4 ± 4.6 points.

These results also exposed a bug in the gate. It compared FP8 with the
*highest*-scoring BF16 run, which turned a −2.4 point difference into a
−5.2 point "fail". The gate now compares with the most similar BF16 run (same
engine, same TP, no speculative decoding), and calls a result inconclusive when
the ±2-standard-error interval straddles the threshold.

### 5. N-gram speculative decoding hurt on this workload

Goodput fell from 9.5 to 8.0 req/s, and median ITL rose from 13–14 ms to
19–23 ms. N-gram drafting proposes tokens by matching text earlier in the
prompt, and these synthetic prompts are random filler words, so drafts are
rarely accepted and the extra work is wasted. **This is a result about this
workload, not about speculative decoding.** Code, retrieval or chat prompts with
repeated text would likely behave differently.

## Limitations

- **One run per deployment, with no repeats.** Goodput is bracketed to within
  ~20% (30 s probes); read small differences with caution.
- **Synthetic prompts.** No shared prefixes (so no prefix caching) and little
  repetition (unfavourable to n-gram drafting).
- **Small quality sample.** 250 GSM8K examples, no MMLU.
- **vLLM only, default settings.** No SGLang, MoE, pipeline parallelism, or
  tuning of `--max-num-batched-tokens`.
- **No cost figures.** The provider price wasn't recorded, so the comparisons use
  output tokens per second per GPU, which is proportional to cost on one box.

## Next experiments these results suggest

1. **Attack the ITL tail.** Sweep `--max-num-batched-tokens` (e.g. 512, 2048,
   8192) on 7B TP 1, and see whether goodput moves more than it did with TP or FP8.
2. **Make quality decidable.** Run full GSM8K (1,319 examples), and score BF16 and FP8
   on the same examples so the comparison is paired.
3. **Repeat runs** (3× per deployment) to put an error bar on goodput.
4. **A realistic workload** (multi-turn chat or code) to re-test speculative
   decoding and exercise prefix caching.
5. **Native FP8** on Hopper or Ada, to separate "FP8" from "weight-only on Ampere".
6. **The rest of the matrix:** SGLang, MoE expert parallelism, and TP vs PP at 8 GPUs.
