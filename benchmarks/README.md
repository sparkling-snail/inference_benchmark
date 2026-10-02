# Phase 5 Benchmarks

This directory contains the first benchmark harness for comparing the
local runtime against itself and external runtimes through a shared
OpenAI-compatible HTTP surface.

## Start the local server

```bash
python servers/local_openai_server.py --runtime continuous --port 8000
```

For the naive baseline:

```bash
python servers/local_openai_server.py --runtime naive --port 8001
```

## Run the benchmark matrix

```bash
python -m benchmarks.runner --config configs/phase5_matrix.yaml
```

Or target a single runtime:

```bash
python -m benchmarks.runner --config configs/phase5_matrix.yaml --runtime local_continuous
```

## Output

- Raw JSON: `results/phase5/raw/`
- Markdown summary: `results/phase5/summary.md`
- Plots: `results/phase5/plots/phase5_summary.png`
