# KVPress Evaluation Framework

This repository is a lightweight, runnable evaluation package for workload-aware comparison of KV-cache optimization methods using [KVPress](https://github.com/NVIDIA/kvpress) as the execution backbone. The goal is not to invent a new compression technique, but to help answer a practical systems question: when should someone choose one KV-cache optimization strategy over another for a given workload, budget, and context length?

The framework reads methods and scenarios from YAML configs, executes scenario x method x budget x context-length sweeps, stores raw measurements to CSV, aggregates them into summary tables, generates plots, and emits a simple recommendation table. The initial workloads use built-in example generators so the package is runnable end to end immediately, with clear hooks for replacing those examples with benchmark datasets later.

## Repo Structure

```text
KVCompass/
├── configs/
│   ├── methods.yaml
│   └── scenarios.yaml
├── results/
│   ├── plots/
│   ├── raw/
│   └── summary/
├── scripts/
│   ├── aggregate_results.py
│   ├── make_plots.py
│   ├── make_recommendations.py
│   ├── run_budget_sweep.py
│   └── run_kvpress_eval.py
├── src/
│   └── kvpress_eval/
│       ├── aggregate.py
│       ├── config.py
│       ├── evaluate.py
│       ├── io_utils.py
│       ├── methods.py
│       ├── plotting.py
│       ├── recommendations.py
│       ├── runner.py
│       └── scenarios.py
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Install

Create and activate a Python 3.10+ environment, then install dependencies:

```bash
cd /Users/aarnavsawant/Documents/CS6675/KVCompass
source ~/miniconda3/bin/activate kvpress-eval
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configuration

- `configs/methods.yaml`: declares the initial methods:
  - `no_compression`
  - `snapkv`
  - `expected_attention`
  - `adakv_expected`
  - `layerwise`
  - `quant_4bit`
- `configs/scenarios.yaml`: declares the initial workloads:
  - `retrieval_long`
  - `reasoning_long`
  - `context_sweep`
  - `prefix_serving`

Budgets map to compression ratio as `compression_ratio = 1 - budget`. The `layerwise` method uses a small per-layer schedule wrapper around `PerLayerCompressionPress`, while `quant_4bit` uses `transformers.QuantizedCache`.
The default `quant_4bit` backend is `quanto`, so `optimum-quanto` is included as a dependency.

## Run

Run one evaluation slice:

```bash
python scripts/run_kvpress_eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --scenario retrieval_long \
  --method snapkv \
  --max-cases 1
```

Run a broader sweep:

```bash
python scripts/run_budget_sweep.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --scenario retrieval_long \
  --scenario prefix_serving
```

Aggregate raw results:

```bash
python scripts/aggregate_results.py \
  --input results/raw/kvpress_eval_YYYYMMDD_HHMMSS.csv \
  --output results/summary/summary.csv
```

Generate plots:

```bash
python scripts/make_plots.py \
  --input results/raw/kvpress_eval_YYYYMMDD_HHMMSS.csv \
  --output-dir results/plots
```

Generate recommendations:

```bash
python scripts/make_recommendations.py \
  --input results/summary/summary.csv \
  --output results/summary/recommendations.csv
```

Run a benchmark with KVPress's repo metrics:

```bash
python scripts/run_kvpress_benchmark_eval.py \
  --dataset ruler \
  --data-dir 4096 \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --method snapkv \
  --budget 0.5
```

Run a whole benchmark sweep from YAML:

```bash
python scripts/run_kvpress_benchmark_sweep.py \
  --config configs/benchmark_sweeps.yaml
```

## Outputs

- `results/raw/*.csv`: one row per scenario/method/budget/context-length/repeat run. Includes latency, throughput, peak GPU memory when available, quality score, generated text, and any error message.
- `results/summary/*.csv`: grouped averages and standard deviations by scenario, method, budget, and context length.
- `results/plots/*.png`: score-vs-budget, latency-vs-budget, and memory-vs-budget plots split by scenario.
- `results/summary/recommendations.csv`: scenario-level recommendations with `best_for_memory`, `best_for_latency`, and `best_balanced`.
- `results/benchmark_eval/<run>/predictions.csv`: benchmark predictions using the same output shape expected by KVPress's benchmark scorers.
- `results/benchmark_eval/<run>/metrics.json`: dataset-specific metrics calculated from the vendored KVPress benchmark metric modules.
- `results/benchmark_eval/<run>/run_stats.json`: benchmark runtime telemetry including total runtime, average latency, throughput, and peak GPU memory when available.
- `results/benchmark_eval/*__summary.csv`: one-row-per-run sweep summary for config-driven benchmark sweeps.

## Notes and TODOs

- The execution path is real KVPress: methods instantiate actual KVPress press classes and run through `KVPressTextGenerationPipeline`.
- The built-in workloads are lightweight prompt generators in `src/kvpress_eval/scenarios.py`, not benchmark datasets yet. They are intended as clean placeholders so the package runs before you integrate external benchmarks.
- For real benchmark metrics, use `scripts/run_kvpress_benchmark_eval.py`, which vendors KVPress's benchmark metric code from the GitHub repo and evaluates against the same benchmark dataset identifiers used there.
- If you want one command to run a matrix of methods and budgets, define scenarios in `configs/benchmark_sweeps.yaml` and use `scripts/run_kvpress_benchmark_sweep.py`. The sweep runner reuses the loaded model across runs to reduce overhead.
- `quant_4bit` depends on the configured quantized cache backend (`quanto` by default). If the backend is missing in your environment, the framework records an error row instead of crashing the full sweep.
- Quality scoring is intentionally simple for now: it checks whether the expected answer appears in the generated text. Replacing this with benchmark-specific scoring is the next natural extension point.
