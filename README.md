# KVCompass

KVCompass is a notebook-first evaluation project for comparing KV-cache optimization methods on long-context benchmark workloads using [KVPress](https://github.com/NVIDIA/kvpress) as the execution backbone.

The main entry point is [`KVCompass.ipynb`](./KVCompass.ipynb). The notebook is designed to run in Google Colab, mount Google Drive for persistent storage, pull this repository, install the package, execute config-driven benchmark sweeps, and analyze the resulting metrics with tables and plots.

## What This Project Does

This project answers a practical question:

Which KV-cache optimization method should be used for a given long-context workload when balancing quality, latency, throughput, and memory?

The current notebook evaluates benchmark-backed workloads from the `ruler` dataset and compares methods such as:

- `no_compression`
- `snapkv`
- `expected_attention`
- `knorm`
- `tova`
- `streaming_llm`

Across the notebook assignments, these methods are tested on task families including:

- Needle in a haystack
- Question answering
- Multi-hop tracing
- Aggregation

Each run writes per-scenario benchmark artifacts, then the notebook aggregates the summaries and builds presentation-friendly result tables and charts.

## Main Entry Point

The primary workflow is the notebook:

- [`KVCompass.ipynb`](./KVCompass.ipynb)

At a high level, the notebook:

1. Mounts Google Drive.
2. Clones or refreshes this repository in Colab.
3. Installs Python dependencies and the local package in editable mode.
4. Loads a Hugging Face access token from Colab secrets.
5. Writes temporary benchmark sweep YAML files for each assignment.
6. Runs benchmark sweeps through `scripts/run_kvpress_benchmark_sweep.py`.
7. Reads the generated summary CSV files and per-run `metrics.json` files.
8. Produces leaderboard tables, matrix views, and plots.

## How The Notebook Executes

The notebook uses this runtime path:

```text
KVCompass.ipynb
  -> scripts/run_kvpress_benchmark_sweep.py
    -> src/kvpress_eval/benchmark_sweep.py
      -> src/kvpress_eval/benchmark_eval.py
      -> src/kvpress_eval/config.py
      -> src/kvpress_eval/methods.py
      -> src/kvpress_eval/runner.py
      -> src/kvpress_eval/compat.py
      -> src/kvpress_eval/benchmark_registry.py
        -> src/kvpress_eval/benchmarks/ruler/calculate_metrics.py
```

For the current notebook, the benchmark dataset is:

- Hugging Face dataset: `simonjegou/ruler`

The model is configured in the notebook itself. The current default is:

- `Qwen/Qwen2.5-1.5B-Instruct`

## Repository Structure

```text
KVCompass/
├── KVCompass.ipynb
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── benchmark_sweeps.yaml
│   ├── methods.yaml
│   └── scenarios.yaml
├── scripts/
│   ├── aggregate_results.py
│   ├── make_plots.py
│   ├── make_recommendations.py
│   ├── run_budget_sweep.py
│   ├── run_kvpress_benchmark_eval.py
│   ├── run_kvpress_benchmark_sweep.py
│   └── run_kvpress_eval.py
├── src/
│   └── kvpress_eval/
│       ├── aggregate.py
│       ├── benchmark_eval.py
│       ├── benchmark_registry.py
│       ├── benchmark_sweep.py
│       ├── compat.py
│       ├── config.py
│       ├── evaluate.py
│       ├── io_utils.py
│       ├── methods.py
│       ├── plotting.py
│       ├── recommendations.py
│       ├── runner.py
│       ├── scenarios.py
│       └── benchmarks/
└── results/
    ├── benchmark_eval/
    ├── plots/
    ├── raw/
    └── summary/
```

## Requirements

### Runtime environment

The notebook is intended for:

- Google Colab
- Python 3.10+
- GPU-backed runtime recommended
- Google Drive access for saving outputs

### Accounts and access

You will need:

- A Hugging Face account
- A valid Hugging Face token stored in Colab secrets as `HF_TOKEN`
- Access to the selected model if it is gated

### Python dependencies

Install from:

- [`requirements.txt`](./requirements.txt)
- editable package install via [`pyproject.toml`](./pyproject.toml)

Core dependencies include:

- `kvpress`
- `optimum-quanto`
- `pandas`
- `matplotlib`
- `PyYAML`
- `nltk`
- `bert-score`
- `rouge`
- `jieba`
- `fuzzywuzzy`

Note: the notebook also imports `seaborn` in its later analysis cells, but `seaborn` is not currently listed in `requirements.txt` or `pyproject.toml`.

## Setup

### Option 1: Run the notebook in Google Colab

Open [`KVCompass.ipynb`](./KVCompass.ipynb) in Colab and run the cells in order.

The setup cells will:

- mount Google Drive at `/content/drive`
- clone this repo to `/content/KVCompass` if needed
- update the checked-out branch
- install dependencies
- install the package with `pip install -e .`


## Notebook Workflow

The notebook is organized around assignment-specific benchmark sweeps.

For each assignment, it:

1. Builds a temporary YAML config inside `configs/`.
2. Selects a benchmark dataset, context size, task prefixes, methods, and budgets.
3. Runs the sweep with `scripts/run_kvpress_benchmark_sweep.py`.
4. Saves outputs under the shared Drive results directory.

The generated temporary configs follow this pattern:

- `configs/benchmark_sweeps.assignment_1.yaml`
- `configs/benchmark_sweeps.assignment_2.yaml`
- `configs/benchmark_sweeps.assignment_3.yaml`
- `configs/benchmark_sweeps.assignment_4.yaml`
- `configs/benchmark_sweeps.assignment_5.yaml`

## Methods Configuration

Method definitions live in:

- [`configs/methods.yaml`](./configs/methods.yaml)

That file declares the supported KV-cache strategies and their parameters, including:

- baseline no-compression execution
- KVPress press-based methods
- quantized cache options
- nested and layerwise compression variants

The notebook currently uses a subset of those declared methods, especially:

- `no_compression`
- `snapkv`
- `expected_attention`
- `knorm`
- `tova`
- `streaming_llm`

## Outputs

The notebook writes benchmark outputs into a shared Google Drive directory, typically:

```text
/content/drive/MyDrive/KVCompass/benchmark_eval
```

Important generated artifacts include:

- `assignment_1__summary.csv`
- `assignment_2__summary.csv`
- `assignment_3__summary.csv`
- `assignment_4__summary.csv`
- `assignment_5__summary.csv`

Each benchmark run also produces a dedicated run directory containing:

- `predictions.csv`
- `metrics.json`
- `run_stats.json`
- `config.yaml`

## Analysis Performed In The Notebook

After the sweeps finish, the notebook:

- loads all available assignment summary CSV files
- combines them into one dataframe
- reads each run's `metrics.json`
- computes average quality values
- builds a leaderboard
- pivots results into a scenario-by-method matrix
- generates bar charts for quality, latency, and memory
- creates simple recommendation tables
- explores task-level and task-category-level benchmark metrics

## Supported Scripts

Even though the notebook is the main interface, the repo also includes standalone scripts:

- `scripts/run_kvpress_benchmark_sweep.py`: run a benchmark sweep from YAML
- `scripts/run_kvpress_benchmark_eval.py`: run one benchmark evaluation slice
- `scripts/run_kvpress_eval.py`: run the non-benchmark synthetic scenario pipeline
- `scripts/run_budget_sweep.py`: run broader synthetic workload sweeps
- `scripts/aggregate_results.py`: aggregate raw CSV outputs
- `scripts/make_plots.py`: generate plots from raw results
- `scripts/make_recommendations.py`: create summary recommendations

For the current notebook workflow, `run_kvpress_benchmark_sweep.py` is the critical script.

## Limitations And Notes

- The notebook is Colab-specific as written.
- Benchmark data is pulled from Hugging Face at runtime.
- Model weights are also pulled at runtime.
- A GPU runtime is strongly recommended for practical execution time.
- The repo includes code paths that are not used by the notebook, especially the synthetic scenario evaluation pipeline.
- `src/kvpress_eval/compat.py` applies compatibility patches so KVPress works with newer Transformers cache behavior.
- Later notebook visualizations require `seaborn`, which is not currently declared as a dependency.

## Reproducibility Notes

To reproduce notebook results reliably, keep the following consistent:

- Colab runtime type
- selected model name
- `HF_TOKEN` access
- fraction value used in the notebook
- task prefixes per assignment
- benchmark `data_dir` values such as `4096` and `8192`
- method list and budget list
- branch checked out by the setup cell

## Troubleshooting

### Hugging Face auth errors

Check that:

- `HF_TOKEN` exists in Colab secrets
- the token has permission to access the selected model
- you are using the correct model name

### Missing benchmark outputs

Check that:

- the sweep cells completed successfully
- Google Drive is mounted
- the shared results directory exists
- the summary CSV files were written under `benchmark_eval/`

### Plotting cells fail

If the later analysis cells fail on `seaborn`, install it in the runtime before rerunning the plotting cells.

### Out-of-memory or slow execution

Try:

- a smaller model
- a smaller `FRACTION`
- fewer methods per assignment
- fewer benchmark scenarios

