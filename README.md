# KVCompass

KVCompass is a notebook-first evaluation project for comparing KV-cache optimization methods on long-context benchmark workloads using [KVPress](https://github.com/NVIDIA/kvpress) as the execution backbone.

The workflow is split across two notebooks:

- [`benchmarks.ipynb`](./benchmarks.ipynb): collect benchmark data
- [`analysis.ipynb`](./analysis.ipynb): load saved artifacts and generate tables and plots

## What This Project Does

This project answers a practical question:

Which KV-cache optimization method should be used for a given long-context workload when balancing quality, latency, throughput, and memory?

The current workflow evaluates benchmark-backed workloads from the `ruler` dataset and compares methods such as:

- `no_compression`
- `snapkv`
- `expected_attention`
- `knorm`
- `tova`
- `streaming_llm`

Across the benchmark assignments, these methods are tested on task families including:

- Needle in a haystack
- Question answering
- Multi-hop tracing
- Aggregation

The benchmark notebook writes per-scenario artifacts to Google Drive, and the analysis notebook loads those saved summaries and metrics to build presentation-friendly tables and visualizations.

## Main Entry Points

The primary workflow is split into two notebooks:

- [`benchmarks.ipynb`](./benchmarks.ipynb)
- [`analysis.ipynb`](./analysis.ipynb)

At a high level:

1. `benchmarks.ipynb` mounts Google Drive, installs dependencies, authenticates with Hugging Face, writes temporary sweep YAML files, and runs benchmark sweeps.
2. `analysis.ipynb` mounts Google Drive, loads the generated `assignment_*__summary.csv` files and per-run `metrics.json` files, and produces leaderboards, plots, and granular analyses.

## How The Workflow Executes

The benchmark collection path is:

```text
benchmarks.ipynb
  -> scripts/run_kvpress_benchmark_sweep.py
    -> src/kvpress_eval/benchmark_sweep.py
      -> src/kvpress_eval/benchmark_eval.py
      -> src/kvpress_eval/config.py
      -> src/kvpress_eval/methods.py
      -> src/kvpress_eval/runner.py
      -> src/kvpress_eval/compat.py
      -> src/kvpress_eval/benchmark_registry.py
```

The scoring path for the current repo uses:

- Hugging Face dataset: `simonjegou/ruler`
- scorer: `kvpress_eval.benchmarks.ruler.calculate_metrics`

The analysis path is notebook-local:

```text
analysis.ipynb
  -> load summary CSV files
  -> load per-run metrics.json files
  -> build leaderboard and pivot tables
  -> generate plots and granular task analysis
```

## Repository Structure

```text
KVCompass/
├── README.md
├── pyproject.toml
├── requirements.txt
├── benchmarks.ipynb
├── analysis.ipynb
├── Evaluation_Barnes.ipynb
├── Evaluation_Sawant.ipynb
├── ARCHITECTURE.md
├── configs/
│   └── methods.yaml
├── scripts/
│   ├── plot_memory_savings.py
│   ├── plot_quality_heatmap_16k.py
│   ├── run_kvpress_benchmark_eval.py
│   └── run_kvpress_benchmark_sweep.py
├── src/
│   └── kvpress_eval/
│       ├── __init__.py
│       ├── benchmark_eval.py
│       ├── benchmark_registry.py
│       ├── benchmark_sweep.py
│       ├── compat.py
│       ├── config.py
│       ├── methods.py
│       └── runner.py
└── results/
    ├── memory_savings.png
    ├── quality_heatmap_16k.png
    ├── plots/
    ├── raw/
    └── summary/
```

## Notebook Roles

### `benchmarks.ipynb`

Use this notebook when you want to run or rerun experiments.

It is responsible for:

- mounting Google Drive
- cloning or refreshing the repo in Colab
- installing dependencies and the local package
- loading `HF_TOKEN` from Colab secrets
- defining shared run settings such as model name and fraction
- writing temporary benchmark sweep configs for assignments
- running `scripts/run_kvpress_benchmark_sweep.py`
- optionally running a one-off smoke test through `scripts/run_kvpress_benchmark_eval.py`

### `analysis.ipynb`

Use this notebook after benchmark artifacts already exist.

It is responsible for:

- mounting Google Drive
- installing analysis dependencies used by the notebook cells
- loading assignment summary CSV files
- reading each run's `metrics.json`
- computing leaderboard views
- building pivot tables
- generating presentation plots
- performing granular task-level and grouped-category analysis

## Requirements

### Runtime environment

The notebook workflow is intended for:

- Google Colab
- Python 3.10+
- GPU-backed runtime recommended for benchmark collection
- Google Drive access for saving and reloading outputs

### Accounts and access

You will need:

- a Hugging Face account
- a valid Hugging Face token stored in Colab secrets as `HF_TOKEN`
- access to the selected model if it is gated

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

Note: `analysis.ipynb` also installs and uses `seaborn` in its own setup cell.

## Setup

### Run in Google Colab

Open [`benchmarks.ipynb`](./benchmarks.ipynb) in Colab to generate benchmark artifacts.

Then open [`analysis.ipynb`](./analysis.ipynb) in Colab to analyze the saved results.

The benchmark notebook setup cells will:

- mount Google Drive at `/content/drive`
- clone this repo to `/content/KVCompass` if needed
- update the checked-out branch
- install dependencies
- install the package with `pip install -e .`

The analysis notebook setup cells will:

- mount Google Drive
- install the plotting and dataframe dependencies it needs
- point to the shared results directory

## Benchmark Workflow

The benchmark notebook is organized around assignment-specific benchmark sweeps.

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

These files are created by the notebook at runtime and are not committed to the repo.

## Methods Configuration

Method definitions live in:

- [`configs/methods.yaml`](./configs/methods.yaml)

That file declares the KV-cache strategies used by the benchmark notebook, including:

- baseline no-compression execution
- KVPress press-based methods

The current workflow uses:

- `no_compression`
- `snapkv`
- `expected_attention`
- `knorm`
- `tova`
- `streaming_llm`

## Outputs

The benchmark notebook writes outputs into a shared Google Drive directory, typically:

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

## Analysis Performed In `analysis.ipynb`

After the benchmark runs finish, the analysis notebook:

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

The repo includes a small benchmark runner surface:

- `scripts/run_kvpress_benchmark_sweep.py`: run a benchmark sweep from a generated YAML config
- `scripts/run_kvpress_benchmark_eval.py`: run one benchmark evaluation slice
- `scripts/plot_memory_savings.py`: standalone plotting utility for saved memory results
- `scripts/plot_quality_heatmap_16k.py`: standalone plotting utility for saved quality results

For the notebook workflow, `run_kvpress_benchmark_sweep.py` is the primary execution script, while `run_kvpress_benchmark_eval.py` supports the optional smoke-test cell in `benchmarks.ipynb`.

## Limitations And Notes

- The notebook workflow is Colab-specific as written.
- Benchmark data is pulled from Hugging Face at runtime.
- Model weights are also pulled at runtime.
- A GPU runtime is strongly recommended for practical benchmark execution time.
- `src/kvpress_eval/compat.py` applies compatibility patches so KVPress works with newer Transformers cache behavior.
- `analysis.ipynb` assumes benchmark artifacts already exist in the shared Drive location.

## Reproducibility Notes

To reproduce results reliably, keep the following consistent:

- Colab runtime type
- selected model name
- `HF_TOKEN` access
- fraction value used in the benchmark notebook
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

- the benchmark sweep cells completed successfully
- Google Drive is mounted
- the shared results directory exists
- the summary CSV files were written under `benchmark_eval/`

### Analysis notebook shows no data

Check that:

- the benchmark notebook finished first
- the expected `assignment_*__summary.csv` files exist
- each summary row points to a valid `metrics.json`
- `SHARED_RESULTS_DIR` in `analysis.ipynb` matches the directory used by `benchmarks.ipynb`

### Out-of-memory or slow execution

Try:

- a smaller model
- a smaller `FRACTION`
- fewer methods per assignment
- fewer benchmark scenarios
