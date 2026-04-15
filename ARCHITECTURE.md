# KVCompass Architecture

## Overview

KVCompass is a workload-aware evaluation framework for comparing KV-cache optimization methods. It answers the practical systems question: *given a workload, compression budget, and context length, which KV-cache strategy should you choose?*

The framework wraps [KVPress](https://github.com/princeton-nlp/kvpress) as its execution backbone. The `KVCompass.ipynb` notebook is the primary interface: it runs the RULER benchmark across four task categories (needle-in-a-haystack, question answering, multi-hop tracing, aggregation), then aggregates and visualizes results.

---

## Repository Layout

Files and directories relevant to the notebook:

```
KVCompass/
├── configs/
│   └── methods.yaml                        # KV-cache method definitions
├── scripts/
│   ├── run_kvpress_benchmark_sweep.py      # Entry point invoked by each assignment cell
│   └── run_kvpress_benchmark_eval.py       # Entry point invoked by the smoke test cell
├── src/kvpress_eval/
│   ├── config.py                           # YAML loading
│   ├── methods.py                          # Method instantiation
│   ├── compat.py                           # KVPress compatibility patches
│   ├── benchmark_registry.py              # Dataset → HF path and scorer mapping
│   ├── benchmark_eval.py                  # Single benchmark run orchestration
│   ├── benchmark_sweep.py                 # Config-driven batch sweep orchestration
│   └── benchmarks/
│       └── ruler/
│           └── calculate_metrics.py        # RULER-specific scoring
├── KVCompass.ipynb
├── pyproject.toml
└── requirements.txt
```

---

## Core Concepts

### KV-Cache Compression

Modern LLMs store key-value pairs from attention layers in a KV-cache to avoid recomputation. This cache grows linearly with context length and becomes a memory and throughput bottleneck at long contexts. *Compression methods* prune or otherwise reduce the size of this cache at the cost of some generation quality.

KVCompass uses the KVPress library, which provides a unified `Press` interface wrapping many compression methods. A `Press` object is injected into a `KVPressTextGenerationPipeline`, which applies it during inference.

### Budget

The central tuning knob across all evaluations is the *budget* — the fraction of the KV-cache retained:
- `budget = 1.0` → no compression (full cache)
- `budget = 0.5` → 50% of tokens kept
- `compression_ratio = 1 - budget`

### Benchmark Matrix

The notebook evaluates a fixed matrix:

| Dimension | Values |
|---|---|
| Dataset | RULER (`simonjegou/ruler` on HuggingFace) |
| Task categories | `niah`, `qa`, `vt`, `cwe`, `fwe` |
| Context lengths | 4096, 8192 tokens |
| Methods | `no_compression`, `snapkv`, `expected_attention`, `knorm`, `tova`, `streaming_llm` |
| Budgets | `1.0` (baseline), `0.5` (compressed) |
| Fraction of dataset | 0.02 (2% sample) |

---

## Module Reference

### `config.py`

Thin YAML loading layer used at sweep startup.

- `load_yaml(path)` — reads a YAML file with basic validation
- `get_method_configs()` — returns the list of method entries from `configs/methods.yaml`

### `methods.py`

Translates YAML method definitions into live KVPress objects.

**`MethodRuntime`** (dataclass)

| Field | Type | Description |
|---|---|---|
| `name` | str | Method identifier |
| `budget` | float | Fraction of cache retained |
| `press` | BasePress \| None | KVPress compression object |
| `cache` | QuantizedCache \| None | Quantized cache object |
| `compression_ratio` | float | `1 - budget` |

**`build_method_runtime(method_cfg, budget)`** — factory that maps YAML `kind` values to objects:
- `kind: none` → no-op (baseline)
- `kind: press` → any `kvpress.*Press` class, e.g. `SnapKVPress`, `ExpectedAttentionPress`, `KnormPress`, `TOVAPress`, `StreamingLLMPress`

The notebook uses six methods in total: `no_compression` (kind: none) and five press methods (kind: press) at budget 0.5.

### `compat.py`

Patches KVPress for compatibility with newer versions of the Transformers library. Applied once at model load time and otherwise invisible.

**`apply_kvpress_compat_patches()`** applies two patches:
1. **`BasePress.forward_hook`** — handles a missing `cache_position` argument introduced in newer Transformers versions
2. **`search_hyperplane()`** in the attention patch module — adds numerical stability guards

### `benchmark_registry.py`

Maps dataset names to HuggingFace paths and scorer modules. The notebook uses only the `ruler` entry.

**`DATASET_REGISTRY`** — `ruler` → `simonjegou/ruler`

**`get_scorer(dataset_name)`** — dynamically imports `calculate_metrics` from `benchmarks/ruler/calculate_metrics.py`.

### `benchmark_eval.py`

Orchestrates a single benchmark run (one dataset × method × budget × scenario slice).

**`BenchmarkConfig`** (dataclass) — all parameters for one run: `dataset`, `model`, `method`, `budget`, `data_dir`, `task_prefixes`, `fraction`, `output_dir`, etc.

**`run_benchmark_evaluation(config, model_bundle)`**:
1. `_load_benchmark_df()` — fetches the RULER dataset from HuggingFace; subsamples by `fraction`; filters rows by `task_prefixes`
2. `_prepare_df()` — assembles the prompt column (query-aware context if configured)
3. `_execute_benchmark_dataframe()` — main inference loop:
   - **DecodingPress** (e.g., StreamingLLM): runs per-example since cache must be rebuilt each time
   - **Regular Press**: groups rows by shared context, then runs all questions for each group
   - Records per-example latency, token throughput, and peak GPU memory
   - Calls `get_scorer("ruler")` to compute task-specific metrics
4. `_save_benchmark_outputs()` — writes four files to a run-specific directory under `output_dir/benchmark_eval/<run_name>/`:
   - `predictions.csv` — generated text alongside gold answers
   - `metrics.json` — RULER task-specific scores
   - `run_stats.json` — latency, throughput, token counts
   - `config.yaml` — the `BenchmarkConfig` used for this run

### `benchmark_sweep.py`

Expands a sweep YAML into many `BenchmarkConfig` objects and runs them sequentially, reusing the model.

**`load_sweep_config(path)`** — parses a sweep YAML file (the notebook writes these inline before calling the script)

**`_expand_runs(sweep_cfg)`** — expands the `scenarios × methods × budgets` matrix into a flat list of `BenchmarkConfig` objects

**`run_benchmark_sweep(config_path)`**:
1. Parses the sweep config and expands into a run list
2. Loads the model once (`AutoModelForCausalLM` + tokenizer + `KVPressTextGenerationPipeline`); reuses it for all runs
3. Caches loaded HuggingFace datasets in memory by `(dataset, data_dir)` to avoid repeated network fetches
4. For each run, calls `run_benchmark_evaluation()` and appends one summary row to `<output_dir>/benchmark_eval/<sweep_name>__summary.csv`

**Summary CSV columns**: `scenario_name`, `dataset`, `data_dir`, `task_prefixes`, `method`, `budget`, `avg_latency_seconds`, `avg_throughput_tokens_per_second`, `peak_gpu_memory_mb`, `metrics_path`, `predictions_path`

### `benchmarks/ruler/calculate_metrics.py`

RULER-specific scorer called by `benchmark_eval.py` after inference.

Scoring is dispatched by task type:

| Task prefix | Metric |
|---|---|
| `niah` (needle-in-a-haystack) | Token recall — fraction of needle tokens present in the generated output |
| `qa` (question answering) | F1 and exact match between generated answer and gold answer |
| `vt` (variable tracing / multi-hop) | Exact string match on the traced value |
| `cwe`, `fwe` (aggregation) | Word recall — fraction of expected words found in output |

Returns a nested `metrics` dict keyed by task name, then metric name.

---

## Configuration: `configs/methods.yaml`

The notebook references this file via the `methods_config_path` field in each sweep YAML. It defines all available compression methods. Each entry has a `name`, `kind`, and kind-specific parameters:

```yaml
- name: snapkv
  kind: press
  class: SnapKVPress
  window_size: 32
  kernel_size: 5

- name: no_compression
  kind: none
```

The six methods used by the notebook: `no_compression`, `snapkv`, `expected_attention`, `knorm`, `tova`, `streaming_llm`.

---

## Notebook Structure and Data Flow

### Per-assignment cells (Tony, Will, Grady, Jamez, Aarnav)

Each teammate runs two cells:

1. **Config write cell** — generates a sweep YAML (e.g. `configs/benchmark_sweeps.assignment_1.yaml`) with their assigned task category, context lengths, methods, and budget. The YAML is written to disk inline using Python's `pathlib`.

2. **Run cell** — invokes `scripts/run_kvpress_benchmark_sweep.py --config <yaml>`, which calls `benchmark_sweep.run_benchmark_sweep()`.

Results are written to Google Drive (`/content/drive/MyDrive/KVCompass/benchmark_eval/`).

### Final aggregation cells (Aarnav)

After all five assignments complete, the final cells:

1. Load all five `assignment_*__summary.csv` files from Drive and concatenate them into `combined_summary`
2. For each row, open the referenced `metrics_path`, flatten all numeric metric values, and compute `avg_quality`
3. Build a `leaderboard` DataFrame sorted by scenario → quality → latency
4. Render a pivot table (`scenario_name × data_dir` vs `method`)
5. Plot three bar charts (quality, latency, GPU memory) using matplotlib
6. Print a recommendations table with `best_for_quality`, `best_for_latency`, `best_for_memory` rows

### Full data flow

```
Notebook assignment cell
  │  Writes configs/benchmark_sweeps.assignment_N.yaml
  │
  ▼
scripts/run_kvpress_benchmark_sweep.py --config <yaml>
  │
  ▼
benchmark_sweep.run_benchmark_sweep()
  │  Load model once (AutoModelForCausalLM + KVPressTextGenerationPipeline)
  │  Apply compat patches
  │  _expand_runs() → list of BenchmarkConfig
  │  Cache datasets by (dataset, data_dir)
  │
  ├─ For each BenchmarkConfig (scenario × method × budget):
  │     _load_benchmark_df()            → RULER DataFrame (filtered by task_prefixes, fraction)
  │     _prepare_df()                   → prompt column assembled
  │     _execute_benchmark_dataframe()
  │       ├─ pipeline(prompt, press=...) → generated text
  │       ├─ measure latency / throughput / GPU memory
  │       └─ ruler.calculate_metrics()  → task-specific scores
  │     _save_benchmark_outputs()       → per-run directory on Drive
  │     Append summary row to assignment_N__summary.csv
  │
  ▼
Drive/KVCompass/benchmark_eval/
  ├── <run_name>/
  │     ├── predictions.csv
  │     ├── metrics.json
  │     ├── run_stats.json
  │     └── config.yaml
  └── assignment_N__summary.csv

Notebook aggregation cells
  │  pd.concat(assignment_1..5__summary.csv)
  │  Read metrics.json for each row → avg_quality
  │  Build leaderboard, pivot table, bar charts, recommendations
  ▼
Inline display (no files written)
```

---

## External Dependencies

| Dependency | Role |
|---|---|
| `kvpress` | KV-cache compression method implementations and pipeline |
| `transformers` | Model loading (`AutoModelForCausalLM`, `AutoTokenizer`) |
| `datasets` | HuggingFace dataset loading (RULER) |
| `pandas` | DataFrame manipulation in benchmark eval and notebook aggregation |
| `matplotlib` | Bar charts in the aggregation cells |
| `PyYAML` | Sweep config parsing |
| `nltk`, `rouge`, `fuzzywuzzy` | Token/word-level scoring inside the RULER scorer |

**Runtime**: Python 3.10+, CUDA GPU (notebook targets Google Colab)

---

## Key Architectural Decisions

**Model loaded once per assignment.** `run_benchmark_sweep()` loads the model a single time and reuses it across all `BenchmarkConfig` runs in the sweep. Each teammate's assignment is one sweep, so the model is loaded once per Colab session.

**Dataset cached within a sweep.** Within a single sweep, datasets are cached in memory by `(dataset, data_dir)`. A teammate running both 4096 and 8192 context lengths fetches each data_dir slice only once.

**Per-run directories on Drive.** Each `BenchmarkConfig` produces its own subdirectory. Results are written incrementally, so a partial run still yields valid outputs for completed configurations.

**Summary CSV as the aggregation contract.** The five assignment summary CSVs are the handoff artifact between teammates and the aggregation cells. Each row in the summary points to its `metrics_path` and `predictions_path`, so the aggregation cells never need to know where individual run directories are located.

**Sweep YAML written inline.** The notebook writes each assignment's sweep YAML at runtime rather than committing static config files. This keeps all experiment parameters visible in the notebook cells and avoids config file proliferation.

**YAML-driven method instantiation.** All compression methods are declared in `configs/methods.yaml`. `build_method_runtime()` dynamically resolves class names via `getattr(kvpress_module, class_name)`. Changing a method's hyperparameters requires only editing the YAML.
