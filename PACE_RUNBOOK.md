# PACE Runbook

This branch standardizes the PACE workflow around one reproducible experiment matrix and one benchmark-aware reporting path.

## Standardized Experiments

### Core matrix
- Benchmark: `RULER`
- Categories:
  - `niah`
  - `qa`
  - `vt`
  - `aggregation` (`cwe`, `fwe`)
- Context lengths:
  - `4096`
  - `8192`
- Methods:
  - `no_compression`
  - `snapkv`
  - `expected_attention`
  - `knorm`
  - `tova`
  - `streaming_llm`
- Budgets:
  - `no_compression @ 1.0`
  - compressed methods `@ 0.5`

### Budget sensitivity
- Categories:
  - `niah`
  - `qa`
- Context lengths:
  - `4096`
  - `8192`
- Budgets:
  - `1.0` baseline
  - `0.75`
  - `0.5`
  - `0.25`

## Generate configs locally

```bash
python3 scripts/write_pace_configs.py \
  --dest-dir configs \
  --output-dir results/benchmark_eval \
  --fraction 1.0
```

## Submit PACE jobs

### Full core matrix

```bash
bash scripts/submit_pace_jobs.sh matrix
```

### Budget sensitivity

```bash
bash scripts/submit_pace_jobs.sh budget4k
bash scripts/submit_pace_jobs.sh budget8k
```

### Assignment split

```bash
bash scripts/submit_pace_jobs.sh assignments
```

## Notes on the sbatch template

The generic batch script is:

- [scripts/pace_sweep.sbatch](/Users/aarnavsawant/Documents/CS6675/KVCompass/scripts/pace_sweep.sbatch)

You may need to tweak:

- `#SBATCH --gres=gpu:1`
- `#SBATCH --mem=64G`
- `#SBATCH --time=24:00:00`
- module or environment activation if your PACE setup requires it

The script expects:

- `CONFIG_PATH`
- `RESULTS_DIR`
- `REPO_ROOT`

Those are provided automatically by [scripts/submit_pace_jobs.sh](/Users/aarnavsawant/Documents/CS6675/KVCompass/scripts/submit_pace_jobs.sh).

## Build the final report

After the sweeps finish, generate the standardized report from the benchmark artifacts:

```bash
python3 scripts/build_benchmark_report.py \
  --summary-dir results/benchmark_eval \
  --output-dir results/benchmark_report
```

This produces:

- `combined_summary.csv`
- `task_metrics.csv`
- `baseline_relative.csv`
- `recommendations.csv`
- `failure_examples.csv`
- `failure_mode_summary.csv`
- plots for:
  - quality by category
  - quality drop from baseline
  - context-length comparison
  - quality vs memory
  - quality vs latency
  - NIAH subtask heatmap
  - budget sensitivity
