from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .benchmark_eval import (
    BenchmarkConfig,
    _execute_benchmark_dataframe,
    _load_benchmark_df,
    _prepare_df,
    _save_benchmark_outputs,
    _set_seed,
    _setup_logging,
)
from .compat import apply_kvpress_compat_patches
from .config import get_method_configs
from .methods import build_method_runtime
from .runner import load_model_bundle


@dataclass
class SweepArtifacts:
    summary_csv: Path
    run_count: int


def _write_summary_csv(summary_csv: Path, summary_rows: list[dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["dataset"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "sweep" not in data:
        raise ValueError("Sweep config must contain a top-level 'sweep' mapping")
    return data["sweep"]


def _expand_runs(sweep: dict[str, Any]) -> list[BenchmarkConfig]:
    runs: list[BenchmarkConfig] = []
    base = {
        "model": sweep["model"],
        "device": sweep.get("device", "auto"),
        "torch_dtype": sweep.get("torch_dtype", "auto"),
        "output_dir": sweep.get("output_dir", "results/benchmark_eval"),
        "methods_config_path": sweep.get("methods_config_path", "configs/methods.yaml"),
        "seed": int(sweep.get("seed", 42)),
        "verbose": bool(sweep.get("verbose", False)),
    }
    for scenario in sweep.get("scenarios", []):
        methods = scenario.get("methods", [])
        budgets_cfg = scenario.get("budgets", {})
        default_budgets = budgets_cfg.get("default", [0.5])
        for method in methods:
            for budget in budgets_cfg.get(method, default_budgets):
                runs.append(
                    BenchmarkConfig(
                        scenario_name=scenario["name"],
                        dataset=scenario["dataset"],
                        data_dir=scenario.get("data_dir"),
                        model=base["model"],
                        method=method,
                        budget=float(budget),
                        task_prefixes=scenario.get("task_prefixes"),
                        fraction=float(scenario.get("fraction", 1.0)),
                        device=base["device"],
                        torch_dtype=base["torch_dtype"],
                        output_dir=base["output_dir"],
                        methods_config_path=base["methods_config_path"],
                        seed=base["seed"],
                        verbose=base["verbose"],
                    )
                )
    return runs


def run_benchmark_sweep(config_path: str | Path) -> SweepArtifacts:
    import torch

    sweep = load_sweep_config(config_path)
    _setup_logging(bool(sweep.get("verbose", False)))
    _set_seed(int(sweep.get("seed", 42)))
    apply_kvpress_compat_patches()

    runs = _expand_runs(sweep)
    methods = {m["name"]: m for m in get_method_configs(sweep.get("methods_config_path", "configs/methods.yaml"))}
    model, tokenizer, pipeline = load_model_bundle(
        model_name=sweep["model"],
        device=sweep.get("device", "auto"),
        torch_dtype=sweep.get("torch_dtype", "auto"),
    )

    dataset_cache: dict[tuple[str, str | None, tuple[str, ...], float], Any] = {}
    summary_rows: list[dict[str, Any]] = []
    output_dir = Path(sweep.get("output_dir", "results/benchmark_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_name = sweep.get("name", "benchmark_sweep")
    summary_csv = output_dir / f"{sweep_name}__summary.csv"

    for run in runs:
        runtime = build_method_runtime(
            methods[run.method],
            budget=run.budget,
            model_config=model.config,
            model_layer_count=len(model.model.layers),
        )
        runtime.model = model
        runtime.tokenizer = tokenizer

        cache_key = (run.dataset, run.data_dir, tuple(run.task_prefixes or []), run.fraction)
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = _load_benchmark_df(run, tokenizer)

        df = _prepare_df(dataset_cache[cache_key], run, runtime)
        df, metrics, run_stats = _execute_benchmark_dataframe(
            df=df,
            config=run,
            pipeline=pipeline,
            tokenizer=tokenizer,
            runtime=runtime,
            torch_module=torch,
        )
        predictions_path, metrics_path = _save_benchmark_outputs(
            df=df,
            metrics=metrics,
            run_stats=run_stats,
            config=run,
        )
        summary_rows.append(
            {
                "scenario_name": run.scenario_name or "",
                "dataset": run.dataset,
                "data_dir": run.data_dir or "",
                "task_prefixes": ",".join(run.task_prefixes or []),
                "model": run.model,
                "method": run.method,
                "budget": run.budget,
                "predictions_path": str(predictions_path),
                "metrics_path": str(metrics_path),
                "avg_latency_seconds": run_stats.get("avg_latency_seconds"),
                "avg_throughput_tokens_per_second": run_stats.get("avg_throughput_tokens_per_second"),
                "peak_gpu_memory_mb": run_stats.get("peak_gpu_memory_mb"),
            }
        )
        _write_summary_csv(summary_csv, summary_rows)

    return SweepArtifacts(summary_csv=summary_csv, run_count=len(summary_rows))
