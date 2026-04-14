from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .benchmark_registry import DATASET_REGISTRY, get_scorer
from .config import get_method_configs

LOGGER = logging.getLogger(__name__)


class _SequentialPipelineWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "You seem to be using the pipelines sequentially on GPU" not in record.getMessage()


@dataclass
class BenchmarkConfig:
    dataset: str
    model: str
    method: str
    scenario_name: str | None = None
    budget: float = 0.5
    data_dir: str | None = None
    task_prefixes: list[str] | None = None
    fraction: float = 1.0
    max_new_tokens: int | None = None
    max_context_length: int | None = None
    query_aware: bool = False
    needle_depth: int | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    output_dir: str = "results/benchmark_eval"
    methods_config_path: str = "configs/methods.yaml"
    seed: int = 42
    verbose: bool = False


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("transformers.pipelines.base").addFilter(_SequentialPipelineWarningFilter())


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_benchmark_df(config: BenchmarkConfig, tokenizer) -> pd.DataFrame:
    from datasets import load_dataset

    if config.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported benchmark dataset: {config.dataset}")
    df = load_dataset(
        DATASET_REGISTRY[config.dataset],
        data_dir=str(config.data_dir) if config.data_dir else None,
        split="test",
    ).to_pandas()

    if config.task_prefixes:
        if "task" not in df.columns:
            raise ValueError(f"Dataset '{config.dataset}' does not expose a 'task' column for task_prefix filtering")
        prefixes = tuple(config.task_prefixes)
        df = df[df["task"].astype(str).str.startswith(prefixes)].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No rows matched task prefixes {config.task_prefixes} for dataset '{config.dataset}'"
            )

    if config.fraction < 1.0:
        df = df.sample(frac=config.fraction, random_state=config.seed)

    if config.dataset == "needle_in_haystack":
        from .benchmarks.needle_in_haystack.utils import insert_needle_in_haystack

        if config.needle_depth is None or config.max_context_length is None:
            raise ValueError("needle_in_haystack requires --needle-depth and --max-context-length")
        df = insert_needle_in_haystack(df, tokenizer, config.max_context_length, config.needle_depth)

    return df


def _prepare_df(df: pd.DataFrame, config: BenchmarkConfig, runtime) -> pd.DataFrame:
    from kvpress import FinchPress

    df = df.copy()

    if isinstance(runtime.press, FinchPress):
        if not config.query_aware:
            raise ValueError("FinchPress requires query-aware evaluation")
        runtime.press.update_model_and_tokenizer(runtime.model, runtime.tokenizer)
        df["context"] = df["context"] + runtime.press.delimiter_token

    if config.query_aware:
        df["context"] = df["context"] + df["question"]
        df["question"] = ""

    return df


def _normalize_output(output: Any, multi: bool) -> list[str]:
    if isinstance(output, dict):
        if multi:
            return [str(x) for x in output.get("answers", [])]
        return [str(output.get("answer", ""))]
    if isinstance(output, list):
        return [str(x) for x in output]
    return [str(output)]


def _reset_gpu_stats(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb(torch_module) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return torch_module.cuda.max_memory_allocated() / (1024**2)


def _execute_benchmark_dataframe(
    *,
    df: pd.DataFrame,
    config: BenchmarkConfig,
    pipeline,
    tokenizer,
    runtime,
    torch_module,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    from kvpress import DecodingPress

    df = df.copy()
    df["predicted_answer"] = None
    df["compression_ratio"] = runtime.compression_ratio
    df["budget"] = config.budget
    df["method"] = config.method
    df["model_name"] = config.model
    df["latency_seconds"] = None
    df["throughput_tokens_per_second"] = None
    df["peak_gpu_memory_mb"] = None
    df["output_tokens"] = None
    df["input_tokens"] = None

    total_examples = len(df)
    total_output_tokens = 0
    inference_start = time.perf_counter()
    peak_memory_overall_mb: float | None = None

    if isinstance(runtime.press, DecodingPress):
        iterator = df.iterrows()
        for index, row in iterator:
            _reset_gpu_stats(torch_module)
            start = time.perf_counter()
            output = pipeline(
                row["context"],
                question=row["question"],
                answer_prefix=row.get("answer_prefix", ""),
                press=runtime.press,
                cache=runtime.cache,
                max_new_tokens=config.max_new_tokens or row.get("max_new_tokens", 32),
                max_context_length=config.max_context_length,
            )
            latency = time.perf_counter() - start
            answer = _normalize_output(output, multi=False)[0]
            output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
            input_tokens = len(tokenizer.encode(row["context"], add_special_tokens=False))
            peak_memory = _peak_gpu_memory_mb(torch_module)
            total_output_tokens += output_tokens
            if peak_memory is not None:
                peak_memory_overall_mb = (
                    peak_memory
                    if peak_memory_overall_mb is None
                    else max(peak_memory_overall_mb, peak_memory)
                )

            df.loc[index, "predicted_answer"] = answer
            df.loc[index, "latency_seconds"] = latency
            df.loc[index, "throughput_tokens_per_second"] = output_tokens / latency if latency > 0 else None
            df.loc[index, "peak_gpu_memory_mb"] = peak_memory
            df.loc[index, "output_tokens"] = output_tokens
            df.loc[index, "input_tokens"] = input_tokens
    else:
        grouped = df.groupby("context", sort=False)
        for context, df_group in grouped:
            _reset_gpu_stats(torch_module)
            start = time.perf_counter()
            output = pipeline(
                context,
                questions=df_group["question"].tolist(),
                answer_prefix=df_group["answer_prefix"].iloc[0] if "answer_prefix" in df_group.columns else "",
                press=runtime.press,
                cache=runtime.cache,
                max_new_tokens=config.max_new_tokens or int(df_group["max_new_tokens"].iloc[0]),
                max_context_length=config.max_context_length,
            )
            latency = time.perf_counter() - start
            answers = _normalize_output(output, multi=True)
            peak_memory = _peak_gpu_memory_mb(torch_module)
            output_tokens_list = [len(tokenizer.encode(answer, add_special_tokens=False)) for answer in answers]
            input_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            total_output_tokens += sum(output_tokens_list)
            if peak_memory is not None:
                peak_memory_overall_mb = (
                    peak_memory
                    if peak_memory_overall_mb is None
                    else max(peak_memory_overall_mb, peak_memory)
                )

            df.loc[df_group.index, "predicted_answer"] = answers
            df.loc[df_group.index, "latency_seconds"] = latency / max(len(df_group), 1)
            df.loc[df_group.index, "throughput_tokens_per_second"] = (
                sum(output_tokens_list) / latency if latency > 0 else None
            )
            df.loc[df_group.index, "peak_gpu_memory_mb"] = peak_memory
            df.loc[df_group.index, "output_tokens"] = output_tokens_list
            df.loc[df_group.index, "input_tokens"] = input_tokens

    scorer = get_scorer(config.dataset)
    metrics = scorer(df)
    total_runtime_seconds = time.perf_counter() - inference_start

    run_stats = {
        "scenario_name": config.scenario_name,
        "dataset": config.dataset,
        "data_dir": config.data_dir,
        "task_prefixes": config.task_prefixes,
        "model": config.model,
        "method": config.method,
        "budget": config.budget,
        "compression_ratio": runtime.compression_ratio,
        "examples": total_examples,
        "total_runtime_seconds": total_runtime_seconds,
        "avg_seconds_per_example": total_runtime_seconds / total_examples if total_examples else None,
        "examples_per_second": total_examples / total_runtime_seconds if total_runtime_seconds > 0 else None,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": total_output_tokens / total_runtime_seconds if total_runtime_seconds > 0 else None,
        "peak_gpu_memory_mb": peak_memory_overall_mb,
        "avg_latency_seconds": pd.to_numeric(df["latency_seconds"], errors="coerce").mean(),
        "avg_throughput_tokens_per_second": pd.to_numeric(
            df["throughput_tokens_per_second"], errors="coerce"
        ).mean(),
    }
    return df, metrics, run_stats


def _save_benchmark_outputs(
    *, df: pd.DataFrame, metrics: dict[str, Any], run_stats: dict[str, Any], config: BenchmarkConfig
) -> tuple[Path, Path]:
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = "__".join(
        [part for part in [
            config.scenario_name,
            config.dataset,
            str(config.data_dir or ""),
            config.model.replace("/", "--"),
            config.method,
            f"budget{config.budget:.2f}",
        ] if part]
    ).strip("_")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.json"
    run_stats_path = run_dir / "run_stats.json"
    config_path = run_dir / "config.yaml"

    df[list(set(df.columns) - {"context"})].to_csv(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with run_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(run_stats, handle, indent=2)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)

    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved run stats to %s", run_stats_path)
    return predictions_path, metrics_path


def run_benchmark_evaluation(config: BenchmarkConfig) -> tuple[Path, Path]:
    import torch
    from kvpress import DecodingPress, ObservedAttentionPress

    from .compat import apply_kvpress_compat_patches
    from .methods import build_method_runtime
    from .runner import load_model_bundle

    _setup_logging(config.verbose)
    _set_seed(config.seed)
    apply_kvpress_compat_patches()

    methods = {m["name"]: m for m in get_method_configs(config.methods_config_path)}
    if config.method not in methods:
        raise ValueError(f"Method '{config.method}' not found in {config.methods_config_path}")

    model, tokenizer, pipeline = load_model_bundle(
        model_name=config.model,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )
    runtime = build_method_runtime(
        methods[config.method],
        budget=config.budget,
        model_config=model.config,
        model_layer_count=len(model.model.layers),
    )
    runtime.model = model
    runtime.tokenizer = tokenizer

    if isinstance(runtime.press, ObservedAttentionPress):
        raise RuntimeError("ObservedAttentionPress benchmark mode is not yet wired in this project.")

    df = _prepare_df(_load_benchmark_df(config, tokenizer), config, runtime)
    df, metrics, run_stats = _execute_benchmark_dataframe(
        df=df,
        config=config,
        pipeline=pipeline,
        tokenizer=tokenizer,
        runtime=runtime,
        torch_module=torch,
    )
    return _save_benchmark_outputs(df=df, metrics=metrics, run_stats=run_stats, config=config)
