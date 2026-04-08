from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .benchmark_registry import DATASET_REGISTRY, get_scorer
from .config import get_method_configs

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    dataset: str
    model: str
    method: str
    budget: float = 0.5
    data_dir: str | None = None
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

    df = _load_benchmark_df(config, tokenizer)
    df = _prepare_df(df, config, runtime)
    df["predicted_answer"] = None
    df["compression_ratio"] = runtime.compression_ratio
    df["budget"] = config.budget
    df["method"] = config.method
    df["model_name"] = config.model

    if isinstance(runtime.press, DecodingPress):
        iterator = df.iterrows()
        for index, row in iterator:
            output = pipeline(
                row["context"],
                question=row["question"],
                answer_prefix=row.get("answer_prefix", ""),
                press=runtime.press,
                cache=runtime.cache,
                max_new_tokens=config.max_new_tokens or row.get("max_new_tokens", 32),
                max_context_length=config.max_context_length,
            )
            df.loc[index, "predicted_answer"] = _normalize_output(output, multi=False)[0]
    else:
        grouped = df.groupby("context")
        for context, df_group in grouped:
            output = pipeline(
                context,
                questions=df_group["question"].tolist(),
                answer_prefix=df_group["answer_prefix"].iloc[0] if "answer_prefix" in df_group.columns else "",
                press=runtime.press,
                cache=runtime.cache,
                max_new_tokens=config.max_new_tokens or int(df_group["max_new_tokens"].iloc[0]),
                max_context_length=config.max_context_length,
            )
            answers = _normalize_output(output, multi=True)
            df.loc[df_group.index, "predicted_answer"] = answers

    scorer = get_scorer(config.dataset)
    metrics = scorer(df)

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = "__".join(
        [
            config.dataset,
            str(config.data_dir or ""),
            config.model.replace("/", "--"),
            config.method,
            f"budget{config.budget:.2f}",
        ]
    ).strip("_")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"

    df[list(set(df.columns) - {"context"})].to_csv(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)

    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved metrics to %s", metrics_path)
    return predictions_path, metrics_path
