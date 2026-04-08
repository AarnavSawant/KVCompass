from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import KVPressTextGenerationPipeline

from .compat import apply_kvpress_compat_patches
from .config import get_method_configs, get_scenario_configs
from .evaluate import score_prediction
from .io_utils import write_rows
from .methods import MethodRuntime, build_method_runtime
from .scenarios import build_example, select_questions

LOGGER = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    results_path: Path
    row_count: int


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_model_bundle(model_name: str, device: str = "auto", torch_dtype: str = "auto") -> tuple[Any, Any, Any]:
    apply_kvpress_compat_patches()
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    elif torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    if device == "auto" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    elif device not in {"auto", "cpu"}:
        kwargs["device_map"] = device

    LOGGER.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    return model, tokenizer, pipeline


def _peak_gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    peak = torch.cuda.max_memory_allocated()
    return peak / (1024 ** 2)


def _reset_gpu_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _single_run(
    *,
    pipeline: Any,
    method_runtime: MethodRuntime,
    scenario_name: str,
    context_length: int,
    max_new_tokens: int,
    repeats: int,
    run_name: str,
    model_name: str,
) -> list[dict[str, Any]]:
    example = build_example(scenario_name, context_length)
    questions = list(select_questions(example, repeats))
    rows: list[dict[str, Any]] = []

    for repeat_index, question in enumerate(questions, start=1):
        _reset_gpu_stats()
        start = time.perf_counter()
        generated = pipeline(
            example.context,
            question=question,
            press=method_runtime.press,
            cache=method_runtime.cache,
            max_context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
        latency = time.perf_counter() - start
        if isinstance(generated, dict):
            answer = str(generated.get("answer", generated))
        elif isinstance(generated, list):
            answer = str(generated[0])
        else:
            answer = str(generated)
        output_tokens = len(pipeline.tokenizer.encode(answer, add_special_tokens=False))
        input_tokens = len(pipeline.tokenizer.encode(example.context, add_special_tokens=False))
        quality = score_prediction(answer, example.reference_answer, example.quality_label)
        throughput = output_tokens / latency if latency > 0 else None

        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_name": run_name,
                "model_name": model_name,
                "scenario": scenario_name,
                "method": method_runtime.name,
                "budget": method_runtime.budget,
                "context_length": context_length,
                "repeat_index": repeat_index,
                "latency_seconds": latency,
                "throughput_tokens_per_second": throughput,
                "peak_gpu_memory_mb": _peak_gpu_memory_mb(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "compression_ratio": method_runtime.compression_ratio,
                "cache_type": method_runtime.cache_type,
                "quality_score": quality.score,
                "quality_label": quality.label,
                "generated_text": answer,
                "reference_answer": example.reference_answer,
                "status": "ok",
                "error_message": "",
            }
        )
    return rows


def run_evaluation(
    *,
    model_name: str,
    methods_config_path: str | Path,
    scenarios_config_path: str | Path,
    output_path: str | Path,
    scenario_filter: list[str] | None = None,
    method_filter: list[str] | None = None,
    max_cases: int | None = None,
    device: str = "auto",
    torch_dtype: str = "auto",
    run_name: str = "kvpress_eval",
) -> RunArtifacts:
    method_configs = get_method_configs(methods_config_path)
    scenario_configs = get_scenario_configs(scenarios_config_path)

    if scenario_filter:
        allowed = set(scenario_filter)
        scenario_configs = [cfg for cfg in scenario_configs if cfg["name"] in allowed]
    if method_filter:
        allowed = set(method_filter)
        method_configs = [cfg for cfg in method_configs if cfg["name"] in allowed]

    if not scenario_configs:
        raise ValueError("No scenarios selected")
    if not method_configs:
        raise ValueError("No methods selected")

    model, _, pipeline = load_model_bundle(model_name=model_name, device=device, torch_dtype=torch_dtype)
    model_layer_count = len(model.model.layers)
    rows: list[dict[str, Any]] = []
    completed = 0

    for scenario in scenario_configs:
        scenario_name = scenario["name"]
        for context_length in scenario["context_lengths"]:
            for budget in scenario["budgets"]:
                for method in method_configs:
                    if max_cases is not None and completed >= max_cases:
                        LOGGER.info("Reached max_cases=%s", max_cases)
                        results_path = write_rows(rows, output_path)
                        return RunArtifacts(results_path=results_path, row_count=len(rows))

                    LOGGER.info(
                        "Running scenario=%s method=%s budget=%s context_length=%s",
                        scenario_name,
                        method["name"],
                        budget,
                        context_length,
                    )
                    try:
                        runtime = build_method_runtime(
                            method,
                            budget=budget,
                            model_config=model.config,
                            model_layer_count=model_layer_count,
                        )
                        rows.extend(
                            _single_run(
                                pipeline=pipeline,
                                method_runtime=runtime,
                                scenario_name=scenario_name,
                                context_length=int(context_length),
                                max_new_tokens=int(scenario.get("max_new_tokens", 32)),
                                repeats=int(scenario.get("repeats", 1)),
                                run_name=run_name,
                                model_name=model_name,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - defensive logging path
                        LOGGER.exception("Evaluation failed for %s", method["name"])
                        rows.append(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "run_name": run_name,
                                "model_name": model_name,
                                "scenario": scenario_name,
                                "method": method["name"],
                                "budget": budget,
                                "context_length": context_length,
                                "repeat_index": 1,
                                "latency_seconds": None,
                                "throughput_tokens_per_second": None,
                                "peak_gpu_memory_mb": _peak_gpu_memory_mb(),
                                "input_tokens": None,
                                "output_tokens": None,
                                "compression_ratio": max(0.0, min(1.0, 1.0 - float(budget))),
                                "cache_type": method.get("kind", "press"),
                                "quality_score": None,
                                "quality_label": "",
                                "generated_text": "",
                                "reference_answer": "",
                                "status": "error",
                                "error_message": str(exc),
                            }
                        )
                    completed += 1

    results_path = write_rows(rows, output_path)
    return RunArtifacts(results_path=results_path, row_count=len(rows))
