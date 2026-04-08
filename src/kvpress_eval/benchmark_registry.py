from __future__ import annotations

from importlib import import_module
from typing import Callable

DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
}

_SCORER_SPECS = {
    "loogle": ("kvpress_eval.benchmarks.loogle.calculate_metrics", "calculate_metrics"),
    "ruler": ("kvpress_eval.benchmarks.ruler.calculate_metrics", "calculate_metrics"),
    "zero_scrolls": ("kvpress_eval.benchmarks.zero_scrolls.calculate_metrics", "calculate_metrics"),
    "infinitebench": ("kvpress_eval.benchmarks.infinite_bench.calculate_metrics", "calculate_metrics"),
    "longbench": ("kvpress_eval.benchmarks.longbench.calculate_metrics", "calculate_metrics"),
    "longbench-e": ("kvpress_eval.benchmarks.longbench.calculate_metrics", "calculate_metrics_e"),
    "longbench-v2": ("kvpress_eval.benchmarks.longbenchv2.calculate_metrics", "calculate_metrics"),
    "needle_in_haystack": ("kvpress_eval.benchmarks.needle_in_haystack.calculate_metrics", "calculate_metrics"),
    "aime25": ("kvpress_eval.benchmarks.aime25.calculate_metrics", "calculate_metrics"),
    "math500": ("kvpress_eval.benchmarks.math500.calculate_metrics", "calculate_metrics"),
}


def get_scorer(dataset: str) -> Callable:
    if dataset not in _SCORER_SPECS:
        raise ValueError(f"Unsupported scorer dataset: {dataset}")
    module_name, attr = _SCORER_SPECS[dataset]
    module = import_module(module_name)
    return getattr(module, attr)
