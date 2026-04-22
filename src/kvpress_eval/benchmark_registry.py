from __future__ import annotations

from importlib import import_module
from typing import Callable

DATASET_REGISTRY = {
    "ruler": "simonjegou/ruler",
}

_SCORER_SPECS = {
    "ruler": ("kvpress_eval.benchmarks.ruler.calculate_metrics", "calculate_metrics"),
}


def get_scorer(dataset: str) -> Callable:
    if dataset not in _SCORER_SPECS:
        raise ValueError(f"Unsupported scorer dataset: {dataset}")
    module_name, attr = _SCORER_SPECS[dataset]
    module = import_module(module_name)
    return getattr(module, attr)
