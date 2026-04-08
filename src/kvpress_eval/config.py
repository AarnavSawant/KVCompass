from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, found {type(data).__name__}")
    return data


def get_method_configs(path: str | Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    methods = data.get("methods", [])
    if not isinstance(methods, list):
        raise ValueError("methods.yaml must define a top-level 'methods' list")
    return methods


def get_scenario_configs(path: str | Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenarios.yaml must define a top-level 'scenarios' list")
    return scenarios
