from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import kvpress


@dataclass
class MethodRuntime:
    name: str
    budget: float
    press: Any | None = None
    cache: Any | None = None
    cache_type: str = "dynamic"
    compression_ratio: float = 0.0


def build_method_runtime(
    method_config: dict[str, Any],
    budget: float,
    model_config: Any | None = None,
    model_layer_count: int | None = None,
) -> MethodRuntime:
    del model_config, model_layer_count

    kind = method_config.get("kind", "press")
    name = method_config["name"]
    compression_ratio = max(0.0, min(1.0, 1.0 - float(budget)))

    if kind == "none":
        return MethodRuntime(name=name, budget=float(budget), cache_type="dynamic", compression_ratio=0.0)

    press_class = getattr(kvpress, method_config["press_class"])
    params = dict(method_config.get("params", {}))
    params["compression_ratio"] = compression_ratio
    press = press_class(**params)
    return MethodRuntime(
        name=name,
        budget=float(budget),
        press=press,
        cache_type="dynamic",
        compression_ratio=compression_ratio,
    )
