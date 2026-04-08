from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from transformers import QuantizedCache

import kvpress

LOGGER = logging.getLogger(__name__)


@dataclass
class MethodRuntime:
    name: str
    budget: float
    press: Any | None = None
    cache: Any | None = None
    cache_type: str = "dynamic"
    compression_ratio: float = 0.0


def _build_press_from_nested(spec: dict[str, Any], compression_ratio: float):
    press_class = getattr(kvpress, spec["press_class"])
    params = dict(spec.get("params", {}))
    params["compression_ratio"] = compression_ratio
    return press_class(**params)


def _layer_schedule(schedule: str, compression_ratio: float, layer_count: int) -> list[float]:
    if schedule == "uniform":
        return [compression_ratio] * layer_count
    if schedule == "linear_decay":
        if layer_count == 1:
            return [compression_ratio]
        values = []
        for idx in range(layer_count):
            weight = 1.0 - (idx / (layer_count - 1)) * 0.5
            values.append(min(max(compression_ratio * weight, 0.0), 0.95))
        return values
    raise ValueError(f"Unsupported layerwise schedule: {schedule}")


def build_method_runtime(
    method_config: dict[str, Any],
    budget: float,
    model_config: Any | None = None,
    model_layer_count: int | None = None,
) -> MethodRuntime:
    kind = method_config.get("kind", "press")
    name = method_config["name"]
    compression_ratio = max(0.0, min(1.0, 1.0 - float(budget)))

    if kind == "none":
        return MethodRuntime(name=name, budget=float(budget), cache_type="dynamic", compression_ratio=0.0)

    if kind == "quantized_cache":
        if model_config is None:
            raise ValueError("A loaded model config is required for quantized-cache methods")
        backend = method_config.get("backend", "quanto")
        try:
            cache = QuantizedCache(
                backend=backend,
                config=model_config,
                nbits=int(method_config.get("nbits", 4)),
                q_group_size=int(method_config.get("q_group_size", 64)),
                residual_length=int(method_config.get("residual_length", 128)),
            )
        except ImportError as exc:
            raise RuntimeError(
                f"Quantized cache backend '{backend}' is unavailable. Install its backend package "
                "for example `optimum-quanto` for the default quanto backend."
            ) from exc
        return MethodRuntime(
            name=name,
            budget=float(budget),
            cache=cache,
            cache_type=f"quantized_{method_config.get('nbits', 4)}bit",
            compression_ratio=compression_ratio,
        )

    press_class_name = method_config.get("press_class")
    params = dict(method_config.get("params", {}))

    if press_class_name == "AdaKVPress":
        nested_spec = params.pop("nested_press")
        nested_press = _build_press_from_nested(nested_spec, compression_ratio)
        press = kvpress.AdaKVPress(press=nested_press, **params)
        return MethodRuntime(
            name=name,
            budget=float(budget),
            press=press,
            cache_type="dynamic",
            compression_ratio=compression_ratio,
        )

    if press_class_name == "PerLayerCompressionPress":
        if model_layer_count is None:
            raise ValueError("Layerwise methods require a loaded model so the layer count is known")
        if getattr(model_config, "_attn_implementation", None) != "flash_attention_2":
            raise RuntimeError(
                "PerLayerCompressionPress currently requires flash attention. "
                "This model/backend is using "
                f"{getattr(model_config, '_attn_implementation', 'unknown')}."
            )
        base_press_class = getattr(kvpress, params.pop("base_press_class"))
        base_press_params = dict(params.pop("base_press_params", {}))
        schedule = params.pop("schedule", "linear_decay")
        base_press = base_press_class(compression_ratio=compression_ratio, **base_press_params)
        compression_ratios = _layer_schedule(schedule, compression_ratio, model_layer_count)
        press = kvpress.PerLayerCompressionPress(press=base_press, compression_ratios=compression_ratios)
        return MethodRuntime(
            name=name,
            budget=float(budget),
            press=press,
            cache_type="dynamic",
            compression_ratio=compression_ratio,
        )

    press_class = getattr(kvpress, press_class_name)
    params["compression_ratio"] = compression_ratio
    press = press_class(**params)
    return MethodRuntime(
        name=name,
        budget=float(budget),
        press=press,
        cache_type="dynamic",
        compression_ratio=compression_ratio,
    )


def summarize_method(method_config: dict[str, Any]) -> str:
    return f"{method_config['name']} ({method_config.get('description', method_config.get('kind', 'press'))})"
