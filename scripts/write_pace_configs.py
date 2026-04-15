#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COMPRESSED_METHODS = [
    "snapkv",
    "expected_attention",
    "knorm",
    "tova",
    "streaming_llm",
]


def _base_sweep(model: str, torch_dtype: str, output_dir: str, seed: int) -> dict:
    return {
        "model": model,
        "device": "auto",
        "torch_dtype": torch_dtype,
        "methods_config_path": "configs/methods.yaml",
        "output_dir": output_dir,
        "seed": seed,
        "verbose": False,
    }


def _scenario(name: str, data_dir: str, task_prefixes: list[str], fraction: float, methods: list[str], budgets: dict) -> dict:
    return {
        "name": name,
        "dataset": "ruler",
        "data_dir": data_dir,
        "task_prefixes": task_prefixes,
        "fraction": fraction,
        "methods": methods,
        "budgets": budgets,
    }


def _write_yaml(path: Path, sweep_name: str, base: dict, scenarios: list[dict]) -> None:
    payload = {"sweep": {"name": sweep_name, **base, "scenarios": scenarios}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _compressed_scenarios(prefix_name: str, task_prefixes: list[str], fraction: float) -> list[dict]:
    return [
        _scenario(
            name=f"{prefix_name}_4k",
            data_dir="4096",
            task_prefixes=task_prefixes,
            fraction=fraction,
            methods=COMPRESSED_METHODS,
            budgets={"default": [0.5]},
        ),
        _scenario(
            name=f"{prefix_name}_8k",
            data_dir="8192",
            task_prefixes=task_prefixes,
            fraction=fraction,
            methods=COMPRESSED_METHODS,
            budgets={"default": [0.5]},
        ),
    ]


def _baseline_scenarios(fraction: float) -> list[dict]:
    scenario_specs = [
        ("niah_4k_baseline", "4096", ["niah"]),
        ("niah_8k_baseline", "8192", ["niah"]),
        ("qa_4k_baseline", "4096", ["qa"]),
        ("qa_8k_baseline", "8192", ["qa"]),
        ("vt_4k_baseline", "4096", ["vt"]),
        ("vt_8k_baseline", "8192", ["vt"]),
        ("aggregation_4k_baseline", "4096", ["cwe", "fwe"]),
        ("aggregation_8k_baseline", "8192", ["cwe", "fwe"]),
    ]
    return [
        _scenario(
            name=name,
            data_dir=data_dir,
            task_prefixes=task_prefixes,
            fraction=fraction,
            methods=["no_compression"],
            budgets={"no_compression": [1.0]},
        )
        for name, data_dir, task_prefixes in scenario_specs
    ]


def _core_matrix_scenarios(fraction: float) -> list[dict]:
    return [
        _scenario(
            name="needle_in_a_haystack_4k",
            data_dir="4096",
            task_prefixes=["niah"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="needle_in_a_haystack_8k",
            data_dir="8192",
            task_prefixes=["niah"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="question_answering_4k",
            data_dir="4096",
            task_prefixes=["qa"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="question_answering_8k",
            data_dir="8192",
            task_prefixes=["qa"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="multi_hop_tracing_4k",
            data_dir="4096",
            task_prefixes=["vt"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="multi_hop_tracing_8k",
            data_dir="8192",
            task_prefixes=["vt"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="aggregation_4k",
            data_dir="4096",
            task_prefixes=["cwe", "fwe"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
        _scenario(
            name="aggregation_8k",
            data_dir="8192",
            task_prefixes=["cwe", "fwe"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.5]},
        ),
    ]


def _budget_sensitivity_scenarios(fraction: float, data_dir: str) -> list[dict]:
    length_label = "4k" if data_dir == "4096" else "8k"
    return [
        _scenario(
            name=f"needle_in_a_haystack_{length_label}_budget_sensitivity",
            data_dir=data_dir,
            task_prefixes=["niah"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.75, 0.5, 0.25]},
        ),
        _scenario(
            name=f"question_answering_{length_label}_budget_sensitivity",
            data_dir=data_dir,
            task_prefixes=["qa"],
            fraction=fraction,
            methods=["no_compression", *COMPRESSED_METHODS],
            budgets={"no_compression": [1.0], "default": [0.75, 0.5, 0.25]},
        ),
    ]


def write_pace_configs(*, dest_dir: Path, output_dir: str, model: str, torch_dtype: str, fraction: float, seed: int) -> list[Path]:
    base = _base_sweep(model=model, torch_dtype=torch_dtype, output_dir=output_dir, seed=seed)
    written: list[Path] = []

    configs: list[tuple[str, str, list[dict]]] = [
        ("benchmark_sweeps.pace_assignment_1.yaml", "assignment_1", _compressed_scenarios("needle_in_a_haystack", ["niah"], fraction)),
        ("benchmark_sweeps.pace_assignment_2.yaml", "assignment_2", _compressed_scenarios("question_answering", ["qa"], fraction)),
        ("benchmark_sweeps.pace_assignment_3.yaml", "assignment_3", _compressed_scenarios("multi_hop_tracing", ["vt"], fraction)),
        ("benchmark_sweeps.pace_assignment_4.yaml", "assignment_4", _baseline_scenarios(fraction)),
        ("benchmark_sweeps.pace_assignment_5.yaml", "assignment_5", _compressed_scenarios("aggregation", ["cwe", "fwe"], fraction)),
        ("benchmark_sweeps.pace_full_matrix.yaml", "pace_full_matrix", _core_matrix_scenarios(fraction)),
        ("benchmark_sweeps.pace_budget_sensitivity_4k.yaml", "pace_budget_sensitivity_4k", _budget_sensitivity_scenarios(fraction, "4096")),
        ("benchmark_sweeps.pace_budget_sensitivity_8k.yaml", "pace_budget_sensitivity_8k", _budget_sensitivity_scenarios(fraction, "8192")),
    ]

    for filename, sweep_name, scenarios in configs:
        path = dest_dir / filename
        _write_yaml(path, sweep_name, base, scenarios)
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write standardized PACE benchmark sweep configs.")
    parser.add_argument("--dest-dir", default="configs", help="Directory where YAML configs will be written.")
    parser.add_argument("--output-dir", default="results/benchmark_eval", help="Output directory embedded in the configs.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = write_pace_configs(
        dest_dir=Path(args.dest_dir),
        output_dir=args.output_dir,
        model=args.model,
        torch_dtype=args.torch_dtype,
        fraction=args.fraction,
        seed=args.seed,
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
