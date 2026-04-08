#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvpress_eval.runner import configure_logging, run_evaluation


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a budget-oriented sweep across configured scenarios.")
    parser.add_argument("--model", required=True, help="Model name or local path for AutoModelForCausalLM.")
    parser.add_argument("--methods-config", default=PROJECT_ROOT / "configs" / "methods.yaml")
    parser.add_argument("--scenarios-config", default=PROJECT_ROOT / "configs" / "scenarios.yaml")
    parser.add_argument("--scenario", action="append", help="Scenario name to include. Repeatable.")
    parser.add_argument("--method", action="append", help="Method name to include. Repeatable.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "results" / "raw" / f"budget_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    parser.add_argument("--run-name", default="budget_sweep")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    artifacts = run_evaluation(
        model_name=args.model,
        methods_config_path=resolve_path(args.methods_config),
        scenarios_config_path=resolve_path(args.scenarios_config),
        output_path=resolve_path(args.output),
        scenario_filter=args.scenario,
        method_filter=args.method,
        max_cases=args.max_cases,
        device=args.device,
        torch_dtype=args.torch_dtype,
        run_name=args.run_name,
    )
    print(f"Saved {artifacts.row_count} result rows to {artifacts.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
