#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvpress_eval.aggregate import aggregate_results


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw KVPress evaluation CSVs into a summary table.")
    parser.add_argument("--input", required=True, help="Raw CSV produced by run_kvpress_eval.py or run_budget_sweep.py.")
    parser.add_argument("--output", default=PROJECT_ROOT / "results" / "summary" / "summary.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = aggregate_results(resolve_path(args.input), resolve_path(args.output))
    print(f"Saved summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
