#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvpress_eval.plotting import create_plots


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate score/latency/memory plots from raw evaluation results.")
    parser.add_argument("--input", required=True, help="Raw results CSV.")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "results" / "plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_paths = create_plots(resolve_path(args.input), resolve_path(args.output_dir))
    for output_path in output_paths:
        print(f"Created plot: {output_path}")
    if not output_paths:
        print("No plots were generated. Check whether the input CSV has successful rows with metric values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
