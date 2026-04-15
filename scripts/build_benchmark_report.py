#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a benchmark report from benchmark sweep summary CSVs.")
    parser.add_argument("--summary-dir", default=PROJECT_ROOT / "results" / "benchmark_eval")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "results" / "benchmark_report")
    return parser.parse_args()


def main() -> int:
    from kvpress_eval.benchmark_reporting import build_benchmark_report

    args = parse_args()
    artifacts = build_benchmark_report(
        summary_dir=resolve_path(args.summary_dir),
        output_dir=resolve_path(args.output_dir),
    )
    print(f"Combined summary: {artifacts.combined_summary_csv}")
    print(f"Task metrics: {artifacts.task_metrics_csv}")
    print(f"Baseline-relative table: {artifacts.baseline_relative_csv}")
    print(f"Recommendations: {artifacts.recommendations_csv}")
    print(f"Failure examples: {artifacts.failure_examples_csv}")
    print(f"Failure mode summary: {artifacts.failure_mode_summary_csv}")
    for plot_path in artifacts.plot_paths:
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
