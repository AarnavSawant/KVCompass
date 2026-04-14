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
    parser = argparse.ArgumentParser(description="Run a config-driven KVPress benchmark sweep.")
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "benchmark_sweeps.yaml")
    return parser.parse_args()


def main() -> int:
    from kvpress_eval.benchmark_sweep import run_benchmark_sweep

    args = parse_args()
    artifacts = run_benchmark_sweep(resolve_path(args.config))
    print(f"Completed {artifacts.run_count} benchmark runs")
    print(f"Saved summary to {artifacts.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
