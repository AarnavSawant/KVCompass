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
    parser = argparse.ArgumentParser(description="Run KVPress benchmark evaluation using the repo's benchmark metrics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--query-aware", action="store_true")
    parser.add_argument("--needle-depth", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable the Transformer KV cache and generate with use_cache=False.",
    )
    parser.add_argument("--methods-config", default=PROJECT_ROOT / "configs" / "methods.yaml")
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "results" / "benchmark_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    from kvpress_eval.benchmark_eval import BenchmarkConfig, run_benchmark_evaluation

    args = parse_args()
    config = BenchmarkConfig(
        dataset=args.dataset,
        model=args.model,
        method=args.method,
        budget=args.budget,
        data_dir=args.data_dir,
        fraction=args.fraction,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        query_aware=args.query_aware,
        needle_depth=args.needle_depth,
        device=args.device,
        torch_dtype=args.torch_dtype,
        use_kv_cache=not args.no_kv_cache,
        output_dir=str(resolve_path(args.output_dir)),
        methods_config_path=str(resolve_path(args.methods_config)),
        seed=args.seed,
        verbose=args.verbose,
    )
    predictions_path, metrics_path = run_benchmark_evaluation(config)
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
