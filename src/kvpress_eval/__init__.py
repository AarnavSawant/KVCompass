"""Lightweight workload-aware KVPress evaluation package."""

__all__ = [
    "aggregate_results",
    "build_benchmark_report",
    "build_recommendations",
    "create_plots",
    "run_evaluation",
]


def __getattr__(name: str):
    if name == "aggregate_results":
        from .aggregate import aggregate_results

        return aggregate_results
    if name == "build_benchmark_report":
        from .benchmark_reporting import build_benchmark_report

        return build_benchmark_report
    if name == "build_recommendations":
        from .recommendations import build_recommendations

        return build_recommendations
    if name == "create_plots":
        from .plotting import create_plots

        return create_plots
    if name == "run_evaluation":
        from .runner import run_evaluation

        return run_evaluation
    raise AttributeError(name)
