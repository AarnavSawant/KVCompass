from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .io_utils import read_results


def _save_metric_plot(frame: pd.DataFrame, metric: str, ylabel: str, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    clean = frame[frame["status"] == "ok"].copy()
    clean = clean.dropna(subset=[metric])
    if clean.empty:
        return None

    summary = (
        clean.groupby(["scenario", "method", "budget"], dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(["scenario", "method", "budget"])
    )

    scenarios = list(summary["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(8, 4 * len(scenarios)), squeeze=False)
    for axis, scenario in zip(axes.flatten(), scenarios):
        subset = summary[summary["scenario"] == scenario]
        for method, method_rows in subset.groupby("method"):
            axis.plot(method_rows["budget"], method_rows[metric], marker="o", label=method)
        axis.set_title(scenario)
        axis.set_xlabel("Budget")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    output_path = output_dir / f"{metric}_vs_budget.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def create_plots(raw_results_path: str | Path, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_root / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    frame = read_results(raw_results_path)

    plots: list[Path] = []
    for metric, ylabel in [
        ("quality_score", "Quality Score"),
        ("latency_seconds", "Latency (s)"),
        ("peak_gpu_memory_mb", "Peak GPU Memory (MB)"),
    ]:
        plot_path = _save_metric_plot(frame, metric, ylabel, output_root)
        if plot_path is not None:
            plots.append(plot_path)
    return plots
