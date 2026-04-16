"""
Confusion matrix: latency speedup vs quality retention per press method and task scenario.

Reads results/benchmark_eval/ and produces results/plots/confusion_matrix.png.

Rows    = press methods
Columns = task scenario × context length
Color   = latency speedup relative to no_compression baseline (blue = faster)
Text    = speedup in top half of cell, quality retention % in bottom half
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark_eval"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "plots"

TASK_LABEL = {
    "niah": "Needle in\na Haystack",
    "qa": "Question\nAnswering",
    "vt": "Multi-Hop\nTracing",
    "aggregation": "Aggregation",
}

METHOD_LABEL = {
    "snapkv": "SnapKV",
    "expected_attention": "Expected\nAttention",
    "knorm": "KNorm",
    "tova": "TOVA",
    "streaming_llm": "StreamingLLM",
}

METHOD_ORDER = ["snapkv", "expected_attention", "knorm", "tova", "streaming_llm"]


def task_key(task_prefixes: list[str]) -> str:
    """Normalize task_prefixes to a single key."""
    joined = "_".join(sorted(task_prefixes))
    if any(p.startswith("niah") for p in task_prefixes):
        return "niah"
    if joined in ("cwe_fwe", "cwe", "fwe"):
        return "aggregation"
    if "qa" in task_prefixes:
        return "qa"
    if "vt" in task_prefixes:
        return "vt"
    return joined


def avg_quality(metrics: dict) -> float:
    """Mean of every leaf numeric metric value across all subtasks."""
    values = []
    for subtask_metrics in metrics.values():
        if isinstance(subtask_metrics, dict):
            values.extend(v for v in subtask_metrics.values() if isinstance(v, (int, float)))
        elif isinstance(subtask_metrics, (int, float)):
            values.append(subtask_metrics)
    return float(np.mean(values)) if values else float("nan")


def load_runs() -> list[dict]:
    runs = []
    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        stats_path = run_dir / "run_stats.json"
        metrics_path = run_dir / "metrics.json"
        if not stats_path.exists() or not metrics_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        runs.append(
            {
                "method": stats["method"],
                "data_dir": str(stats["data_dir"]),
                "task": task_key(stats["task_prefixes"]),
                "avg_latency": stats["avg_latency_seconds"],
                "quality": avg_quality(metrics),
            }
        )
    return runs


def build_matrices(runs: list[dict]):
    # Separate baselines from compressed runs
    baselines: dict[tuple, float] = {}   # (task, data_dir) -> avg_latency
    baseline_quality: dict[tuple, float] = {}
    compressed: dict[tuple, dict] = {}  # (task, data_dir, method) -> run

    for r in runs:
        key = (r["task"], r["data_dir"])
        if r["method"] == "no_compression":
            baselines[key] = r["avg_latency"]
            baseline_quality[key] = r["quality"]
        else:
            compressed[(r["task"], r["data_dir"], r["method"])] = r

    # Determine column order: unique (task, data_dir) pairs with data, sorted
    task_order = ["niah", "qa", "vt", "aggregation"]
    ctx_order = ["4096", "8192"]
    columns = [
        (task, ctx)
        for task in task_order
        for ctx in ctx_order
        if (task, ctx) in baselines and any(
            (task, ctx, m) in compressed for m in METHOD_ORDER
        )
    ]

    n_rows = len(METHOD_ORDER)
    n_cols = len(columns)
    speedup_mat = np.full((n_rows, n_cols), np.nan)
    quality_mat = np.full((n_rows, n_cols), np.nan)

    for ci, (task, ctx) in enumerate(columns):
        base_lat = baselines[(task, ctx)]
        base_q = baseline_quality[(task, ctx)]
        for ri, method in enumerate(METHOD_ORDER):
            run = compressed.get((task, ctx, method))
            if run is None:
                continue
            speedup_mat[ri, ci] = base_lat / run["avg_latency"]
            quality_mat[ri, ci] = run["quality"] / base_q * 100.0 if base_q > 0 else np.nan

    return speedup_mat, quality_mat, columns


def make_plot(speedup_mat, quality_mat, columns):
    n_rows, n_cols = speedup_mat.shape
    method_labels = [METHOD_LABEL.get(m, m) for m in METHOD_ORDER]
    col_labels = [
        f"{TASK_LABEL.get(task, task)}\n{ctx[:1]}K ctx"
        for task, ctx in columns
    ]

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.8), max(5, n_rows * 1.4)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Color scale: white at 1× speedup, deep blue at high speedup, light red below 1×
    # We cap the display range at 30× so the scale stays readable
    vmin, vmax = 0.5, 30.0
    cmap = plt.cm.RdYlGn  # red=slow, green=fast

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(speedup_mat, cmap=cmap, norm=norm, aspect="auto")

    # Grid lines
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="#0f1117", linewidth=1.5)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="#0f1117", linewidth=1.5)

    # Cell annotations
    for ri in range(n_rows):
        for ci in range(n_cols):
            sp = speedup_mat[ri, ci]
            qu = quality_mat[ri, ci]
            if np.isnan(sp):
                ax.text(ci, ri, "N/A", ha="center", va="center", fontsize=9, color="#888888")
                continue

            # Choose text color based on background brightness
            norm_val = (np.log10(sp) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
            bg_color = cmap(norm_val)
            brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            txt_color = "#111111" if brightness > 0.5 else "#eeeeee"

            # Top: speedup
            sp_str = f"{sp:.1f}×" if sp < 100 else f"{sp:.0f}×"
            ax.text(ci, ri - 0.15, sp_str, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)

            # Bottom: quality retention
            qu_str = f"{qu:.0f}% quality" if not np.isnan(qu) else ""
            ax.text(ci, ri + 0.22, qu_str, ha="center", va="center",
                    fontsize=8.5, color=txt_color, alpha=0.88)

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, color="white", fontsize=9.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(method_labels, color="white", fontsize=10)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                        ticks=[1, 2, 5, 10, 20])
    cbar.ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×"], color="white", fontsize=9)
    cbar.set_label("Latency Speedup vs Baseline", color="white", fontsize=10)
    cbar.outline.set_edgecolor("white")
    cbar.ax.yaxis.set_tick_params(color="white")

    ax.set_title(
        "KV-Cache Press Methods: Latency Speedup vs Quality Retention\n"
        "(color = speedup; text = speedup / quality retained relative to no-compression baseline)",
        color="white",
        fontsize=11,
        pad=18,
    )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


def main():
    runs = load_runs()
    speedup_mat, quality_mat, columns = build_matrices(runs)
    make_plot(speedup_mat, quality_mat, columns)


if __name__ == "__main__":
    main()