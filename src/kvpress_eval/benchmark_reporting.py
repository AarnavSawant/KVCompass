from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CATEGORY_DISPLAY = {
    "niah": "Needle In A Haystack",
    "qa": "Question Answering",
    "vt": "Multi-Hop Tracing",
    "aggregation": "Aggregation",
}


@dataclass
class BenchmarkReportArtifacts:
    combined_summary_csv: Path
    task_metrics_csv: Path
    baseline_relative_csv: Path
    recommendations_csv: Path
    failure_examples_csv: Path
    failure_mode_summary_csv: Path
    output_dir: Path
    plot_paths: list[Path]


def _ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _infer_main_category(task_prefixes: str, scenario_name: str) -> str:
    prefixes = [part.strip() for part in str(task_prefixes or "").split(",") if part.strip()]
    prefix_set = set(prefixes)
    if "niah" in prefix_set or "needle" in scenario_name:
        return "Needle In A Haystack"
    if "qa" in prefix_set or "question_answering" in scenario_name:
        return "Question Answering"
    if "vt" in prefix_set or "tracing" in scenario_name:
        return "Multi-Hop Tracing"
    if prefix_set.intersection({"cwe", "fwe"}) or "aggregation" in scenario_name:
        return "Aggregation"
    return "Other"


def _infer_context_length(data_dir: Any, scenario_name: str) -> str:
    if pd.notna(data_dir) and str(data_dir):
        return str(data_dir)
    lowered = str(scenario_name or "").lower()
    if "8k" in lowered:
        return "8192"
    if "4k" in lowered:
        return "4096"
    return ""


def _iter_metric_entries(payload: Any, parent_key: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        numeric_values = {k: v for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if numeric_values:
            task_name = parent_key or "overall"
            for metric_name, score in numeric_values.items():
                rows.append({"task_name": task_name, "metric_name": metric_name, "score": float(score)})
        for key, value in payload.items():
            if isinstance(value, dict):
                child_key = key if not parent_key else key
                rows.extend(_iter_metric_entries(value, parent_key=child_key))
    return rows


def _load_summary_frames(summary_dir: Path) -> pd.DataFrame:
    summary_paths = sorted(summary_dir.glob("*__summary.csv"))
    if not summary_paths:
        return pd.DataFrame()
    frames = []
    for path in summary_paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_summary_csv"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_combined_summary(summary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _load_summary_frames(summary_dir)
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary["scenario_name"] = summary["scenario_name"].fillna("")
    summary["task_prefixes"] = summary["task_prefixes"].fillna("")
    summary["context_length"] = summary.apply(
        lambda row: _infer_context_length(row.get("data_dir"), row.get("scenario_name", "")),
        axis=1,
    )
    summary["main_category"] = summary.apply(
        lambda row: _infer_main_category(row.get("task_prefixes", ""), row.get("scenario_name", "")),
        axis=1,
    )

    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        metrics_path = Path(row["metrics_path"])
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics_payload = json.load(handle)
        metric_entries = _iter_metric_entries(metrics_payload)
        if not metric_entries:
            continue
        scores = [entry["score"] for entry in metric_entries]
        run_rows.append(
            {
                **row.to_dict(),
                "quality_score": float(sum(scores) / len(scores)),
                "task_count": len(metric_entries),
            }
        )
        for entry in metric_entries:
            task_rows.append(
                {
                    "scenario_name": row["scenario_name"],
                    "main_category": row["main_category"],
                    "context_length": row["context_length"],
                    "method": row["method"],
                    "budget": row["budget"],
                    "task_name": entry["task_name"],
                    "metric_name": entry["metric_name"],
                    "score": entry["score"],
                }
            )

    combined = pd.DataFrame(run_rows)
    tasks = pd.DataFrame(task_rows)
    return combined, tasks


def _compute_baseline_relative(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    baseline = combined[combined["method"] == "no_compression"][
        ["main_category", "context_length", "quality_score", "avg_latency_seconds", "peak_gpu_memory_mb"]
    ].rename(
        columns={
            "quality_score": "baseline_quality_score",
            "avg_latency_seconds": "baseline_avg_latency_seconds",
            "peak_gpu_memory_mb": "baseline_peak_gpu_memory_mb",
        }
    )

    merged = combined.merge(
        baseline,
        how="left",
        on=["main_category", "context_length"],
    )
    merged["quality_drop_absolute"] = merged["baseline_quality_score"] - merged["quality_score"]
    merged["quality_drop_pct"] = (
        merged["quality_drop_absolute"] / merged["baseline_quality_score"].replace(0, pd.NA) * 100.0
    )
    merged["latency_delta_seconds"] = merged["avg_latency_seconds"] - merged["baseline_avg_latency_seconds"]
    merged["latency_improvement_pct"] = (
        (merged["baseline_avg_latency_seconds"] - merged["avg_latency_seconds"])
        / merged["baseline_avg_latency_seconds"].replace(0, pd.NA)
        * 100.0
    )
    merged["memory_saved_mb"] = merged["baseline_peak_gpu_memory_mb"] - merged["peak_gpu_memory_mb"]
    merged["memory_saved_pct"] = (
        merged["memory_saved_mb"] / merged["baseline_peak_gpu_memory_mb"].replace(0, pd.NA) * 100.0
    )
    return merged


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    series = series.astype(float)
    if series.nunique(dropna=True) <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    low = series.min()
    high = series.max()
    scaled = (series - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def _build_recommendations(baseline_relative: pd.DataFrame) -> pd.DataFrame:
    if baseline_relative.empty:
        return pd.DataFrame()

    workable = baseline_relative[baseline_relative["method"] != "no_compression"].copy()
    rows: list[dict[str, Any]] = []
    for (category, context_length), subset in workable.groupby(["main_category", "context_length"], dropna=False):
        if subset.empty:
            continue
        quality_winner = subset.sort_values(["quality_score", "memory_saved_mb"], ascending=[False, False]).iloc[0]
        memory_winner = subset.sort_values(["memory_saved_mb", "quality_score"], ascending=[False, False]).iloc[0]
        latency_winner = subset.sort_values(["avg_latency_seconds", "quality_score"], ascending=[True, False]).iloc[0]
        subset = subset.copy()
        subset["quality_norm"] = _normalize(subset["quality_score"], higher_is_better=True)
        subset["latency_norm"] = _normalize(subset["avg_latency_seconds"], higher_is_better=False)
        subset["memory_norm"] = _normalize(subset["peak_gpu_memory_mb"], higher_is_better=False)
        subset["balanced_score"] = 0.6 * subset["quality_norm"] + 0.25 * subset["latency_norm"] + 0.15 * subset["memory_norm"]
        balanced_winner = subset.sort_values("balanced_score", ascending=False).iloc[0]
        rows.append(
            {
                "main_category": category,
                "context_length": context_length,
                "baseline_quality_score": subset["baseline_quality_score"].iloc[0],
                "best_quality_method": quality_winner["method"],
                "best_quality_score": quality_winner["quality_score"],
                "best_memory_method": memory_winner["method"],
                "best_memory_saved_mb": memory_winner["memory_saved_mb"],
                "best_latency_method": latency_winner["method"],
                "best_latency_seconds": latency_winner["avg_latency_seconds"],
                "best_balanced_method": balanced_winner["method"],
                "best_balanced_score": balanced_winner["balanced_score"],
            }
        )
    return pd.DataFrame(rows).sort_values(["main_category", "context_length"])


def _pick_ground_truth(row: pd.Series) -> str:
    for key in ("answer", "answers", "ground_truth", "reference", "references"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return ""


def _detect_failure_mode(row: pd.Series) -> str:
    prediction = _normalize_text(row.get("predicted_answer"))
    ground_truth = _normalize_text(_pick_ground_truth(row))
    if not prediction:
        return "empty_output"
    if not ground_truth:
        return "missing_reference"
    if prediction == ground_truth:
        return "correct"

    tokens = [token for token in prediction.replace(",", " ").split() if token]
    if len(tokens) >= 4 and len(set(tokens)) <= max(1, len(tokens) // 2):
        return "repetition"
    if ground_truth and ground_truth not in prediction:
        overlap = set(tokens) & set(ground_truth.replace(",", " ").split())
        if overlap:
            return "partial_overlap"
        return "missed_reference"
    return "format_or_partial_mismatch"


def _build_failure_examples(combined: pd.DataFrame, limit_per_run: int = 3) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, run in combined.iterrows():
        predictions_path = Path(run["predictions_path"])
        if not predictions_path.exists():
            continue
        try:
            predictions = pd.read_csv(predictions_path)
        except Exception:
            continue
        if "predicted_answer" not in predictions.columns:
            continue
        predictions = predictions.copy()
        predictions["failure_mode"] = predictions.apply(_detect_failure_mode, axis=1)
        failures = predictions[predictions["failure_mode"] != "correct"].head(limit_per_run)
        for _, failure in failures.iterrows():
            rows.append(
                {
                    "scenario_name": run["scenario_name"],
                    "main_category": run["main_category"],
                    "context_length": run["context_length"],
                    "method": run["method"],
                    "budget": run["budget"],
                    "task": failure.get("task", ""),
                    "question": failure.get("question", ""),
                    "ground_truth": _pick_ground_truth(failure),
                    "prediction": str(failure.get("predicted_answer", "")),
                    "failure_mode": failure["failure_mode"],
                }
            )
    return pd.DataFrame(rows)


def _build_failure_mode_summary(failure_examples: pd.DataFrame) -> pd.DataFrame:
    if failure_examples.empty:
        return pd.DataFrame()
    return (
        failure_examples.groupby(
            ["main_category", "context_length", "method", "failure_mode"],
            dropna=False,
        )
        .size()
        .reset_index(name="example_count")
        .sort_values(["main_category", "context_length", "method", "example_count"], ascending=[True, True, True, False])
    )


def _set_mpl_dir(output_dir: Path) -> None:
    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _plot_quality_by_category(combined: pd.DataFrame, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if combined.empty:
        return None
    pivot = (
        combined.pivot_table(
            index="main_category",
            columns="method",
            values="quality_score",
            aggfunc="mean",
        )
        .reindex([label for label in CATEGORY_DISPLAY.values() if label in combined["main_category"].unique()])
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Average Quality by Task Category")
    ax.set_ylabel("Average string_match")
    ax.set_xlabel("Main Task Category")
    ax.legend(title="KV Cache Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = output_dir / "avg_quality_by_category.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_quality_drop(baseline_relative: pd.DataFrame, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if baseline_relative.empty:
        return None
    compressed = baseline_relative[baseline_relative["method"] != "no_compression"]
    pivot = compressed.pivot_table(
        index="main_category",
        columns="method",
        values="quality_drop_pct",
        aggfunc="mean",
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Percentage Quality Drop from Baseline (no_compression) per Main Task Category")
    ax.set_ylabel("Percentage Drop in Quality (string_match)")
    ax.set_xlabel("Main Task Category")
    ax.legend(title="KV Cache Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = output_dir / "quality_drop_from_baseline.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_quality_vs_metric(frame: pd.DataFrame, x_column: str, title: str, xlabel: str, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if frame.empty or x_column not in frame.columns:
        return None
    plotted = frame.dropna(subset=[x_column, "quality_score"]).copy()
    if plotted.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, subset in plotted.groupby("method"):
        ax.scatter(subset[x_column], subset["quality_score"], label=method, s=80, alpha=0.8)
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['main_category']} {row['context_length']}",
                (row[x_column], row["quality_score"]),
                fontsize=8,
                alpha=0.7,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average string_match")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_context_length(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return None
    compressed = frame[frame["method"] != "no_compression"].copy()
    pivot = compressed.pivot_table(
        index=["main_category", "method"],
        columns="context_length",
        values="quality_score",
        aggfunc="mean",
    )
    if pivot.empty or not set(pivot.columns).intersection({"4096", "8192"}):
        return None
    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = range(len(pivot))
    values_4k = pivot.get("4096", pd.Series(index=pivot.index, dtype=float))
    values_8k = pivot.get("8192", pd.Series(index=pivot.index, dtype=float))
    ax.plot(list(x_positions), values_4k.fillna(float("nan")), marker="o", label="4096")
    ax.plot(list(x_positions), values_8k.fillna(float("nan")), marker="o", label="8192")
    labels = [f"{category}\n{method}" for category, method in pivot.index]
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Quality by Method Across Context Lengths")
    ax.set_ylabel("Average string_match")
    ax.set_xlabel("Task Category / Method")
    ax.legend(title="Context Length")
    fig.tight_layout()
    output_path = output_dir / "quality_by_context_length.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_niah_heatmap(tasks: pd.DataFrame, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    if tasks.empty:
        return None
    niah = tasks[tasks["task_name"].astype(str).str.startswith("niah")].copy()
    if niah.empty:
        return None
    pivot = niah.pivot_table(index="method", columns="task_name", values="score", aggfunc="mean")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("NIAH Subtask Performance Heatmap")
    fig.colorbar(image, ax=ax, label="string_match")
    fig.tight_layout()
    output_path = output_dir / "niah_subtask_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_budget_sensitivity(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return None
    subset = frame[
        frame["main_category"].isin(["Needle In A Haystack", "Question Answering"])
        & frame["budget"].isin([0.25, 0.5, 0.75, 1.0])
    ].copy()
    subset = subset.sort_values(["main_category", "context_length", "method", "budget"])
    if subset.empty or subset["budget"].nunique() <= 2:
        return None
    categories = list(subset["main_category"].unique())
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 4.5 * len(categories)), squeeze=False)
    for axis, category in zip(axes.flatten(), categories):
        category_rows = subset[subset["main_category"] == category]
        for (method, context_length), group in category_rows.groupby(["method", "context_length"]):
            axis.plot(group["budget"], group["quality_score"], marker="o", label=f"{method} ({context_length})")
        axis.set_title(f"Budget Sensitivity: {category}")
        axis.set_xlabel("Budget")
        axis.set_ylabel("Average string_match")
        axis.grid(True, alpha=0.3)
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path = output_dir / "budget_sensitivity.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_benchmark_report(summary_dir: str | Path, output_dir: str | Path) -> BenchmarkReportArtifacts:
    summary_root = Path(summary_dir)
    output_root = _ensure_dir(output_dir)
    _set_mpl_dir(output_root)

    combined, tasks = _build_combined_summary(summary_root)
    baseline_relative = _compute_baseline_relative(combined)
    recommendations = _build_recommendations(baseline_relative)
    failure_examples = _build_failure_examples(combined)
    failure_mode_summary = _build_failure_mode_summary(failure_examples)

    combined_csv = output_root / "combined_summary.csv"
    tasks_csv = output_root / "task_metrics.csv"
    baseline_csv = output_root / "baseline_relative.csv"
    recommendations_csv = output_root / "recommendations.csv"
    failure_examples_csv = output_root / "failure_examples.csv"
    failure_mode_summary_csv = output_root / "failure_mode_summary.csv"

    combined.to_csv(combined_csv, index=False)
    tasks.to_csv(tasks_csv, index=False)
    baseline_relative.to_csv(baseline_csv, index=False)
    recommendations.to_csv(recommendations_csv, index=False)
    failure_examples.to_csv(failure_examples_csv, index=False)
    failure_mode_summary.to_csv(failure_mode_summary_csv, index=False)

    plot_paths: list[Path] = []
    for candidate in [
        _plot_quality_by_category(combined, output_root),
        _plot_quality_drop(baseline_relative, output_root),
        _plot_context_length(combined, output_root),
        _plot_quality_vs_metric(
            combined,
            "peak_gpu_memory_mb",
            "Quality vs Peak GPU Memory",
            "Peak GPU Memory (MB)",
            output_root / "quality_vs_memory.png",
        ),
        _plot_quality_vs_metric(
            combined,
            "avg_latency_seconds",
            "Quality vs Average Latency",
            "Average Latency (seconds)",
            output_root / "quality_vs_latency.png",
        ),
        _plot_niah_heatmap(tasks, output_root),
        _plot_budget_sensitivity(combined, output_root),
    ]:
        if candidate is not None:
            plot_paths.append(candidate)

    return BenchmarkReportArtifacts(
        combined_summary_csv=combined_csv,
        task_metrics_csv=tasks_csv,
        baseline_relative_csv=baseline_csv,
        recommendations_csv=recommendations_csv,
        failure_examples_csv=failure_examples_csv,
        failure_mode_summary_csv=failure_mode_summary_csv,
        output_dir=output_root,
        plot_paths=plot_paths,
    )
