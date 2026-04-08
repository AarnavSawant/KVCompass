from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_parent


def _normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    series = series.astype(float)
    if series.nunique(dropna=True) <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    min_value = series.min()
    max_value = series.max()
    scaled = (series - min_value) / (max_value - min_value)
    return scaled if higher_is_better else 1.0 - scaled


def build_recommendations(summary_path: str | Path, output_path: str | Path) -> Path:
    frame = pd.read_csv(summary_path)
    if frame.empty:
        output = ensure_parent(output_path)
        pd.DataFrame().to_csv(output, index=False)
        return output

    recommendations: list[dict[str, str]] = []
    for scenario, subset in frame.groupby("scenario"):
        workable = subset.dropna(subset=["latency_seconds_mean", "quality_score_mean"]).copy()
        if workable.empty:
            continue

        memory_candidates = workable.dropna(subset=["peak_gpu_memory_mb_mean"])
        best_for_memory = (
            memory_candidates.sort_values(["peak_gpu_memory_mb_mean", "quality_score_mean"], ascending=[True, False])
            .iloc[0]["method"]
            if not memory_candidates.empty
            else ""
        )
        best_for_latency = workable.sort_values(
            ["latency_seconds_mean", "quality_score_mean"], ascending=[True, False]
        ).iloc[0]["method"]

        workable["quality_norm"] = _normalize(workable["quality_score_mean"], higher_is_better=True)
        workable["latency_norm"] = _normalize(workable["latency_seconds_mean"], higher_is_better=False)
        if "peak_gpu_memory_mb_mean" in workable.columns and workable["peak_gpu_memory_mb_mean"].notna().any():
            workable["memory_norm"] = _normalize(workable["peak_gpu_memory_mb_mean"], higher_is_better=False)
        else:
            workable["memory_norm"] = 1.0
        workable["balanced_score"] = (
            0.5 * workable["quality_norm"] + 0.3 * workable["latency_norm"] + 0.2 * workable["memory_norm"]
        )
        best_balanced = workable.sort_values("balanced_score", ascending=False).iloc[0]["method"]

        recommendations.append(
            {
                "scenario": scenario,
                "best_for_memory": best_for_memory,
                "best_for_latency": best_for_latency,
                "best_balanced": best_balanced,
            }
        )

    output = ensure_parent(output_path)
    pd.DataFrame(recommendations).to_csv(output, index=False)
    return output
