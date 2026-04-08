from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_parent, read_results


def aggregate_results(raw_results_path: str | Path, output_path: str | Path) -> Path:
    frame = read_results(raw_results_path)
    if frame.empty:
        output = ensure_parent(output_path)
        pd.DataFrame().to_csv(output, index=False)
        return output

    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        output = ensure_parent(output_path)
        pd.DataFrame().to_csv(output, index=False)
        return output

    summary = (
        ok.groupby(["scenario", "method", "budget", "context_length", "cache_type"], dropna=False)
        .agg(
            runs=("status", "count"),
            latency_seconds_mean=("latency_seconds", "mean"),
            latency_seconds_std=("latency_seconds", "std"),
            throughput_tokens_per_second_mean=("throughput_tokens_per_second", "mean"),
            peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"),
            quality_score_mean=("quality_score", "mean"),
            input_tokens_mean=("input_tokens", "mean"),
            output_tokens_mean=("output_tokens", "mean"),
            compression_ratio_mean=("compression_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "context_length", "budget", "method"])
    )

    output = ensure_parent(output_path)
    summary.to_csv(output, index=False)
    return output
