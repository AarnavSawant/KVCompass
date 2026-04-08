from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_COLUMNS = [
    "timestamp",
    "run_name",
    "model_name",
    "scenario",
    "method",
    "budget",
    "context_length",
    "repeat_index",
    "latency_seconds",
    "throughput_tokens_per_second",
    "peak_gpu_memory_mb",
    "input_tokens",
    "output_tokens",
    "compression_ratio",
    "cache_type",
    "quality_score",
    "quality_label",
    "generated_text",
    "reference_answer",
    "status",
    "error_message",
]


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def write_rows(rows: Iterable[dict], output_path: str | Path) -> Path:
    output_path = ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def read_results(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    numeric_cols = [
        "budget",
        "context_length",
        "repeat_index",
        "latency_seconds",
        "throughput_tokens_per_second",
        "peak_gpu_memory_mb",
        "input_tokens",
        "output_tokens",
        "compression_ratio",
        "quality_score",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame
