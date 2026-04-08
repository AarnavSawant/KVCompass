from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityResult:
    score: float
    label: str


def score_prediction(prediction: str, reference: str, label: str = "contains_reference") -> QualityResult:
    pred = (prediction or "").strip().lower()
    ref = (reference or "").strip().lower()

    if not ref:
        return QualityResult(score=0.0, label="missing_reference")

    if label == "exact_match":
        return QualityResult(score=1.0 if pred == ref else 0.0, label=label)

    if ref in pred:
        return QualityResult(score=1.0, label=label)

    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    overlap = len(pred_tokens & ref_tokens) / max(len(ref_tokens), 1)
    return QualityResult(score=overlap, label=label)
