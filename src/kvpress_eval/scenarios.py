from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Example:
    context: str
    questions: list[str]
    reference_answer: str
    quality_label: str


def _repeat_to_length(seed: str, target_length: int) -> str:
    if len(seed) >= target_length:
        return seed[:target_length]
    repeats = (target_length // len(seed)) + 2
    return (seed * repeats)[:target_length]


def _make_noise_block(prefix: str, count: int) -> str:
    lines = [
        f"{prefix} note {idx}: archival logs mention shipping manifests, analyst comments, and routine updates."
        for idx in range(1, count + 1)
    ]
    return "\n".join(lines)


def build_example(scenario_name: str, context_length: int) -> Example:
    """
    Build a small synthetic-but-structured workload example.

    TODO:
    Replace these examples with benchmark or dataset-backed loaders once the
    target tasks and scoring methodology are finalized.
    """
    if scenario_name == "retrieval_long":
        answer = "silver pine"
        core = (
            "Project Atlas archive\n"
            f"Buried fact: the access code is {answer}.\n"
            "Use the buried fact exactly as written when answering.\n"
        )
        context = _repeat_to_length(_make_noise_block("retrieval", 250) + "\n" + core, context_length)
        return Example(
            context=context,
            questions=["What is the access code mentioned in the archive?"],
            reference_answer=answer,
            quality_label="contains_reference",
        )

    if scenario_name == "reasoning_long":
        answer = "merchant district"
        core = (
            "City planning notes:\n"
            "The museum opens near the river.\n"
            "The tram line connects the river to the merchant district.\n"
            "The research library is adjacent to the museum.\n"
            "Question target: identify the district reached from the museum using the tram line.\n"
        )
        context = _repeat_to_length(_make_noise_block("reasoning", 250) + "\n" + core, context_length)
        return Example(
            context=context,
            questions=["From the museum, which district can be reached using the tram line?"],
            reference_answer=answer,
            quality_label="contains_reference",
        )

    if scenario_name == "context_sweep":
        answer = "north observatory"
        core = (
            "Operations digest:\n"
            f"The calibration checkpoint is the {answer}.\n"
            "Respond with the checkpoint name only.\n"
        )
        context = _repeat_to_length(_make_noise_block("context", 150) + "\n" + core, context_length)
        return Example(
            context=context,
            questions=["What is the calibration checkpoint?"],
            reference_answer=answer,
            quality_label="contains_reference",
        )

    if scenario_name == "prefix_serving":
        answer = "17"
        core = (
            "Shared prefix service log:\n"
            "User alpha quota: 11.\n"
            "User beta quota: 17.\n"
            "User gamma quota: 23.\n"
            "Answer using only the requested number.\n"
        )
        context = _repeat_to_length(_make_noise_block("prefix", 220) + "\n" + core, context_length)
        return Example(
            context=context,
            questions=[
                "What is the quota for user beta?",
                "Repeat the quota for user beta.",
                "State beta's quota as a number.",
            ],
            reference_answer=answer,
            quality_label="contains_reference",
        )

    raise ValueError(f"Unknown scenario: {scenario_name}")


def select_questions(example: Example, repeats: int) -> Sequence[str]:
    if repeats <= 1:
        return example.questions[:1]
    return example.questions[:repeats]
