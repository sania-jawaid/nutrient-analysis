"""Entity-level and normalized extraction metrics."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def entity_metrics(
    true_sequences: Sequence[Sequence[str]], predicted_sequences: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Compute exact entity-span scores for BIO sequences, excluding padded tokens upstream."""
    if len(true_sequences) != len(predicted_sequences):
        raise ValueError("True and predicted sequence counts must match.")
    return {
        "entity_precision": precision_score(true_sequences, predicted_sequences),
        "entity_recall": recall_score(true_sequences, predicted_sequences),
        "entity_f1": f1_score(true_sequences, predicted_sequences),
        "classification_report": classification_report(true_sequences, predicted_sequences, output_dict=True),
    }


def normalize_value(value: str, unit: str | None = None) -> tuple[float | None, str | None]:
    """Normalize a simple decimal amount and unit; define domain conversions before scoring."""
    cleaned = value.strip().replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    amount = float(match.group()) if match else None
    canonical_unit = unit.strip().lower() if unit and unit.strip() else None
    return amount, canonical_unit


def normalized_match(
    predicted_value: str, predicted_unit: str | None, gold_value: str, gold_unit: str | None,
    *, absolute_tolerance: float = 1e-6,
) -> bool:
    """Compare numeric amounts and normalized unit strings, without implicit conversions."""
    predicted_amount, predicted_unit_norm = normalize_value(predicted_value, predicted_unit)
    gold_amount, gold_unit_norm = normalize_value(gold_value, gold_unit)
    if predicted_amount is None or gold_amount is None or predicted_unit_norm != gold_unit_norm:
        return False
    return abs(predicted_amount - gold_amount) <= absolute_tolerance
