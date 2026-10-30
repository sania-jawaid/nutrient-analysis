"""Metrics for OCR quality and structured nutritional information extraction."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from nutrition_extraction.extraction import canonicalize_measurement
from nutrition_extraction.schema import NutritionRecord


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Compute Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_value in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_value in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (ref_value != hyp_value),
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return normalized character edit distance; empty reference uses explicit convention."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return normalized word edit distance after whitespace tokenization."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def entity_prf(gold: Sequence[tuple[str, int, int]], predicted: Sequence[tuple[str, int, int]]) -> dict[str, float | int]:
    """Compute exact typed-span micro precision, recall, and F1 from (type,start,end) tuples."""
    gold_counts, pred_counts = Counter(gold), Counter(predicted)
    true_positive = sum((gold_counts & pred_counts).values())
    precision = true_positive / sum(pred_counts.values()) if pred_counts else 0.0
    recall = true_positive / sum(gold_counts.values()) if gold_counts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": true_positive, "predicted": sum(pred_counts.values()), "gold": sum(gold_counts.values()), "precision": precision, "recall": recall, "f1": f1}


def _field_map(record: NutritionRecord) -> dict[str, dict[str, Any]]:
    """Use nutrient, basis, and occurrence index to pair same fields across different units."""
    mapped: dict[str, dict[str, Any]] = {}
    occurrence_counts: dict[tuple[str, str], int] = {}
    for field in record.fields:
        base_key = (field.nutrient, field.basis or "")
        occurrence = occurrence_counts.get(base_key, 0)
        occurrence_counts[base_key] = occurrence + 1
        key = "|".join((*base_key, str(occurrence)))
        canonical_value, canonical_unit = canonicalize_measurement(float(field.value), field.unit) if field.value is not None else (None, None)
        mapped[key] = {
            "value": field.value,
            "unit": field.unit,
            "basis": field.basis,
            "normalized_value": field.normalized_value if field.normalized_value is not None else canonical_value,
            "normalized_unit": field.normalized_unit if field.normalized_unit is not None else canonical_unit,
        }
    return mapped


def structured_metrics(gold_records: Sequence[NutritionRecord], predicted_records: Sequence[NutritionRecord]) -> dict[str, Any]:
    """Calculate field/record exactness, numeric error, unit/basis accuracy, and support."""
    if len(gold_records) != len(predicted_records):
        raise ValueError("Gold and predicted record counts must be equal and ordered consistently.")
    field_correct = field_total = record_correct = 0
    numeric_errors: list[float] = []
    unit_correct = unit_total = basis_correct = basis_total = 0
    for gold, predicted in zip(gold_records, predicted_records):
        gold_fields, pred_fields = _field_map(gold), _field_map(predicted)
        record_matches = gold_fields.keys() == pred_fields.keys()
        for key, gold_field in gold_fields.items():
            field_total += 1
            pred_field = pred_fields.get(key)
            exact = (
                pred_field is not None
                and gold_field["normalized_value"] == pred_field["normalized_value"]
                and gold_field["normalized_unit"] == pred_field["normalized_unit"]
                and gold_field["basis"] == pred_field["basis"]
            )
            field_correct += int(exact)
            if pred_field is not None:
                if gold_field["normalized_value"] is not None and pred_field["normalized_value"] is not None:
                    numeric_errors.append(abs(float(gold_field["normalized_value"]) - float(pred_field["normalized_value"])))
                unit_total += 1
                unit_correct += int(gold_field["unit"] == pred_field["unit"])
                basis_total += 1
                basis_correct += int(gold_field["basis"] == pred_field["basis"])
            record_matches = record_matches and exact
        record_correct += int(record_matches)
    return {
        "record_count": len(gold_records),
        "field_support": field_total,
        "field_exact_match_accuracy": field_correct / field_total if field_total else 0.0,
        "record_exact_match_accuracy": record_correct / len(gold_records) if gold_records else 0.0,
        "numeric_mae": sum(numeric_errors) / len(numeric_errors) if numeric_errors else None,
        "numeric_mae_support": len(numeric_errors),
        "unit_accuracy": unit_correct / unit_total if unit_total else 0.0,
        "unit_support": unit_total,
        "basis_accuracy": basis_correct / basis_total if basis_total else 0.0,
        "basis_support": basis_total,
    }


def measure_latency(callable_: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Measure wall-clock latency in seconds for one end-to-end invocation."""
    start = time.perf_counter()
    result = callable_(*args, **kwargs)
    return result, time.perf_counter() - start
