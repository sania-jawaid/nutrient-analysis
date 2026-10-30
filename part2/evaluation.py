"""OCR, structured nutrient, and aligned BIO evaluation helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median, mean
from typing import Any, Iterable

from part2.normalization import canonicalize, normalize_unit


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    return edit_distance(list(reference), list(hypothesis)) / max(1, len(reference))


def wer(reference: str, hypothesis: str) -> float:
    return edit_distance(reference.split(), hypothesis.split()) / max(1, len(reference.split()))


def evaluate_verified_ocr(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    verified = [row for row in records if row.get("verified") is True and isinstance(row.get("reference_text"), str)]
    if not verified:
        return {"status": "not_established", "reason": "No manually verified OCR transcription references provided.", "support": 0, "cer": None, "wer": None}
    return {
        "status": "measured_on_manual_references",
        "support": len(verified),
        "cer": sum(cer(row["reference_text"], row.get("hypothesis_text", "")) for row in verified) / len(verified),
        "wer": sum(wer(row["reference_text"], row.get("hypothesis_text", "")) for row in verified) / len(verified),
    }


def _field_map(record: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for basis, nutrients in record.get("nutrients", {}).items():
        for nutrient, values in nutrients.items():
            result[(basis, nutrient)] = list(values if isinstance(values, list) else [values])
    return result


def structured_extraction_metrics(
    gold_records: list[dict[str, Any]], predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute field PRF and exact/value/unit/basis metrics over aligned records."""
    if len(gold_records) != len(predictions):
        raise ValueError("Gold and prediction record counts must match.")
    tp = fp = fn = exact_match = numeric_count = 0
    errors: list[float] = []
    unit_correct = unit_total = record_exact = 0
    for gold, pred in zip(gold_records, predictions):
        gold_map, pred_map = _field_map(gold), _field_map(pred)
        gold_keys, pred_keys = set(gold_map), set(pred_map)
        tp += len(gold_keys & pred_keys)
        fp += len(pred_keys - gold_keys)
        fn += len(gold_keys - pred_keys)
        record_ok = gold_keys == pred_keys
        for key in gold_keys & pred_keys:
            gold_items, pred_items = gold_map[key], pred_map[key]
            # Occurrence order is retained; multiple ambiguous values are not collapsed.
            for index in range(max(len(gold_items), len(pred_items))):
                if index >= len(gold_items) or index >= len(pred_items):
                    record_ok = False
                    continue
                expected, actual = gold_items[index], pred_items[index]
                unit_total += 1
                unit_correct += int(normalize_unit(expected.get("unit")) == normalize_unit(actual.get("unit")))
                value_ok = expected.get("value") == actual.get("value") and normalize_unit(expected.get("unit")) == normalize_unit(actual.get("unit"))
                if value_ok:
                    exact_match += 1
                else:
                    record_ok = False
                try:
                    expected_value, expected_unit = canonicalize(float(expected["value"]), expected.get("unit"))
                    actual_value, actual_unit = canonicalize(float(actual["value"]), actual.get("unit"))
                    if expected_unit == actual_unit:
                        errors.append(abs(expected_value - actual_value))
                        numeric_count += 1
                except (KeyError, TypeError, ValueError):
                    pass
        record_exact += int(record_ok)
    # Compare bases separately from basis-qualified field keys, so a nutrient
    # predicted under the wrong basis contributes an incorrect basis.
    basis_correct = basis_total = 0
    for gold, pred in zip(gold_records, predictions):
        gold_map, pred_map = _field_map(gold), _field_map(pred)
        gold_by_nutrient: dict[str, list[str]] = {}
        pred_by_nutrient: dict[str, list[str]] = {}
        for basis, nutrient in sorted(gold_map):
            gold_by_nutrient.setdefault(nutrient, []).extend([basis] * len(gold_map[(basis, nutrient)]))
        for basis, nutrient in sorted(pred_map):
            pred_by_nutrient.setdefault(nutrient, []).extend([basis] * len(pred_map[(basis, nutrient)]))
        for nutrient, gold_bases in gold_by_nutrient.items():
            predicted_bases = pred_by_nutrient.get(nutrient, [])
            for index, expected_basis in enumerate(gold_bases):
                basis_total += 1
                basis_correct += int(index < len(predicted_bases) and expected_basis == predicted_bases[index])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    latencies = [
        float(record["processing_time_seconds"])
        for record in predictions
        if isinstance(record.get("processing_time_seconds"), (int, float))
        and record["processing_time_seconds"] >= 0
    ]
    return {
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "field_tp": tp,
        "field_fp": fp,
        "field_fn": fn,
        "exact_value_unit_match": exact_match / max(1, tp),
        "exact_value_unit_support": tp,
        "basis_accuracy": basis_correct / basis_total if basis_total else 0.0,
        "basis_support": basis_total,
        "unit_accuracy": unit_correct / unit_total if unit_total else 0.0,
        "unit_support": unit_total,
        "numeric_mae_compatible_units": sum(errors) / len(errors) if errors else None,
        "numeric_mae_support": numeric_count,
        "record_exact_match": record_exact / len(gold_records) if gold_records else 0.0,
        "record_support": len(gold_records),
        "processing_time_mean_seconds_per_image": mean(latencies) if latencies else None,
        "processing_time_median_seconds_per_image": median(latencies) if latencies else None,
        "processing_time_support": len(latencies),
    }


def _bio_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    current: str | None = None
    start = 0
    for index, tag in enumerate(tags + ["O"]):
        if tag == "O" or "-" not in tag:
            if current is not None:
                spans.add((current, start, index))
                current = None
            continue
        prefix, entity = tag.split("-", 1)
        if prefix == "B" or current != entity:
            if current is not None:
                spans.add((current, start, index))
            current, start = entity, index
    return spans


def aligned_bio_entity_metrics(
    gold_tag_sequences: list[list[str]],
    predicted_tag_sequences: list[list[str]],
    *,
    token_alignment_verified: bool,
) -> dict[str, Any]:
    """Exact BIO span PRF, only when predictions are aligned to the same tokens."""
    if not token_alignment_verified:
        raise ValueError("BIO metrics require an explicitly verified OCR-to-gold token alignment.")
    if len(gold_tag_sequences) != len(predicted_tag_sequences):
        raise ValueError("Gold and predicted BIO sequence counts must match.")
    gold_spans: list[tuple[str, int, int, int]] = []
    pred_spans: list[tuple[str, int, int, int]] = []
    for record_index, (gold, predicted) in enumerate(zip(gold_tag_sequences, predicted_tag_sequences)):
        if len(gold) != len(predicted):
            raise ValueError("Aligned gold and prediction BIO sequences must have equal token lengths.")
        gold_spans.extend((kind, start, end, record_index) for kind, start, end in _bio_spans(gold))
        pred_spans.extend((kind, start, end, record_index) for kind, start, end in _bio_spans(predicted))
    common = Counter(gold_spans) & Counter(pred_spans)
    tp = sum(common.values())
    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall = tp / len(gold_spans) if gold_spans else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "tp": tp, "predicted": len(pred_spans), "gold": len(gold_spans)}


def main() -> int:
    # These are intentionally illustrative fixtures for verifying metric code;
    # they are not model outputs and must never be reported as measured results.
    gold = [{"nutrition_basis": "per_100g", "nutrients": {"per_100g": {"fat": [{"value": 10.0, "unit": "g"}]}}}]
    illustrative_prediction = [{"nutrition_basis": "per_100g", "nutrients": {"per_100g": {"fat": [{"value": 9.8, "unit": "g"}]}}}]
    print(json.dumps({"label": "illustrative fixture only; not model performance", "metrics": structured_extraction_metrics(gold, illustrative_prediction), "ocr_metrics": evaluate_verified_ocr([])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())