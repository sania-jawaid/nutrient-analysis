"""Reference-aware field scoring and real-output diagnostics, without model loads."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any
import math
import re

from part2.normalization import canonicalize, normalize_unit


def reference_fields(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt existing manual fields; never infer missing reference quantities."""
    fields = []
    for field in record.get("fields", []):
        raw_basis = str(field.get("basis", "")).casefold().replace(" ", "")
        basis = {"100g": "per_100g", "100ml": "per_100ml"}.get(raw_basis, raw_basis)
        if "serv" in basis or "portion" in basis:
            basis = "per_serving"
        fields.append({**field, "basis": basis})
    return fields


def _quantity(field: dict[str, Any]) -> tuple[tuple[str, str, str | None], float, Any]:
    unit = normalize_unit(field.get("unit"))
    value = float(field["value"])
    if not math.isfinite(value):
        raise ValueError("Reference and predicted numeric values must be finite.")
    # Keep kJ and kcal separate: rounded panel declarations are distinct fields.
    if unit in {"g", "mg", "ug"}:
        value, unit = canonicalize(value, unit)
    return (str(field["nutrient"]), str(field["basis"]), unit), value, field.get("qualifier")


def evaluate_fields(
    gold_fields: list[dict[str, Any]], predicted_fields: list[dict[str, Any]],
    *, reference_source: str,
) -> dict[str, Any]:
    """Exact tuple PRF plus value accuracy conditional on semantic-key matching.

    Duplicate occurrences are counted, not collapsed. Exact matches are paired
    first, then remaining occurrences in source order, never nearest-value matching.
    Missing fields reduce recall but are not silently included in value accuracy.
    """
    gold = [_quantity(field) for field in gold_fields]
    predicted = [_quantity(field) for field in predicted_fields]
    gold_counts, pred_counts = Counter(gold), Counter(predicted)
    tp = sum((gold_counts & pred_counts).values())
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    gold_by_key, pred_by_key = defaultdict(list), defaultdict(list)
    for key, value, qualifier in gold:
        gold_by_key[key].append((value, qualifier))
    for key, value, qualifier in predicted:
        pred_by_key[key].append((value, qualifier))
    comparisons = []
    value_matches = comparable = 0
    for key in sorted(set(gold_by_key) | set(pred_by_key), key=str):
        expected, actual = Counter(gold_by_key[key]), Counter(pred_by_key[key])
        common = expected & actual
        pair_count = min(sum(expected.values()), sum(actual.values()))
        comparable += pair_count
        value_matches += sum(common.values())
        pairs = [(value, value, "exact_match") for value in common.elements()]
        remaining_gold = list((expected - common).elements())
        remaining_pred = list((actual - common).elements())
        for index in range(max(len(remaining_gold), len(remaining_pred))):
            left = remaining_gold[index] if index < len(remaining_gold) else None
            right = remaining_pred[index] if index < len(remaining_pred) else None
            status = "missing_prediction" if right is None else "unexpected_prediction" if left is None else "value_or_qualifier_mismatch"
            pairs.append((left, right, status))
        for left, right, status in pairs:
            comparisons.append({
                "nutrient": key[0], "basis": key[1], "unit": key[2],
                "reference_value": left[0] if left is not None else None,
                "predicted_value": right[0] if right is not None else None,
                "reference_qualifier": left[1] if left is not None else None,
                "predicted_qualifier": right[1] if right is not None else None,
                "status": status,
            })
    return {
        "reference_source": reference_source,
        "true_positive": tp, "false_positive": len(predicted) - tp,
        "false_negative": len(gold) - tp,
        "predicted_fields": len(predicted), "gold_fields": len(gold),
        "field_precision": precision if gold or predicted else None,
        "field_recall": recall if gold else None,
        "field_f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if gold or predicted else None,
        "exact_value_matches": value_matches, "comparable_value_fields": comparable,
        "exact_value_accuracy": value_matches / comparable if comparable else None,
        "exact_value_accuracy_note": "Exact value AND qualifier on matching nutrient/basis/compatible-unit keys; no tolerance; unmatched fields are disclosed separately.",
        "comparisons": comparisons,
    }


def summarize_field_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro aggregate from explicit counts, not averages of per-image scores."""
    keys = ("true_positive", "false_positive", "false_negative", "predicted_fields", "gold_fields", "exact_value_matches", "comparable_value_fields", "gold_uncertain_spans", "predicted_uncertain_spans")
    totals = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    tp, predicted, gold = totals["true_positive"], totals["predicted_fields"], totals["gold_fields"]
    precision, recall = tp / predicted if predicted else 0, tp / gold if gold else 0
    return {
        **totals, "images": len(rows),
        "field_precision": precision if predicted or gold else None,
        "field_recall": recall if gold else None,
        "field_f1": (2 * precision * recall / (precision + recall) if precision + recall else 0) if predicted or gold else None,
        "exact_value_accuracy": totals["exact_value_matches"] / totals["comparable_value_fields"] if totals["comparable_value_fields"] else None,
    }


def prediction_diagnostics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Surface actual parser/coverage evidence; do not call suspicions OCR truth."""
    if len(records) != len(predictions):
        raise ValueError("Records and predictions must align.")
    examples, coverage = [], []
    for record, prediction in zip(records, predictions):
        barcode = record.get("barcode")
        for field in prediction.get("structured_output", {}).get("uncertain_fields", []):
            for reason in field["uncertainty_reasons"]:
                category = "duplicate_prediction" if reason.startswith("duplicate_field") else "unit_or_numeric_parsing"
                examples.append({"barcode": barcode, "category": category, "reason": reason,
                    "raw_text": field["raw_text"], "label": field["label"],
                    "start": field["start"], "end": field["end"],
                    "evidence": "predicted span parser diagnostic, not independently confirmed OCR error"})
        coverage.append({"barcode": barcode, "source_index": record.get("source_index"),
            "tokens": prediction["token_count"], "covered": prediction["covered_token_count"],
            "windows": prediction.get("inference_windows"),
            "uncovered_positions": [index for index, covered in enumerate(prediction["coverage_mask"]) if not covered]})
    return {"category_counts": dict(Counter(row["category"] for row in examples)),
            "reason_counts": dict(Counter(row["reason"] for row in examples)),
            "examples": examples, "coverage_by_image": coverage}


def transcription_differences(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    """Reproducible word diff on an existing manual transcription, not BIO alignment."""
    def words(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)

    expected, actual = words(reference), words(hypothesis)
    rows = []
    for operation, i, j, a, b in SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes():
        if operation != "equal":
            rows.append({"operation": operation, "reference": " ".join(expected[i:j]),
                "ocr": " ".join(actual[a:b]), "reference_start": i, "ocr_start": a,
                "note": "Text/order difference against manual reference; not a verified spatial word alignment."})
    return rows


def run_manual_reference_review(root, *, model, processor, id2label) -> dict[str, Any]:
    """Run the existing one-image manual fixture; it is not a new held-out set."""
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path
    from part2.evaluation import evaluate_verified_ocr
    from part2.entity_extraction import extract_nutrients
    from part2.layoutlm_bio import (
        available_ram_bytes, InsufficientResourcesError, predict_uploaded_image,
        INFERENCE_SCHEMA_VERSION, MODEL_ID, MODEL_REVISION,
    )

    root = Path(root)
    if available_ram_bytes() / 1024**3 < 0.75:
        raise InsufficientResourcesError("Manual reference review requires 0.75 GiB free with the model resident.")
    reference_path = root / "data/evaluation/gold_records.jsonl"
    text_path = root / "data/evaluation/gold_ocr.txt"
    manual = next(row for row in (json.loads(line) for line in reference_path.read_text(encoding="utf-8").splitlines() if line.strip()) if str(row.get("product_id")) == "241013128")
    image_path = root / manual["source_image"]
    reference_text = text_path.read_text(encoding="utf-8")
    result = predict_uploaded_image(
        image_path, model=model, processor=processor, id2label=id2label,
        cache_dir=root / "data/part2_cache/layoutlm_bio",
        easyocr_model_dir=root / "models/easyocr", ocr_cache_dir=root / "data/part2_cache/ocr",
    )
    paths = [reference_path, text_path, image_path]
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "reference_product_id": manual["product_id"], "reference_images": 1,
        "reference_source": "existing manually reviewed gold files from part2_ocr_validation; not new independent annotation or a population benchmark",
        "source_hashes": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "ocr_error": result.get("ocr_error"),
        "ocr_cache_hit": result.get("ocr_cache_hit"),
        "reference_text": reference_text,
    }
    if result.get("ocr_error"):
        report.update(status="ocr_failed", field_metrics=None, ocr_metrics=None)
    else:
        ocr_text = " ".join(item.text for item in result["ocr_result"].detections)
        gold_fields = reference_fields(manual)
        report.update({
            "status": "completed_single_manual_reference",
            "ocr_text": ocr_text,
            "ocr_metrics": evaluate_verified_ocr([{"verified": True, "reference_text": reference_text, "hypothesis_text": ocr_text}]),
            "ocr_difference_examples": transcription_differences(reference_text, ocr_text),
            "ocr_seconds_recorded": result["ocr_elapsed_seconds"],
            "model_cache_hit": result["prediction"]["cache_hit"],
            "onnx_seconds_recorded": result["prediction"]["inference_seconds"],
            "timing_note": "Recorded stage times may originate from prediction/OCR caches; cache flags indicate reuse. No new latency is claimed for cache hits.",
            "field_metrics": evaluate_fields(gold_fields, result["structured_output"]["parsed_fields"], reference_source=report["reference_source"]),
            "parse_failure_count": len(result["structured_output"]["uncertain_fields"]),
            "structured_output": result["structured_output"],
            "token_count": len(result["tokens"]),
            "covered_tokens": result["prediction"]["covered_token_count"],
            "bio_metrics": None,
            "bio_metrics_note": "Manual reference is record-level; EasyOCR tokens have no gold BIO alignment.",
        })
        # Reuse the original rule extractor on the identical OCR detections.
        from part2.config import PreprocessingConfig
        from part2.preprocessing import preprocess_image
        processed = preprocess_image(result["image"], PreprocessingConfig(max_side=2000, contrast=1.15, grayscale=False, resize=True))
        rule = extract_nutrients(result["ocr_result"].detections, barcode=manual["product_id"], image_path=str(image_path), image_width=processed.width).to_dict()
        rule_fields = [{**value, "nutrient": "energy" if nutrient in {"energy_kj", "energy_kcal"} else nutrient, "basis": basis}
                       for basis, nutrients in rule.get("nutrients", {}).items()
                       for nutrient, values in nutrients.items()
                       for value in (values if isinstance(values, list) else [values])]
        report["rule_field_metrics_same_ocr"] = evaluate_fields(gold_fields, rule_fields, reference_source=report["reference_source"])
    destination = root / "reports/part2/bio_manual_reference_review.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    return report


def load_manual_reference_review(root) -> dict[str, Any] | None:
    """Return saved evidence only when its referenced files/model still match."""
    import hashlib
    import json
    from pathlib import Path
    from part2.layoutlm_bio import INFERENCE_SCHEMA_VERSION, MODEL_REVISION

    root = Path(root)
    destination = root / "reports/part2/bio_manual_reference_review.json"
    if not destination.is_file():
        return None
    report = json.loads(destination.read_text(encoding="utf-8"))
    if report.get("inference_schema_version") != INFERENCE_SCHEMA_VERSION or report.get("model_revision") != MODEL_REVISION:
        raise ValueError("Saved manual-reference review uses a different model/schema; rerun that review.")
    required = {"data/evaluation/gold_records.jsonl", "data/evaluation/gold_ocr.txt", "data/manual_samples/241013128_nutrition.jpg"}
    if set(report.get("source_hashes", {})) != required:
        raise ValueError("Saved manual review lacks complete reference provenance.")
    for relative, expected in report["source_hashes"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Manual reference/image changed after the saved evaluation; rerun that review.")
    return report