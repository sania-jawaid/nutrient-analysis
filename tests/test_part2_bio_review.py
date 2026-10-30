import pytest

from part2.bio_review import (
    evaluate_fields, reference_fields, summarize_field_evaluations,
    prediction_diagnostics, transcription_differences,
)


def field(value, unit="g", basis="per_100g", nutrient="fat", qualifier=None):
    return dict(nutrient=nutrient, basis=basis, value=value, unit=unit, qualifier=qualifier)


def test_exact_value_accuracy_is_not_recall_and_duplicates_count():
    result = evaluate_fields([field(1), field(2)], [field(1), field(1)], reference_source="fixture")
    assert result["field_precision"] == 0.5
    assert result["field_recall"] == 0.5
    assert result["exact_value_accuracy"] == 0.5
    assert result["comparable_value_fields"] == 2
    missing = evaluate_fields([field(1), field(2)], [field(1)], reference_source="fixture")
    assert missing["exact_value_accuracy"] == 1
    assert missing["field_recall"] == 0.5
    assert missing["false_negative"] == 1


def test_units_basis_qualifiers_and_energy_declarations_are_distinct():
    result = evaluate_fields([field(0.1)], [field(100, "mg")], reference_source="fixture")
    assert result["true_positive"] == 1
    for prediction in (field(0.1, basis="per_serving"), field(0.1, qualifier="<")):
        assert evaluate_fields([field(0.1)], [prediction], reference_source="fixture")["true_positive"] == 0
    energy = evaluate_fields([field(4.184, "kj", nutrient="energy")],
                             [field(1, "kcal", nutrient="energy")], reference_source="fixture")
    assert energy["comparable_value_fields"] == 0
    assert energy["exact_value_accuracy"] is None


def test_micro_counts_and_empty_reference_do_not_invent_accuracy():
    rows = [evaluate_fields([field(1), field(2)], [field(1)], reference_source="fixture"),
            evaluate_fields([field(3)], [field(4)], reference_source="fixture")]
    summary = summarize_field_evaluations(rows)
    assert summary["field_precision"] == 0.5
    assert summary["field_recall"] == pytest.approx(1/3)
    assert summary["exact_value_accuracy"] == 0.5
    assert summarize_field_evaluations([])["field_f1"] is None
    with pytest.raises(ValueError, match="finite"):
        evaluate_fields([field(float("nan"))], [], reference_source="fixture")


def test_manual_basis_adapter_does_not_change_annotation():
    original = {"fields": [field(2, basis="serving_90g"), field(3, basis="100ml")]}
    adapted = reference_fields(original)
    assert [row["basis"] for row in adapted] == ["per_serving", "per_100ml"]
    assert original["fields"][0]["basis"] == "serving_90g"


def test_diagnostics_use_actual_uncertainties_and_coverage():
    prediction = {"token_count": 2, "covered_token_count": 1, "coverage_mask": [True, False],
                  "inference_windows": 1, "structured_output": {"uncertain_fields": [
                      {"uncertainty_reasons": ["duplicate_field", "unit_missing_or_not_read"],
                       "raw_text": "1", "label": "FAT_100G", "start": 0, "end": 1}]}}
    result = prediction_diagnostics([{"barcode": "001"}], [prediction])
    assert result["coverage_by_image"][0]["uncovered_positions"] == [1]
    assert result["category_counts"] == {"duplicate_prediction": 1, "unit_or_numeric_parsing": 1}
    assert "not independently confirmed" in result["examples"][0]["evidence"]
    with pytest.raises(ValueError):
        prediction_diagnostics([], [prediction])


def test_manual_transcription_diff_reports_observed_text_not_bio():
    rows = transcription_differences("Salt 0,36 g", "Salt 036 g")
    assert rows
    assert any("036" in row["ocr"] for row in rows)
    assert all("not a verified spatial" in row["note"] for row in rows)


def test_saved_manual_review_requires_current_reference_hashes(tmp_path):
    import hashlib
    import json
    from part2.bio_review import load_manual_reference_review
    from part2.layoutlm_bio import INFERENCE_SCHEMA_VERSION, MODEL_REVISION

    assert load_manual_reference_review(tmp_path) is None
    source_paths = ["data/evaluation/gold_records.jsonl", "data/evaluation/gold_ocr.txt",
                    "data/manual_samples/241013128_nutrition.jpg"]
    hashes = {}
    for relative in source_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reference fixture")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = tmp_path / "reports/part2/bio_manual_reference_review.json"
    destination.parent.mkdir(parents=True)
    report = {"inference_schema_version": INFERENCE_SCHEMA_VERSION,
              "model_revision": MODEL_REVISION, "source_hashes": hashes}
    destination.write_text(json.dumps(report), encoding="utf-8")
    assert load_manual_reference_review(tmp_path) == report
    (tmp_path / source_paths[0]).write_bytes(b"changed reference")
    with pytest.raises(ValueError, match="changed"):
        load_manual_reference_review(tmp_path)
    report["source_hashes"] = {}
    destination.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_manual_reference_review(tmp_path)


def test_manual_review_refuses_low_memory_before_loading_references(tmp_path, monkeypatch):
    from part2.bio_review import run_manual_reference_review
    from part2.layoutlm_bio import InsufficientResourcesError

    monkeypatch.setattr("part2.layoutlm_bio.available_ram_bytes", lambda: 100)
    with pytest.raises(InsufficientResourcesError, match="0.75 GiB"):
        run_manual_reference_review(tmp_path, model=None, processor=None, id2label={})