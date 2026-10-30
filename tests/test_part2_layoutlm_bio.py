import numpy as np
from PIL import Image
import pytest

from part2.dataset import BIO_LABELS
from part2.layoutlm_bio import (
    _entity_spans,
    draw_prediction_boxes,
    evaluate_sequences,
    load_pretrained_model,
    resource_snapshot,
    rule_baseline_coarse_bio,
    predict_record,
    to_coarse_bio,
    InsufficientResourcesError,
    decode_bio_entities,
)


def test_fine_bio_labels_project_to_shared_coarse_types():
    assert to_coarse_bio(["O", "B-FAT_100G", "I-FAT_100G", "B-SERVING_SIZE"]) == [
        "O", "B-VALUE", "I-VALUE", "B-BASIS",
    ]


def test_entity_spans_close_on_o_and_new_b_label():
    assert _entity_spans(["B-FAT_100G", "I-FAT_100G", "O", "B-SUGARS_SERVING"]) == [
        {"label": "FAT_100G", "start": 0, "end": 2},
        {"label": "SUGARS_SERVING", "start": 3, "end": 4},
    ]


def test_exact_span_metrics_require_aligned_input():
    gold = [["O", "B-VALUE", "I-VALUE", "O"]]
    predicted = [["O", "B-VALUE", "I-VALUE", "O"]]
    metrics = evaluate_sequences(gold, predicted, aligned=True)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_existing_rule_baseline_returns_same_length_coarse_bio():
    tokens = ["Fat", "18", "g"]
    boxes = [[10, 10, 100, 50], [120, 10, 160, 50], [165, 10, 180, 50]]
    labels = rule_baseline_coarse_bio(tokens, boxes, (400, 200))
    assert len(labels) == len(tokens)
    assert labels[0] == "O"
    assert labels[1:] == ["B-VALUE", "I-VALUE"]


def test_prediction_preview_does_not_modify_original_image():
    image = Image.new("RGB", (100, 100), "white")
    preview = draw_prediction_boxes(image, [[100, 100, 500, 500]], ["B-FAT_100G"])
    assert preview.size == image.size
    assert image.getpixel((10, 10)) == (255, 255, 255)


def test_resource_snapshot_reports_cached_weights_false_for_empty_dir(tmp_path):
    info = resource_snapshot(tmp_path / "model")
    assert info["weights_cached"] is False
    assert info["checkpoint_size_gb"] > 1.0


def test_onnx_word_alignment_decoding_and_prediction_cache(tmp_path):
    class FakeEncoding(dict):
        def word_ids(self, batch_index=0):
            return [None, 0, 1, 2, None]

    class FakeProcessor:
        def __call__(self, **kwargs):
            return FakeEncoding({
                "input_ids": np.zeros((1, 5), dtype=np.int64),
                "attention_mask": np.ones((1, 5), dtype=np.int64),
                "bbox": np.zeros((1, 5, 4), dtype=np.int64),
                "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32),
            })

    class FakeInput:
        def __init__(self, name, input_type):
            self.name = name
            self.type = input_type

    class FakeSession:
        def get_inputs(self):
            return [
                FakeInput("input_ids", "tensor(int64)"),
                FakeInput("attention_mask", "tensor(int64)"),
                FakeInput("bbox", "tensor(int64)"),
                FakeInput("pixel_values", "tensor(float)"),
            ]

        def run(self, _outputs, feed):
            assert feed["pixel_values"].shape == (1, 3, 224, 224)
            logits = np.zeros((1, 5, len(BIO_LABELS)), dtype=np.float32)
            logits[0, 1, 27] = 4.0  # first OCR word -> B-FAT_100G
            logits[0, 2, 28] = 4.0  # second OCR word -> I-FAT_100G
            return [logits]

    image_path = tmp_path / "tiny.jpg"
    Image.new("RGB", (16, 16), "white").save(image_path)
    record = {
        "image": Image.new("RGB", (16, 16), "white"),
        "image_path": str(image_path),
        "tokens": ["18", "g", "unused"],
        "bboxes": [[0, 0, 100, 100]] * 3,
        "ner_tags": [27, 28, 0],
    }
    mapping = {index: label for index, label in enumerate(BIO_LABELS)}
    cache_dir = tmp_path / "predictions"
    prediction = predict_record(
        record, model=FakeSession(), processor=FakeProcessor(), id2label=mapping, cache_dir=cache_dir,
    )
    assert prediction["runtime"] == "onnxruntime-cpu"
    assert prediction["predicted_labels"] == ["B-FAT_100G", "I-FAT_100G", "O"]
    assert prediction["extracted_entities"][0]["text"] == "18 g"
    assert prediction["cache_hit"] is False
    cached = predict_record(
        record, model=FakeSession(), processor=FakeProcessor(), id2label=mapping, cache_dir=cache_dir,
    )
    assert cached["cache_hit"] is True
    upload_record = {key: value for key, value in record.items() if key != "ner_tags"}
    upload = predict_record(
        upload_record, model=FakeSession(), processor=FakeProcessor(), id2label=mapping, cache_dir=cache_dir,
    )
    assert upload["cache_hit"] is True
    assert upload["gold_labels"] is None


def test_subword_alignment_uses_word_ids_once_per_original_token():
    word_ids = [None, 0, 0, 1, None]
    subword_labels = ["O", "B-FAT_100G", "I-FAT_100G", "I-FAT_100G", "O"]
    aligned = [None, None]
    for word_id, label in zip(word_ids, subword_labels):
        if word_id is not None and aligned[word_id] is None:
            aligned[word_id] = label
    assert aligned == ["B-FAT_100G", "I-FAT_100G"]


def test_sliding_windows_cover_long_original_ocr_sequence(tmp_path):
    class DynamicEncoding(dict):
        def __init__(self, length):
            super().__init__({
                "input_ids": np.zeros((1, length + 2), dtype=np.int64),
                "attention_mask": np.ones((1, length + 2), dtype=np.int64),
                "bbox": np.zeros((1, length + 2, 4), dtype=np.int64),
                "pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32),
            })
            self.length = length

        def word_ids(self, batch_index=0):
            return [None, *range(self.length), None]

    class DynamicProcessor:
        def __call__(self, **kwargs):
            return DynamicEncoding(len(kwargs["text"]))

    class FakeInput:
        def __init__(self, name, input_type):
            self.name, self.type = name, input_type

    class DynamicSession:
        def get_inputs(self):
            return [FakeInput("input_ids", "tensor(int64)"), FakeInput("bbox", "tensor(int64)"),
                    FakeInput("attention_mask", "tensor(int64)"), FakeInput("pixel_values", "tensor(float)")]

        def run(self, _outputs, feed):
            length = feed["input_ids"].shape[1]
            logits = np.zeros((1, length, len(BIO_LABELS)), dtype=np.float32)
            logits[:, :, 0] = 3.0
            return [logits]

    image_path = tmp_path / "long.jpg"
    image = Image.new("RGB", (32, 32), "white")
    image.save(image_path)
    count = 453
    record = {
        "image": image, "image_path": str(image_path), "barcode": "test-long",
        "tokens": [f"w{index}" for index in range(count)],
        "bboxes": [[0, 0, 20, 20]] * count,
        "ner_tags": [0] * count,
    }
    mapping = {index: label for index, label in enumerate(BIO_LABELS)}
    result = predict_record(
        record, model=DynamicSession(), processor=DynamicProcessor(), id2label=mapping,
        cache_dir=tmp_path / "cache",
    )
    assert result["covered_token_count"] == count
    assert result["coverage_fraction"] == 1.0
    assert len(result["predicted_labels"]) == count
    assert result["inference_windows"] > 1


def test_local_onnx_loader_checks_ram_and_never_downloads(tmp_path, monkeypatch):
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"not opened by test")
    monkeypatch.setattr("part2.layoutlm_bio.available_ram_bytes", lambda: 1024**3)
    with pytest.raises(InsufficientResourcesError, match="No weights were loaded"):
        load_pretrained_model(model_dir)
    assert not (model_dir / "model.safetensors").exists()


def test_local_onnx_loader_reports_missing_sidecars_without_download(tmp_path, monkeypatch):
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"local model placeholder")
    monkeypatch.setattr("part2.layoutlm_bio.available_ram_bytes", lambda: 8 * 1024**3)
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_pretrained_model(model_dir)
    assert not (model_dir / "config.json").exists()


def test_structured_decoder_normalizes_decimal_comma_and_basis():
    payload = decode_bio_entities(
        ["6,2g", "1,6 g"],
        ["B-FAT_100G", "B-FAT_SERVING"],
        barcode="123",
    )
    assert payload["nutrients"]["per_100g"]["fat"][0]["normalized_value"] == 6.2
    assert payload["nutrients"]["per_100g"]["fat"][0]["normalized_unit"] == "g"
    assert payload["nutrients"]["per_serving"]["fat"][0]["value"] == 1.6
    assert payload["raw_predictions"]["bio_labels"] == ["B-FAT_100G", "B-FAT_SERVING"]


def test_structured_decoder_flags_missing_units_and_ambiguous_values():
    payload = decode_bio_entities(
        ["12", "12 g and 14 g"],
        ["B-FAT_100G", "B-SUGARS_100G"],
    )
    assert len(payload["uncertain_fields"]) == 2
    assert payload["uncertain_fields"][0]["uncertainty_reasons"] == ["unit_missing_or_not_read"]
    assert "multiple_numeric_values_in_one_BIO_span" in payload["uncertain_fields"][1]["uncertainty_reasons"]
    assert payload["nutrients"] == {}


def test_structured_decoder_preserves_duplicate_entity_warnings():
    payload = decode_bio_entities(
        ["6 g", "7 g"],
        ["B-FAT_100G", "B-FAT_100G"],
    )
    assert payload["nutrients"] == {}
    assert len(payload["uncertain_fields"]) == 2
    assert all("duplicate_field_conflicting_values" in item["uncertainty_reasons"] for item in payload["uncertain_fields"])


def test_coverage_metrics_do_not_score_uncovered_gold_span():
    metrics = evaluate_sequences(
        [["O", "B-FAT_100G", "I-FAT_100G"]],
        [["O", "B-FAT_100G", "O"]], aligned=True,
        coverage_masks=[[True, True, False]],
    )
    assert metrics["token_support"] == 2
    assert metrics["token_coverage"] == pytest.approx(2 / 3)
    assert metrics["gold_entity_coverage"] == 0
    assert metrics["gold_spans"] == 0


def test_gold_field_comparison_preserves_basis_and_discloses_uncertain():
    from part2.layoutlm_bio import compare_fields_to_gold

    result = compare_fields_to_gold(
        ["6,2g", "12"], ["B-FAT_100G", "B-SUGARS_100G"],
        ["B-FAT_SERVING", "B-SUGARS_100G"],
    )
    assert result["gold_fields"] == 1
    assert result["gold_uncertain_spans"] == 1
    assert result["true_positive"] == 0
    assert result["f1"] == 0
    assert result["missing_gold_tuples"][0][1] == "per_100g"


def test_api_mass_units_are_compatible_and_bounds_are_not_exact():
    from part2.layoutlm_bio import compare_fields_to_off_metadata

    reference = {"found": True, "product": {"nutriments": {"salt_100g": 0.1}}}
    decoded = decode_bio_entities(["100 mg"], ["B-SALT_100G"])
    result = compare_fields_to_off_metadata(decoded, reference)
    assert result["reference_is_visual_gold"] is False
    assert result["exact_value_accuracy"] == 1
    bounded = decode_bio_entities(["<0.1 g"], ["B-SALT_100G"])
    assert compare_fields_to_off_metadata(bounded, reference)["exact_value_accuracy"] == 0


def test_empty_api_reference_does_not_invent_zero_accuracy():
    from part2.layoutlm_bio import compare_fields_to_off_metadata

    result = compare_fields_to_off_metadata(
        decode_bio_entities(["6g"], ["B-FAT_100G"]),
        {"found": True, "product": {"nutriments": {}}},
    )
    assert result["exact_value_accuracy"] is None
    assert result["field_tuple_f1"] is None


def test_adaptive_windows_retry_truncated_words_and_fail_missing_alignment():
    from part2.layoutlm_bio import _sliding_window_logits

    class Encoding(dict):
        def __init__(self, count):
            self.count = count
            super().__init__(input_ids=np.zeros((1, count), dtype=np.int64))

        def word_ids(self, batch_index=0):
            return list(range(self.count))

    class Processor:
        def __call__(self, **kwargs):
            return Encoding(min(40, len(kwargs["text"])))

    class Session:
        def get_inputs(self):
            from types import SimpleNamespace
            return [SimpleNamespace(name="input_ids", type="tensor(int64)")]

        def run(self, outputs, feed):
            return [np.zeros((1, feed["input_ids"].shape[1], 2))]

    rows, windows, final_limit = _sliding_window_logits(
        Image.new("RGB", (16, 16)), ["word"] * 453, [[0, 0, 1, 1]] * 453,
        model=Session(), processor=Processor(), id2label={0: "O", 1: "B-FAT_100G"},
    )
    assert len(rows) == 453
    assert final_limit < 180
    assert windows > 3

    class MissingProcessor:
        def __call__(self, **kwargs):
            return Encoding(0)

    with pytest.raises(RuntimeError, match="misses original token"):
        _sliding_window_logits(
            Image.new("RGB", (16, 16)), ["word"] * 20, [[0, 0, 1, 1]] * 20,
            model=Session(), processor=MissingProcessor(), id2label={0: "O", 1: "B-FAT_100G"},
        )


def test_easyocr_reader_disables_downloads_gpu_and_quantization(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import part2.ocr as adapter
    from part2.config import OCRConfig

    captured = {}

    def fake_reader(languages, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=fake_reader))
    monkeypatch.setattr(adapter, "_READERS", {})
    adapter._reader(OCRConfig(model_storage_directory=tmp_path, allow_model_downloads=False))
    assert captured["gpu"] is False
    assert captured["download_enabled"] is False
    assert captured["quantize"] is False


def test_upload_word_splitting_box_normalization_and_no_gold(tmp_path, monkeypatch):
    import part2.layoutlm_bio as bio
    from part2.schemas import OCRDetection, OCRResult

    path = tmp_path / "upload.png"
    Image.new("RGB", (100, 100), "white").save(path)
    detection = OCRDetection("Fat 6 g", [[10, 10], [90, 10], [90, 20], [10, 20]], 0.9, 0)
    monkeypatch.setattr("part2.ocr.run_easyocr", lambda *args: OCRResult(str(path), [detection], elapsed_seconds=0.1))

    def fake_predict(record, **kwargs):
        assert "ner_tags" not in record
        assert record["tokens"] == ["Fat", "6", "g"]
        assert record["bboxes"] == [[100, 100, 900, 200]] * 3
        return {"predicted_labels": ["O", "B-FAT_100G", "I-FAT_100G"], "gold_labels": None}

    monkeypatch.setattr(bio, "predict_record", fake_predict)
    result = bio.predict_uploaded_image(path, model=object(), processor=object(), id2label={})
    assert result["gold_available"] is False
    assert result["structured_output"]["parsed_fields"][0]["value"] == 6
    assert "shared" in result["structured_output"]["ocr_provenance"]["box_granularity"]