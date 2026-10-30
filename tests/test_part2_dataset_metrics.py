import pytest
import torch
from PIL import Image

from part2.config import DatasetConfig
from part2.dataset import BIO_LABELS, collate_documents
from part2.evaluation import aligned_bio_entity_metrics, evaluate_verified_ocr


def test_dataset_subset_is_bounded_and_official_split_is_required() -> None:
    assert len(BIO_LABELS) == 71
    with pytest.raises(ValueError):
        DatasetConfig(subset_size=51)
    with pytest.raises(ValueError):
        DatasetConfig(split="validation")


def test_collate_pads_variable_length_sequences() -> None:
    batch = [
        {"image": Image.new("RGB", (12, 8)), "image_path": "a", "barcode": "001", "tokens": ["fat"], "bboxes": [[1, 2, 3, 4]], "ner_tags": [0], "bio_labels": ["O"], "meta": {}, "split": "train"},
        {"image": Image.new("RGB", (12, 8)), "image_path": "b", "barcode": "002", "tokens": ["sugar", "4g"], "bboxes": [[1, 2, 3, 4], [4, 5, 6, 7]], "ner_tags": [0, 27], "bio_labels": ["O", "B-FAT_100G"], "meta": {}, "split": "train"},
    ]
    result = collate_documents(batch)
    assert result["boxes"].shape == (2, 2, 4)
    assert result["ner_tags"].tolist() == [[0, -100], [0, 27]]
    assert result["attention_mask"].tolist() == [[True, False], [True, True]]
    assert result["barcodes"] == ["001", "002"]


def test_ocr_metrics_require_verified_reference() -> None:
    assert evaluate_verified_ocr([])["status"] == "not_established"
    report = evaluate_verified_ocr([{"verified": True, "reference_text": "fat 9 g", "hypothesis_text": "fat 9 g"}])
    assert report["cer"] == 0
    assert report["wer"] == 0


def test_bio_metrics_require_alignment() -> None:
    with pytest.raises(ValueError):
        aligned_bio_entity_metrics([["B-FAT", "I-FAT"]], [["B-FAT", "I-FAT"]], token_alignment_verified=False)
    metrics = aligned_bio_entity_metrics([["B-FAT", "I-FAT", "O"]], [["B-FAT", "I-FAT", "O"]], token_alignment_verified=True)
    assert metrics["f1"] == 1.0