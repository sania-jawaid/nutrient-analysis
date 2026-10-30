import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.dataset_integration_support import (
    barcode, validate_full_source, scan_complete_part1, flatten_row,
    part2_inventory, overlap_summary, export_linked, _schemas, image_reference_status,
)


def test_conservative_barcode_and_sample_gate(tmp_path):
    assert barcode(" 000123 ") == ("000123", "valid")
    assert barcode("123")[0] != barcode("000123")[0]
    for value in ("123.0", "1e3", "１２３", 123, 123.0):
        assert barcode(value)[0] is None
    assert not validate_full_source(None)["available"]
    sample = tmp_path / "openfoodfacts_sample.csv"
    sample.write_text("code,product_name\n001,A\n")
    with pytest.raises(ValueError, match="sample"):
        validate_full_source(sample, confirmed=True)


def test_full_scan_join_and_parquet_round_trip_on_synthetic_data(tmp_path):
    # Fixtures validate code only; these are never reported as real dataset overlap.
    source = tmp_path / "fixture.tsv"
    source.write_text("code\tproduct_name\tbrands\tfat_100g\timage_nutrition_url\n"
                      "0001\tA\tBrand\t1.2\thttps://example.org/1.jpg\n"
                      "0001\tB\tBrand\t2.2\thttps://example.org/2.jpg\n"
                      "1\tOther\t\t0\t\n"
                      "bad\tInvalid\t\t\t\n", encoding="utf-8")
    rows, tokens = [], []
    labels = ["O", "B-FAT_100G", "I-FAT_100G"]
    for index, code in enumerate(["0001", "0001", "9999", "bad"]):
        image, annotations = flatten_row({"meta": {"barcode": code, "image_id": str(index), "image_url": "https://example.org/1.jpg"},
                                         "image": {}, "tokens": ["1", "g"], "bboxes": [[0, 0, 1, 1]]*2, "ner_tags": [1, 2]},
                                        "test", index, labels)
        rows.append(image)
        tokens.extend(annotations)
    image_path, annotation_path = tmp_path / "images.parquet", tmp_path / "annotations.parquet"
    schemas = _schemas()
    pq.write_table(pa.Table.from_pylist(rows, schema=schemas[0]), image_path)
    pq.write_table(pa.Table.from_pylist(tokens, schema=schemas[1]), annotation_path)
    manifest = dict(images_path=str(image_path), annotations_path=str(annotation_path), images=4, annotations=8)
    images, p2_summary = part2_inventory(manifest)
    scan = scan_complete_part1(source, {"0001", "9999"}, tmp_path / "work", confirmed=True, chunk_rows=2, minimum_rows=1)
    assert scan["rows"] == 4
    assert scan["unique_barcodes"] == 2
    assert scan["duplicate_barcode_rows"] == 1
    assert scan["matched_barcodes"] == ["0001"]
    assert scan["unmatched_barcodes"] == ["9999"]
    summary = overlap_summary(scan, images, p2_summary)
    assert summary["matched_images"] == 2
    assert summary["overlap_percent_of_part2"] == 50
    output = tmp_path / "output"
    products, linked, quality = export_linked(scan, images, manifest, output, summary, write_outputs=False)
    assert not (output / "integrated_products.parquet").exists()
    assert len(products) == 2 and len(linked) == 4
    assert products[0]["barcode"] == "0001" and products[0]["source_record_count"] == 2
    assert products[0]["conflicting_product_records"]
    assert products[0]["product_name"] == "A"
    assert json.loads(products[0]["structured_nutrients"])["fat_100g"] == "1.2"
    assert linked[2]["match_status"] == "barcode_unmatched"
    assert linked[3]["match_status"] == "invalid_or_missing_barcode"
    export_linked(scan, images, manifest, output, summary)
    assert pq.read_table(output / "integrated_products.parquet")["barcode"].to_pylist() == ["0001", "9999"]
    assert (output / "integrated_annotations.parquet").read_bytes() == annotation_path.read_bytes()
    assert quality["metadata_coverage"]["structured_nutrients"] == 1


def test_inconsistent_arrays_are_not_silently_truncated():
    image, annotations = flatten_row({"meta": {"barcode": "001"}, "tokens": ["a", "b"],
                                     "bboxes": [[0, 0, 1, 1]], "ner_tags": [0]}, "train", 0, ["O"])
    assert image["annotation_status"] == "inconsistent"
    assert len(annotations) == 2
    assert annotations[1]["bounding_box"] is None
    assert annotations[1]["gold_bio_id"] is None
    with pytest.raises(ValueError, match="EOF"):
        overlap_summary({}, [], {})
    with pytest.raises(ValueError, match="Confirm"):
        export_linked({}, [], {}, "unused", {})


def test_url_agreement_never_claims_verified_images():
    assert image_reference_status("https://x/a.jpg", "https://x/a.jpg", "a") == "exact_url_metadata_match_not_pixel_verified"
    assert image_reference_status(None, "https://x/a.jpg", "a") == "not_comparable"
    assert "not_pixel_verified" in image_reference_status("https://x/a.jpg?v=1", "https://x/a.jpg?v=2", "a")


def test_full_source_row_count_sidecar_is_not_a_substring_sample(tmp_path):
    source = tmp_path / "complete.tsv"
    source.write_text("code\tproduct_name\n")
    sidecar = source.with_suffix(".tsv.source.txt")
    sidecar.write_text("rows=5000000\n")
    assert validate_full_source(source, confirmed=True)["available"]
    sidecar.write_text("rows=50000\n")
    with pytest.raises(ValueError, match="sample"):
        validate_full_source(source, confirmed=True)


def test_all_splits_cached_without_images_and_partial_page_rejected(tmp_path, monkeypatch):
    from urllib.parse import parse_qs, urlsplit
    import scripts.dataset_integration_support as helper

    def request(url):
        if "api/datasets" in url:
            return {"sha": "test-revision", "cardData": {"dataset_info": {
                "splits": [{"name": "train", "num_examples": 1}, {"name": "test", "num_examples": 1}],
                "features": [{"name": "ner_tags", "sequence": {"class_label": {"names": {"0": "O"}}}}]}}}
        query = parse_qs(urlsplit(url).query)
        split = query["split"][0]
        return {"num_rows_total": 1, "dataset_git_revision": "test-revision",
                "rows": [{"row_idx": 0, "truncated_cells": [], "row": {
                    "meta": {"barcode": "001", "image_id": split, "split": split},
                    "image": {"src": "https://never-downloaded.example/image.jpg"},
                    "tokens": ["text"], "bboxes": [[0, 0, 1, 1]], "ner_tags": [0]}}]}

    monkeypatch.setattr(helper, "_json_request", request)
    manifest = helper.load_all_part2(tmp_path / "cache")
    assert manifest["images"] == 2 and manifest["annotations"] == 2
    assert manifest["split_rows"] == {"train": 1, "test": 1}
    monkeypatch.setattr(helper, "_json_request", lambda url: pytest.fail("must reuse cache"))
    assert helper.load_all_part2(tmp_path / "cache", allow_metadata_requests=False) == manifest

    def incomplete(url):
        result = request(url)
        if "api/datasets" not in url:
            result["rows"] = []
        return result

    monkeypatch.setattr(helper, "_json_request", incomplete)
    with pytest.raises(ValueError, match="Incomplete"):
        helper.load_all_part2(tmp_path / "partial")
    assert not (tmp_path / "partial/manifest.json").exists()