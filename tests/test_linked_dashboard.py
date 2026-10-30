import json
from pathlib import Path

import pandas as pd
import pytest

from nutrition_extraction.linked_data import (
    MATCHED, annotation_preview, linked_product, load_linked, products_for_eda,
)
from nutrition_extraction.eda import analyze_export


def test_common_eda_excludes_unmatched_and_preserves_string_ids():
    products = pd.DataFrame([
        {"barcode": "001", "brand": "Brand", "category": "Snacks", "country": "France",
         "product_name": "Test", "nutrition_image_url": "https://example.org/a.jpg",
         "structured_nutrients": json.dumps({"fat_100g": "1.5", "sugars_100g": "0", "junk": "omit"}), "match_status": MATCHED},
        {"barcode": "002", "structured_nutrients": "{}", "match_status": "barcode_unmatched"},
    ])
    frame = products_for_eda(products)
    assert frame["code"].tolist() == ["001"]
    assert frame.loc[0, "fat_100g"] == "1.5"
    assert frame.loc[0, "sugars_100g"] == "0"
    assert frame.loc[0, "countries"] == "France"
    assert "junk" not in frame
    assert len(products) == 2  # input never mutated
    assert linked_product(products, "1") is None
    assert linked_product(products, "001")["product_name"] == "Test"
    assert analyze_export(frame)[0]["rows_examined"] == 1


def test_duplicate_product_keys_refused():
    frame = pd.DataFrame([{"barcode": "001", "match_status": MATCHED, "structured_nutrients": "{}"}]*2)
    with pytest.raises(ValueError, match="unique"):
        products_for_eda(frame)


def test_real_common_subset_keeps_all_official_images_and_gold():
    directory = Path(__file__).resolve().parents[1] / "outputs/part1_part2_integration"
    if not (directory / "provenance.json").exists():
        pytest.skip("Integration artifacts not bundled")
    products, images, provenance = load_linked(directory)
    eda = products_for_eda(products)
    assert len(eda) == provenance["summary"]["overlapping_barcodes"]
    assert len(images) == provenance["part2"]["images"]
    assert images.groupby("dataset_split").size().to_dict() == provenance["part2"]["split_rows"]
    assert (~images["match_status"].eq(MATCHED)).sum() == provenance["summary"]["unmatched_barcodes"]
    assert eda["code"].str.startswith("0").any()
    key = images.iloc[0]["image_key"]
    tokens = annotation_preview(directory, key, limit=3)
    assert tokens["image_key"].eq(key).all()
    assert len(tokens) <= 3
    assert {"gold_bio_id", "gold_bio_label", "bounding_box", "ocr_token"}.issubset(tokens)