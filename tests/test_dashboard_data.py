import pandas as pd

from nutrition_extraction.dashboard_data import (
    filter_products, numeric_modes, label_support, with_image_links, merged_image_products,
)
from nutrition_extraction.eda import nutrition_statistics


def test_country_and_snacks_filters_apply_and_missing_metadata_does_not_match():
    products = pd.DataFrame({"code": ["001", "002", "003"],
        "countries": [" France , Belgium", "france", "Canada"],
        "categories": ["Snacks, Sweet snacks", "Beverages", "Snacks"]})
    filtered = filter_products(products, ["FRANCE"], ["snacks"])
    assert filtered["code"].tolist() == ["001"]
    assert filter_products(products.drop(columns="countries"), ["France"], []).empty
    assert len(products) == 3


def test_mode_values_include_ties_and_disclose_unique_or_empty():
    assert numeric_modes(pd.Series([1, 1, 2, 2, 3])) == {"values": [1.0, 2.0], "frequency": 2, "status": "multiple_modes"}
    assert numeric_modes(pd.Series([1, 2]))["status"] == "all_values_unique"
    assert numeric_modes(pd.Series(["invalid", None]))["frequency"] == 0
    stats = nutrition_statistics(pd.DataFrame({"fat_100g": [1, 1, 2, 2, 3]}), ["fat_100g"])
    assert stats.iloc[0]["mode"] == "1; 2"
    assert stats.iloc[0]["mode_frequency"] == 2


def test_link_resolution_uses_image_id_and_never_multiplies_product_rows():
    images = pd.DataFrame({"barcode": ["001", "001", "002"], "image_id": ["1", "2", "1"],
        "image_key": ["train:0", "test:0", "train:1"],
        "image_url": ["https://example/1", "https://example/2", "https://example/3"]})
    rows = pd.DataFrame({"barcode": ["001", "002"], "image_id": ["2", "1"]})
    assert with_image_links(rows, images)["image_url"].tolist() == ["https://example/2", "https://example/3"]
    product_rows = with_image_links(pd.DataFrame({"barcode": ["001", "002"]}), images)
    assert len(product_rows) == 2
    assert pd.isna(product_rows.iloc[0]["image_url"])  # ambiguous image, not guessed
    products = pd.DataFrame({"barcode": ["001", "002"], "brand": ["A", "B"]})
    joined = merged_image_products(products, images)
    assert len(joined) == 3
    assert joined["image_key"].tolist() == images["image_key"].tolist()


def test_label_support_respects_actual_selected_tokens_and_keeps_split_labels():
    tokens = pd.DataFrame({"gold_bio_label": ["O", "B-FAT_100G", "O"], "dataset_split": ["train", "test", "train"]})
    counts = label_support(tokens.loc[tokens["dataset_split"].eq("test")])
    assert counts["total_support"].sum() == 1
    assert counts["train_token_support"].sum() == 0
    assert counts["test_token_support"].sum() == 1
    assert label_support(tokens.iloc[:0]).empty