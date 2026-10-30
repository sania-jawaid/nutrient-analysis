"""Read-only dashboard access to the completed linked dataset (no full scan)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MATCHED = "barcode_matched_image_unverified"
PRODUCT_COLUMNS = {
    "barcode": "code", "brand": "brands", "category": "categories",
    "country": "countries", "nutrition_image_url": "image_nutrition_url",
}


def products_for_eda(products: pd.DataFrame) -> pd.DataFrame:
    """One row per matched product; unpack raw nutrient fields, not OCR estimates."""
    if products["barcode"].duplicated().any():
        raise ValueError("Linked Products must have unique barcodes; refusing EDA fan-out.")
    matched = products.loc[products["match_status"].eq(MATCHED)].copy()
    matched["barcode"] = matched["barcode"].astype("string")
    nutrient_rows = []
    for payload in matched["structured_nutrients"]:
        values = json.loads(payload) if isinstance(payload, str) else {}
        if not isinstance(values, dict):
            raise ValueError("Structured nutrients must be a JSON object.")
        nutrient_rows.append({name: value for name, value in values.items()
                              if name.endswith(("_100g", "_100ml", "_serving"))})
    # Reset indexes before expanding nutrients to avoid accidental row misalignment.
    frame = matched.rename(columns=PRODUCT_COLUMNS).drop(columns=["structured_nutrients"]).reset_index(drop=True)
    nutrients = pd.DataFrame(nutrient_rows, index=frame.index)
    if set(nutrients) & set(frame):
        raise ValueError("Nutrient columns overlap product metadata.")
    return pd.concat([frame, nutrients], axis=1)


def load_linked(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = Path(directory)
    products = pd.read_parquet(directory / "integrated_products.parquet")
    images = pd.read_parquet(directory / "integrated_images.parquet")
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if provenance["summary"].get("status") != "measured_complete_part1_snapshot":
        raise ValueError("Linked metadata lacks confirmed complete-source overlap provenance.")
    if products["barcode"].duplicated().any() or images["image_key"].duplicated().any():
        raise ValueError("Linked table keys are not unique.")
    if len(images) != provenance["part2"]["images"]:
        raise ValueError("Linked images do not retain the complete Part 2 inventory.")
    if not set(images["barcode"].dropna()).issubset(set(products["barcode"])):
        raise ValueError("Linked images contain missing product foreign keys.")
    if int(products["match_status"].eq(MATCHED).sum()) != provenance["summary"]["overlapping_barcodes"]:
        raise ValueError("Matched product count differs from the measured summary.")
    products["barcode"] = products["barcode"].astype("string")
    images["barcode"] = images["barcode"].astype("string")
    return products, images, provenance


def annotation_preview(directory: str | Path, image_key: str, limit: int = 100) -> pd.DataFrame:
    """Predicate-filter a single image, without loading all annotations into pandas."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(directory) / "integrated_annotations.parquet", format="parquet")
    return dataset.scanner(filter=ds.field("image_key") == image_key).head(limit).to_pandas()


def linked_product(products: pd.DataFrame, barcode: str) -> dict | None:
    """Exact string lookup: never coerce numeric IDs or remove leading zeros."""
    rows = products.loc[products["barcode"].eq(str(barcode))]
    return rows.iloc[0].to_dict() if not rows.empty else None