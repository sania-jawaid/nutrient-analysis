"""Download a bounded Open Food Facts sample with useful EDA fields."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
BASE_FIELDS = {
    "code", "product_name", "generic_name", "brands", "categories", "countries",
    "ingredients_text", "image_url", "image_front_url", "image_nutrition_url",
    "image_ingredients_url", "quantity", "serving_size", "nutrition_data_per",
    "energy-kcal_100g", "energy-kj_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g",
    "sodium_100g", "nutrition_grade_fr",
}
NUTRIENT_PREFIXES = (
    "energy-kj_", "energy-kcal_", "energy_", "fat_", "saturated-fat_", "trans-fat_",
    "carbohydrates_", "sugars_", "fiber_", "fibre_", "proteins_", "salt_", "sodium_",
    "cholesterol_", "vitamin-", "calcium_", "iron_", "potassium_",
)


def acquire_sample(url: str, output_path: Path, row_limit: int = 50_000) -> pd.DataFrame:
    """Read a reproducible top-of-file sample and save only columns useful for EDA.

    The input export is gzip-compressed TSV. The upstream file is updated over time;
    record the acquisition date and URL alongside the output for reproducibility.
    """
    if row_limit < 1:
        raise ValueError("row_limit must be a positive integer")
    LOGGER.info("Inspecting export header from %s", url)
    header = pd.read_csv(url, sep="\t", compression="gzip", nrows=0)
    available = set(map(str, header.columns))
    selected = [
        name for name in header.columns
        if name in BASE_FIELDS
        or (str(name).endswith(("_100g", "_100ml")) and str(name).startswith(NUTRIENT_PREFIXES))
    ]
    selected.extend(
        name for name in header.columns
        if "image" in str(name).lower() and name not in selected
    )
    selected = list(dict.fromkeys(selected))
    required = {"code", "product_name"}
    missing = required - set(selected)
    if missing:
        raise ValueError(f"Expected columns absent from the current export: {sorted(missing)}")
    LOGGER.info("Loading up to %d rows and %d/%d available columns", row_limit, len(selected), len(available))
    frame = pd.read_csv(
        url,
        sep="\t",
        compression="gzip",
        usecols=selected,
        nrows=row_limit,
        low_memory=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(output_path.suffix + ".source.txt")
    metadata_path.write_text(
        f"source_url={url}\nrows={len(frame)}\ncolumns={len(frame.columns)}\n",
        encoding="utf-8",
    )
    schema_path = output_path.parent / "openfoodfacts_source_schema.json"
    schema_path.write_text(
        json.dumps({"source_url": url, "columns": header.columns.astype(str).tolist()}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved %d rows to %s", len(frame), output_path)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("data/openfoodfacts_sample.csv"))
    parser.add_argument("--rows", type=int, default=50_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    acquire_sample(args.url, args.output, args.rows)


if __name__ == "__main__":
    main()
