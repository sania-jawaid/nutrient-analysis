"""Pure dashboard transformations, separate from model evaluation and source data."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def label_set(value) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip().casefold() for part in str(value).split(",") if part.strip()}


def filter_products(frame, countries, categories):
    """OR within a filter, AND across filters. Missing metadata cannot match."""
    mask = pd.Series(True, index=frame.index)
    for column, selected in [("countries", countries), ("categories", categories)]:
        if selected:
            wanted = {value.strip().casefold() for value in selected}
            mask &= frame[column].map(lambda value: bool(label_set(value) & wanted)) if column in frame else False
    return frame.loc[mask].copy()


def merged_image_products(products, images):
    """Image-grain left join, not a token-level multiplication of product metadata."""
    return images.merge(products, how="left", on="barcode", validate="many_to_one",
                        suffixes=("", "_product"), sort=False)


def with_image_links(table, images):
    """Attach an image URL by image key, barcode+ID, or a unique product barcode."""
    result = table.copy()
    if "image_url" in result and result["image_url"].notna().all():
        return result
    if "image_key" in result:
        keys = ["image_key"]
    elif {"barcode", "image_id"}.issubset(result):
        keys = ["barcode", "image_id"]
    elif "barcode" in result:
        keys = ["barcode"]
    else:
        return result
    links = images[keys + ["image_url"]].copy()
    for column in keys:
        result[column] = result[column].astype("string")
        links[column] = links[column].astype("string")
    # If multiple images are possible without an image ID, do not choose arbitrarily.
    unique = links.groupby(keys, dropna=False)["image_url"].nunique()
    safe_keys = unique.loc[unique.eq(1)].reset_index()[keys]
    links = links.merge(safe_keys, on=keys).drop_duplicates(keys)
    result = result.merge(links, how="left", on=keys, suffixes=("", "_linked"), validate="many_to_one")
    if "image_url_linked" in result:
        result["image_url"] = result["image_url"].fillna(result.pop("image_url_linked"))
    return result


def annotation_rows(directory, image_keys):
    import pyarrow.dataset as ds
    dataset = ds.dataset(Path(directory) / "integrated_annotations.parquet", format="parquet")
    return dataset.to_table(filter=ds.field("image_key").isin(list(image_keys))).to_pandas()


def label_support(tokens):
    if tokens.empty:
        return pd.DataFrame(columns=["label", "train_token_support", "test_token_support", "total_support"])
    counts = tokens.groupby(["gold_bio_label", "dataset_split"]).size().unstack(fill_value=0)
    for split in ["train", "test"]:
        if split not in counts:
            counts[split] = 0
    counts = counts[["train", "test"]].rename(columns={"train": "train_token_support", "test": "test_token_support"})
    counts.index.name = "label"
    counts["total_support"] = counts.sum(axis=1)
    return counts.reset_index()


def numeric_modes(values):
    """Return every tied mode and frequency; do not imply a unique mode when all tie."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"values": [], "frequency": 0, "status": "no_numeric_values"}
    frequencies = numeric.value_counts()
    maximum = int(frequencies.max())
    modes = sorted(float(value) for value in frequencies.loc[frequencies.eq(maximum)].index)
    return {"values": modes, "frequency": maximum,
            "status": "all_values_unique" if maximum == 1 else "multiple_modes" if len(modes) > 1 else "unique_mode"}