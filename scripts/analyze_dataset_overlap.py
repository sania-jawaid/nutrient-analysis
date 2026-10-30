"""Compare Part 1 Open Food Facts products with the HF nutrient-layout dataset.

The Hugging Face datasets-server rows endpoint returns annotation metadata and
image URLs; this script never requests the image URLs or downloads image files.
Product IDs are compared as trimmed strings, with no padding or truncation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_DATASET = "openfoodfacts/nutrient-detection-layout"
DEFAULT_SAMPLE = Path("data/openfoodfacts_eda_enriched.csv")
DEFAULT_API_ROOT = "https://datasets-server.huggingface.co"
HF_API_ROOT = "https://huggingface.co/api/datasets"
PAGE_SIZE = 100
IMAGE_COLUMNS = (
    "image_url",
    "image_small_url",
    "image_ingredients_url",
    "image_ingredients_small_url",
    "image_nutrition_url",
    "image_nutrition_small_url",
)
NUTRITION_IMAGE_COLUMNS = ("image_nutrition_url", "image_nutrition_small_url")
PLACEHOLDERS = {"", "nan", "none", "null", "<na>"}


def request_json(url: str, *, retries: int = 4) -> dict[str, Any]:
    """Fetch JSON metadata with bounded retries; never follows image URLs."""
    request = Request(
        url,
        headers={"User-Agent": "nutrition-extraction-overlap-audit/1.0"},
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f"Could not fetch dataset metadata: {url}\n{error}") from error
            time.sleep(1.0 * (attempt + 1))
    raise AssertionError("unreachable")


def nonempty(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in PLACEHOLDERS else text


def normalize_id(value: Any) -> tuple[str, str]:
    """Return (normalized ID, status), retaining leading zeroes exactly."""
    text = nonempty(value)
    if not text:
        return "", "missing"
    # OFF barcodes are represented as digit strings. Do not impose a length
    # rule: valid local codes vary, and leading zeroes are significant.
    if not text.isdigit():
        return "", "invalid_non_digit"
    return text, "valid"


def fetch_split_rows(
    dataset: str,
    config: str,
    split: str,
    api_root: str,
    expected_count: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = expected_count
    while total is None or offset < total:
        query = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        payload = request_json(f"{api_root.rstrip('/')}/rows?{query}")
        if total is None:
            total = int(payload.get("num_rows_total", 0))
        batch = payload.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item.get("row", {})
            meta = row.get("meta") or {}
            image = row.get("image") or {}
            rows.append(
                {
                    "barcode_raw": meta.get("barcode"),
                    "image_id": nonempty(meta.get("image_id")),
                    "image_url": nonempty(meta.get("image_url")),
                    "split": nonempty(meta.get("split")) or split,
                    "ocr_url": nonempty(meta.get("ocr_url")),
                    "label_studio_id": meta.get("label_studio_id"),
                    "checked": meta.get("checked"),
                    "usda_table": meta.get("usda_table"),
                    "nutrition_text": meta.get("nutrition_text"),
                    "no_nutrition_table": meta.get("no_nutrition_table"),
                    "image_width": image.get("width"),
                    "image_height": image.get("height"),
                    # Preserve shape evidence without retaining tokens/boxes.
                    "token_count": len(row.get("tokens") or []),
                    "bbox_count": len(row.get("bboxes") or []),
                    "tag_count": len(row.get("ner_tags") or []),
                }
            )
        offset += len(batch)
        if offset >= (total or 0):
            break
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(
            f"Fetched {len(rows)} rows for {split}, but repository metadata reports "
            f"{expected_count}. Refusing to report a partial overlap."
        )
    return rows


def get_dataset_metadata(dataset: str) -> dict[str, Any]:
    return request_json(f"{HF_API_ROOT}/{dataset}")


def get_split_counts(metadata: dict[str, Any]) -> dict[str, int]:
    info = metadata.get("cardData", {}).get("dataset_info", {})
    return {
        item["name"]: int(item["num_examples"])
        for item in info.get("splits", [])
        if item.get("name") and item.get("num_examples") is not None
    }


def feature_summary(metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    features = metadata.get("cardData", {}).get("dataset_info", {}).get("features", [])
    top_level = [feature.get("name", "") for feature in features]
    labels: list[str] = []
    for feature in features:
        if feature.get("name") == "ner_tags":
            names = (
                feature.get("sequence", {})
                .get("class_label", {})
                .get("names", {})
            )
            if isinstance(names, dict):
                labels = [str(names[key]) for key in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                labels = [str(name) for name in names]
    return top_level, labels


def csv_list(values: set[str] | list[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def image_url_set(record: dict[str, Any], columns: tuple[str, ...]) -> set[str]:
    return {nonempty(record.get(column)) for column in columns if nonempty(record.get(column))}


def build_summary_rows(
    part1: pd.DataFrame,
    annotated: list[dict[str, Any]],
    part1_id_status: list[tuple[str, str]],
    annotation_id_status: list[tuple[str, str]],
    split_counts: dict[str, int],
) -> list[dict[str, Any]]:
    p1_valid = [key for key, status in part1_id_status if status == "valid"]
    ann_valid = [key for key, status in annotation_id_status if status == "valid"]
    p1_counts = Counter(p1_valid)
    ann_counts = Counter(ann_valid)
    p1_ids = set(p1_counts)
    ann_ids = set(ann_counts)
    overlap = p1_ids & ann_ids

    rows = [
        {
            "metric": "part1_rows",
            "value": len(part1),
            "denominator": "records in enriched Part 1 CSV",
        },
        {
            "metric": "part1_missing_ids",
            "value": sum(status == "missing" for _, status in part1_id_status),
            "denominator": "Part 1 rows",
        },
        {
            "metric": "part1_invalid_non_digit_ids",
            "value": sum(status == "invalid_non_digit" for _, status in part1_id_status),
            "denominator": "Part 1 rows",
        },
        {
            "metric": "part1_valid_id_rows",
            "value": len(p1_valid),
            "denominator": "Part 1 rows",
        },
        {
            "metric": "part1_unique_valid_products",
            "value": len(p1_ids),
            "denominator": "unique trimmed digit-only IDs",
        },
        {
            "metric": "part1_duplicate_valid_id_rows",
            "value": sum(count - 1 for count in p1_counts.values()),
            "denominator": "valid Part 1 ID rows beyond first occurrence",
        },
        {
            "metric": "part2_annotation_rows",
            "value": len(annotated),
            "denominator": "HF train+test examples from repository metadata",
        },
        {
            "metric": "part2_missing_barcodes",
            "value": sum(status == "missing" for _, status in annotation_id_status),
            "denominator": "Part 2 annotation rows",
        },
        {
            "metric": "part2_invalid_non_digit_barcodes",
            "value": sum(status == "invalid_non_digit" for _, status in annotation_id_status),
            "denominator": "Part 2 annotation rows",
        },
        {
            "metric": "part2_valid_barcode_rows",
            "value": len(ann_valid),
            "denominator": "Part 2 annotation rows",
        },
        {
            "metric": "part2_unique_valid_products",
            "value": len(ann_ids),
            "denominator": "unique trimmed digit-only barcodes",
        },
        {
            "metric": "part2_duplicate_valid_barcode_rows",
            "value": sum(count - 1 for count in ann_counts.values()),
            "denominator": "valid Part 2 barcode rows beyond first occurrence",
        },
        {
            "metric": "unique_product_overlap",
            "value": len(overlap),
            "denominator": "unique IDs in both datasets",
        },
        {
            "metric": "overlap_percent_of_part1_unique",
            "value": round(100 * len(overlap) / len(p1_ids), 4) if p1_ids else 0,
            "denominator": "Part 1 unique valid products (%)",
        },
        {
            "metric": "overlap_percent_of_part2_unique",
            "value": round(100 * len(overlap) / len(ann_ids), 4) if ann_ids else 0,
            "denominator": "Part 2 unique valid products (%)",
        },
        {
            "metric": "overlapping_part2_annotation_rows",
            "value": sum(key in overlap for key in ann_valid),
            "denominator": "Part 2 annotation rows with an overlapping barcode",
        },
        {
            "metric": "overlapping_annotation_rows_with_any_part1_image_url_match",
            "value": 0,
            "denominator": "overlapping Part 2 annotation rows; exact URL string match",
        },
        {
            "metric": "overlapping_annotation_rows_with_part1_nutrition_url_match",
            "value": 0,
            "denominator": "overlapping Part 2 annotation rows; exact URL string match",
        },
        {
            "metric": "part1_rows_with_nutrition_image_url",
            "value": sum(
                any(nonempty(row.get(column)) for column in NUTRITION_IMAGE_COLUMNS)
                for row in part1.to_dict(orient="records")
            ),
            "denominator": "Part 1 rows",
        },
        {
            "metric": "overlapping_part1_products_with_nutrition_image_url",
            "value": 0,
            "denominator": "overlapping unique products",
        },
        {
            "metric": "overlapping_products_with_exact_any_image_url_match",
            "value": 0,
            "denominator": "overlapping unique products",
        },
        {
            "metric": "overlapping_products_with_exact_nutrition_image_url_match",
            "value": 0,
            "denominator": "overlapping unique products",
        },
        {
            "metric": "part2_train_examples",
            "value": split_counts.get("train", 0),
            "denominator": "HF repository metadata",
        },
        {
            "metric": "part2_test_examples",
            "value": split_counts.get("test", 0),
            "denominator": "HF repository metadata",
        },
    ]

    p1_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, (key, status) in zip(part1.to_dict(orient="records"), part1_id_status):
        if status == "valid":
            p1_by_id[key].append(record)
    ann_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, (key, status) in zip(annotated, annotation_id_status):
        if status == "valid":
            ann_by_id[key].append(record)

    exact_any_products = 0
    exact_nutrition_products = 0
    p1_nutrition_products = 0
    exact_any_rows = 0
    exact_nutrition_rows = 0
    for product_id in overlap:
        p1_records = p1_by_id[product_id]
        ann_records = ann_by_id[product_id]
        p1_any_urls = set().union(*(image_url_set(record, IMAGE_COLUMNS) for record in p1_records))
        p1_nutrition_urls = set().union(
            *(image_url_set(record, NUTRITION_IMAGE_COLUMNS) for record in p1_records)
        )
        if p1_nutrition_urls:
            p1_nutrition_products += 1
        any_hit = False
        nutrition_hit = False
        for ann in ann_records:
            ann_url = nonempty(ann.get("image_url"))
            if ann_url and ann_url in p1_any_urls:
                any_hit = True
                exact_any_rows += 1
            if ann_url and ann_url in p1_nutrition_urls:
                nutrition_hit = True
                exact_nutrition_rows += 1
        exact_any_products += any_hit
        exact_nutrition_products += nutrition_hit

    metric_values = {item["metric"]: item for item in rows}
    metric_values["overlapping_annotation_rows_with_any_part1_image_url_match"]["value"] = exact_any_rows
    metric_values["overlapping_annotation_rows_with_part1_nutrition_url_match"]["value"] = exact_nutrition_rows
    metric_values["overlapping_part1_products_with_nutrition_image_url"]["value"] = p1_nutrition_products
    metric_values["overlapping_products_with_exact_any_image_url_match"]["value"] = exact_any_products
    metric_values["overlapping_products_with_exact_nutrition_image_url_match"]["value"] = exact_nutrition_products
    return rows


def build_overlap_records(
    part1: pd.DataFrame,
    annotated: list[dict[str, Any]],
    part1_id_status: list[tuple[str, str]],
    annotation_id_status: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    p1_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, (key, status) in zip(part1.to_dict(orient="records"), part1_id_status):
        if status == "valid":
            p1_by_id[key].append(record)
    ann_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, (key, status) in zip(annotated, annotation_id_status):
        if status == "valid":
            ann_by_id[key].append(record)

    output: list[dict[str, Any]] = []
    for product_id in sorted(set(p1_by_id) & set(ann_by_id)):
        p1_records = p1_by_id[product_id]
        ann_records = ann_by_id[product_id]
        p1_any_urls = set().union(*(image_url_set(record, IMAGE_COLUMNS) for record in p1_records))
        p1_nutrition_urls = set().union(
            *(image_url_set(record, NUTRITION_IMAGE_COLUMNS) for record in p1_records)
        )
        part1_first = p1_records[0]
        p1_columns = [
            "product_name",
            "brands",
            "countries",
            "serving_size",
            "image_url",
            "image_nutrition_url",
            "image_nutrition_small_url",
            "energy-kj_100g",
            "energy-kcal_100g",
            "fat_100g",
            "saturated-fat_100g",
            "carbohydrates_100g",
            "sugars_100g",
            "fiber_100g",
            "proteins_100g",
            "salt_100g",
            "sodium_100g",
        ]
        product_ann = ann_records
        for ann in product_ann:
            ann_url = nonempty(ann.get("image_url"))
            row: dict[str, Any] = {
                "product_id": product_id,
                "part1_rows_for_product": len(p1_records),
                "part2_annotation_rows_for_product": len(ann_records),
                "part2_split": ann.get("split", ""),
                "part2_image_id": ann.get("image_id", ""),
                "part2_image_url": ann_url,
                "part2_label_studio_id": ann.get("label_studio_id", ""),
                "part2_checked": ann.get("checked", ""),
                "part2_token_count": ann.get("token_count", ""),
                "part2_bbox_count": ann.get("bbox_count", ""),
                "part2_tag_count": ann.get("tag_count", ""),
                "part2_image_width": ann.get("image_width", ""),
                "part2_image_height": ann.get("image_height", ""),
                "part1_any_image_url_exact_match": bool(ann_url and ann_url in p1_any_urls),
                "part1_nutrition_image_url_exact_match": bool(
                    ann_url and ann_url in p1_nutrition_urls
                ),
                "part1_has_nutrition_image_url": bool(p1_nutrition_urls),
                "part1_image_urls": csv_list(p1_any_urls),
                "part1_nutrition_image_urls": csv_list(p1_nutrition_urls),
            }
            for column in p1_columns:
                row[f"part1_{column}"] = nonempty(part1_first.get(column))
            output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    part1: pd.DataFrame,
    annotated: list[dict[str, Any]],
    metadata: dict[str, Any],
    splits: dict[str, int],
    top_features: list[str],
    labels: list[str],
    overlap_rows: list[dict[str, Any]],
) -> None:
    metrics = {row["metric"]: row["value"] for row in summary}
    total = sum(splits.values())
    overlap = int(metrics["unique_product_overlap"])
    shared_ann_rows = int(metrics["overlapping_part2_annotation_rows"])
    exact_url_products = int(metrics["overlapping_products_with_exact_any_image_url_match"])
    exact_nutrition_products = int(
        metrics["overlapping_products_with_exact_nutrition_image_url_match"]
    )
    id_match_rate = 100 * overlap / metrics["part2_unique_valid_products"] if metrics["part2_unique_valid_products"] else 0
    label_families = sorted(
        {
            name[2:].rsplit("_", 1)[0]
            for name in labels
            if name.startswith(("B-", "I-")) and "_" in name
        }
    )
    last_modified = metadata.get("lastModified", "not supplied")
    sha = metadata.get("sha", "not supplied")
    dataset_url = f"https://huggingface.co/datasets/{metadata.get('id', DEFAULT_DATASET)}"
    lines = [
        "# Part 1 ↔ Part 2 Dataset Overlap Analysis",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Executive summary",
        "",
        f"- Part 1 uses the existing enriched Open Food Facts CSV: {len(part1):,} rows and {len(part1.columns)} columns. No EDA input, EDA outputs, or dashboard files were changed.",
        f"- Part 2 is [{metadata.get('id', DEFAULT_DATASET)}]({dataset_url}), revision `{sha}`, last modified {last_modified}; repository metadata reports {total:,} examples ({splits.get('train', 0):,} train and {splits.get('test', 0):,} test).",
        f"- Unique product overlap by trimmed, digit-only string ID: **{overlap:,} products** ({id_match_rate:.2f}% of Part 2 unique valid products; {metrics['overlap_percent_of_part1_unique']:.2f}% of Part 1 unique valid products). This is an identifier match, not proof of identical image pixels or label correctness.",
        f"- Those products correspond to {shared_ann_rows:,} Part 2 annotation rows. Exact image URL string matches occur for {exact_url_products:,} overlapping products, including {exact_nutrition_products:,} with an exact match to a Part 1 nutrition-image URL.",
        "- The matching is calculated once per unique product, then annotation-image multiplicity is reported separately. IDs are not padded, truncated, or numerically reformatted.",
        "",
        "## Inspected data and schema",
        "",
        "### Part 1",
        "",
        f"- Primary processed input: `data/openfoodfacts_eda_enriched.csv` (read with `code` as a string); source: the first 50,000 rows of the English Open Food Facts compressed TSV export (`https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz`). The compact sample also exists, but the enriched CSV is used because it retains image, country, serving, and nutrition fields.",
        f"- Nutrition fields: {', '.join(column for column in part1.columns if column.endswith('_100g') and not column.startswith('image'))}.",
        f"- Image references present: {', '.join(column for column in IMAGE_COLUMNS if column in part1.columns)}. Nutrition-specific references: `image_nutrition_url`, `image_nutrition_small_url`.",
        f"- Part 1 rows with a nutrition-image URL: {metrics['part1_rows_with_nutrition_image_url']:,}.",
        "",
        "### Part 2",
        "",
        f"- Configuration/splits: `default`, {', '.join(f'{name}={count:,}' for name, count in sorted(splits.items()))}.",
        f"- Top-level features observed in repository metadata: {', '.join(top_features)}.",
        "- `meta` contains string `barcode`, string `image_id`, `image_url`, `split`, `ocr_url`, plus review/source flags; row samples show `tokens`, `bboxes`, and `ner_tags` aligned by index. The audit streamed metadata through the dataset-server API and discarded raw OCR arrays after counting their lengths.",
        f"- BIO label families advertised by metadata: {', '.join(label_families)}.",
        "- Annotation interpretation: nutrient-specific values/units and serving-size-related spans are labeled; nutrient names are generally not labeled as entities. This is not complete nutrient-name/value row-relation supervision.",
        "- License in repository metadata: CC-BY-SA 3.0. No image payloads or model weights were downloaded.",
        "",
        "## Identifier and image-alignment quality",
        "",
        f"- Part 1: {metrics['part1_valid_id_rows']:,} valid ID rows, {metrics['part1_unique_valid_products']:,} unique IDs, {metrics['part1_duplicate_valid_id_rows']:,} repeated valid-ID rows beyond the first, {metrics['part1_missing_ids']:,} missing IDs, {metrics['part1_invalid_non_digit_ids']:,} non-digit IDs excluded.",
        f"- Part 2: {metrics['part2_valid_barcode_rows']:,} valid barcode rows, {metrics['part2_unique_valid_products']:,} unique barcodes, {metrics['part2_duplicate_valid_barcode_rows']:,} repeated valid-barcode rows beyond the first, {metrics['part2_missing_barcodes']:,} missing barcodes, {metrics['part2_invalid_non_digit_barcodes']:,} non-digit barcodes excluded.",
        f"- Of the {overlap:,} overlapping products, {metrics['overlapping_part1_products_with_nutrition_image_url']:,} have a Part 1 nutrition-image URL. Exact URL matches among overlapping product/annotation rows: {metrics['overlapping_annotation_rows_with_any_part1_image_url_match']:,} to any Part 1 image URL and {metrics['overlapping_annotation_rows_with_part1_nutrition_url_match']:,} to a nutrition-image URL.",
        "- URL equality is metadata-level evidence only. A shared barcode does not prove that the Part 1 URL points to the exact annotated image; URL version suffixes may differ, and a URL may later serve changed content. No pixel/hash comparison was run because downloading images is outside scope.",
        "- Image-level leakage is possible when one barcode has multiple annotated image rows or when Part 1 and Part 2 refer to different revisions of the same product. Keep all records for an overlapping barcode grouped together in any later split, and deduplicate by barcode plus image version where justified.",
        "- The repository contains one manually reviewed Part 1 image/record (product 241013128), but it has no token boxes/BIO annotations and is not an overlap match. The local annotation scout is a metadata-ranked review queue, not verified labels; no Part 2 image collection or split manifests were present in the inspected project.",
        "",
        "## Recommendation for Part 2",
        "",
        "**Recommended: Option A for the current 50,000-row sample — keep the HF dataset as a separate, image-grounded supervised source. The measured unique-barcode overlap is zero, so there is no product-level join to integrate into this Part 1 sample.** Do not expand or transfer labels using the current CSV. Part 1 remains useful as a separate EDA/reference resource, but its product metadata is not image-grounded test truth.",
        "",
        "- **Option A — independent datasets (recommended now):** train/evaluate on the HF annotations according to their published splits; keep the current Part 1 sample separate. Any future evaluation on Part 1 requires its own image-grounded labels.",
        "- **Option B — controlled integration (conditional future option):** only reconsider if a newly acquired or broader Part 1 extract produces actual barcode matches. Then verify image-version identity, group matched products across splits, and report matched/disjoint performance separately.",
        "- **Option C — weak-label expansion from Part 1 (not recommended):** the current Part 1 sample has no overlapping products, and product-level values are not image-grounded. Do not synthesize HF labels from metadata. Even with future overlap, require image verification, basis/unit reconciliation, and explicit weak-label uncertainty handling.",
        "",
        "## Next steps",
        "",
        "1. Start Part 2 from the HF dataset's existing train/test split, keeping all images for one barcode in one group if new splits are later created; first inspect split-level barcode duplication.",
        "2. If cross-dataset integration is a firm requirement, re-query Open Food Facts using the 3,083 exact annotated barcodes instead of broadening the current 50k sample blindly; audit only a bounded set of exact matches.",
        "3. Preserve the HF annotations and map their BIO nutrient/value classes to the project schema; add relation/table-row supervision only if the downstream task needs nutrient names linked to values.",
        "",
        "## Reproduction and output files",
        "",
        "Run from the project root: `python scripts/analyze_dataset_overlap.py`. The script reads only the existing enriched CSV and public HF dataset-server JSON rows; it does not change source data, EDA artifacts, or the dashboard.",
        "- `outputs/dataset_overlap_summary.csv` — metrics with denominators.",
        "- `outputs/overlapping_products.csv` — one record per overlapping annotation image (product-level multiplicity is explicitly included); if no overlap exists, a header-only CSV is still emitted.",
        "- This report records the HF revision so later reruns can identify repository changes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--api-root", default=DEFAULT_API_ROOT)
    parser.add_argument("--summary", type=Path, default=Path("outputs/dataset_overlap_summary.csv"))
    parser.add_argument("--overlap", type=Path, default=Path("outputs/overlapping_products.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_overlap_analysis.md"))
    args = parser.parse_args()

    if not args.sample.is_file():
        parser.error(f"Part 1 CSV not found: {args.sample}")
    part1 = pd.read_csv(args.sample, dtype={"code": "string"}, keep_default_na=False)
    if "code" not in part1.columns:
        parser.error("The Part 1 CSV does not contain the expected identifier column 'code'.")
    for column in IMAGE_COLUMNS:
        if column not in part1.columns:
            part1[column] = ""

    metadata = get_dataset_metadata(args.dataset)
    splits = get_split_counts(metadata)
    if not splits:
        parser.error("Hugging Face metadata did not provide exact split counts.")
    top_features, labels = feature_summary(metadata)
    annotated: list[dict[str, Any]] = []
    for split, expected_count in sorted(splits.items()):
        annotated.extend(
            fetch_split_rows(
                args.dataset,
                args.config,
                split,
                args.api_root,
                expected_count,
            )
        )

    part1_id_status = [normalize_id(value) for value in part1["code"].tolist()]
    annotation_id_status = [normalize_id(row.get("barcode_raw")) for row in annotated]
    summary = build_summary_rows(
        part1, annotated, part1_id_status, annotation_id_status, splits
    )
    overlap_rows = build_overlap_records(
        part1, annotated, part1_id_status, annotation_id_status
    )
    write_csv(args.summary, summary, ["metric", "value", "denominator"])
    overlap_fields = list(overlap_rows[0]) if overlap_rows else [
        "product_id",
        "part1_rows_for_product",
        "part2_annotation_rows_for_product",
        "part2_split",
        "part2_image_id",
        "part2_image_url",
        "part1_any_image_url_exact_match",
        "part1_nutrition_image_url_exact_match",
        "part1_has_nutrition_image_url",
    ]
    write_csv(args.overlap, overlap_rows, overlap_fields)
    write_report(
        args.report,
        summary,
        part1,
        annotated,
        metadata,
        splits,
        top_features,
        labels,
        overlap_rows,
    )
    metrics = {row["metric"]: row["value"] for row in summary}
    print(
        json.dumps(
            {
                "part1_rows": len(part1),
                "part2_rows": len(annotated),
                "part2_splits": splits,
                "unique_product_overlap": metrics["unique_product_overlap"],
                "overlapping_annotation_rows": metrics["overlapping_part2_annotation_rows"],
                "exact_any_image_url_products": metrics[
                    "overlapping_products_with_exact_any_image_url_match"
                ],
                "exact_nutrition_image_url_products": metrics[
                    "overlapping_products_with_exact_nutrition_image_url_match"
                ],
                "summary_csv": str(args.summary),
                "overlap_csv": str(args.overlap),
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error