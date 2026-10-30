# Part 1 ↔ Part 2 Dataset Overlap Analysis

> **Historical 50K-sample audit — context added 2026-10-30.** The zero overlap
> and recommendations below apply to the **50,000-row enriched sample** at the
> recorded time; all original counts remain intact. Later [complete-source integration](../outputs/part1_part2_integration/data_dictionary.md)
> scanned **4,881,991 rows** and linked **3,075** of the **3,083** annotated
> images/products, with eight unmatched. That does not change this sample's
> measured zero overlap or establish packaging identity. Earlier statements
> about missing collections and future-only integration are historical, not
> current instructions. See the [current workflow](../README.md#current-workflow--documentation-verified-2026-10-30).
> Reproduction below can access public metadata and overwrite generated reports;
> it was not run during this documentation update.

Generated (UTC): 2026-10-28T13:46:52+00:00

## Executive summary

- Part 1 uses the existing enriched Open Food Facts CSV: 50,000 rows and 44 columns. No EDA input, EDA outputs, or dashboard files were changed.
- Part 2 is [openfoodfacts/nutrient-detection-layout](https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout), revision `7d3245ca596f0c8f41dc740ca0fa5987b1e0fd4d`, last modified 2024-12-23T18:06:29.000Z; repository metadata reports 3,083 examples (2,884 train and 199 test).
- Unique product overlap by trimmed, digit-only string ID: **0 products** (0.00% of Part 2 unique valid products; 0.00% of Part 1 unique valid products). This is an identifier match, not proof of identical image pixels or label correctness.
- Those products correspond to 0 Part 2 annotation rows. Exact image URL string matches occur for 0 overlapping products, including 0 with an exact match to a Part 1 nutrition-image URL.
- The matching is calculated once per unique product, then annotation-image multiplicity is reported separately. IDs are not padded, truncated, or numerically reformatted.

## Inspected data and schema

### Part 1

- Primary processed input: `data/openfoodfacts_eda_enriched.csv` (read with `code` as a string); source: the first 50,000 rows of the English Open Food Facts compressed TSV export ([source URL](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz)). The compact sample also exists, but the enriched CSV is used because it retains image, country, serving, and nutrition fields.
- Nutrition fields: energy-kj_100g, energy-kcal_100g, energy_100g, fat_100g, saturated-fat_100g, trans-fat_100g, cholesterol_100g, carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g, sodium_100g, vitamin-a_100g, vitamin-d_100g, vitamin-e_100g, vitamin-k_100g, vitamin-c_100g, vitamin-b1_100g, vitamin-b2_100g, vitamin-pp_100g, vitamin-b6_100g, vitamin-b9_100g, vitamin-b12_100g, potassium_100g, calcium_100g, iron_100g.
- Image references present: image_url, image_small_url, image_ingredients_url, image_ingredients_small_url, image_nutrition_url, image_nutrition_small_url. Nutrition-specific references: `image_nutrition_url`, `image_nutrition_small_url`.
- Part 1 rows with a nutrition-image URL: 8,803.

### Part 2

- Configuration/splits: `default`, test=199, train=2,884.
- Top-level features observed in repository metadata: ner_tags, tokens, bboxes, image, meta.
- `meta` contains string `barcode`, string `image_id`, `image_url`, `split`, `ocr_url`, plus review/source flags; row samples show `tokens`, `bboxes`, and `ner_tags` aligned by index. The audit streamed metadata through the dataset-server API and discarded raw OCR arrays after counting their lengths.
- BIO label families advertised by metadata: ADDED_SUGARS, CALCIUM, CARBOHYDRATES, CHOLESTEROL, ENERGY_KCAL, ENERGY_KJ, FAT, FIBER, IRON, POTASSIUM, PROTEINS, SALT, SATURATED_FAT, SERVING, SODIUM, SUGARS, TRANS_FAT, VITAMIN_D.
- Annotation interpretation: nutrient-specific values/units and serving-size-related spans are labeled; nutrient names are generally not labeled as entities. This is not complete nutrient-name/value row-relation supervision.
- License in repository metadata: CC-BY-SA 3.0. No image payloads or model weights were downloaded.

## Identifier and image-alignment quality

- Part 1: 50,000 valid ID rows, 50,000 unique IDs, 0 repeated valid-ID rows beyond the first, 0 missing IDs, 0 non-digit IDs excluded.
- Part 2: 3,083 valid barcode rows, 3,083 unique barcodes, 0 repeated valid-barcode rows beyond the first, 0 missing barcodes, 0 non-digit barcodes excluded.
- Of the 0 overlapping products, 0 have a Part 1 nutrition-image URL. Exact URL matches among overlapping product/annotation rows: 0 to any Part 1 image URL and 0 to a nutrition-image URL.
- URL equality is metadata-level evidence only. A shared barcode does not prove that the Part 1 URL points to the exact annotated image; URL version suffixes may differ, and a URL may later serve changed content. No pixel/hash comparison was run because downloading images is outside scope.
- Image-level leakage is possible when one barcode has multiple annotated image rows or when Part 1 and Part 2 refer to different revisions of the same product. Keep all records for an overlapping barcode grouped together in any later split, and deduplicate by barcode plus image version where justified.
- The repository contains one manually reviewed Part 1 image/record (product 241013128), but it has no token boxes/BIO annotations and is not an overlap match. The local annotation scout is a metadata-ranked review queue, not verified labels; no Part 2 image collection or split manifests were present in the inspected project.

## Recommendation for Part 2

**Recommended: Option A for the current 50,000-row sample — keep the HF dataset as a separate, image-grounded supervised source. The measured unique-barcode overlap is zero, so there is no product-level join to integrate into this Part 1 sample.** Do not expand or transfer labels using the current CSV. Part 1 remains useful as a separate EDA/reference resource, but its product metadata is not image-grounded test truth.

- **Option A — independent datasets (recommended now):** train/evaluate on the HF annotations according to their published splits; keep the current Part 1 sample separate. Any future evaluation on Part 1 requires its own image-grounded labels.
- **Option B — controlled integration (conditional future option):** only reconsider if a newly acquired or broader Part 1 extract produces actual barcode matches. Then verify image-version identity, group matched products across splits, and report matched/disjoint performance separately.
- **Option C — weak-label expansion from Part 1 (not recommended):** the current Part 1 sample has no overlapping products, and product-level values are not image-grounded. Do not synthesize HF labels from metadata. Even with future overlap, require image verification, basis/unit reconciliation, and explicit weak-label uncertainty handling.

## Next steps

1. Start Part 2 from the HF dataset's existing train/test split, keeping all images for one barcode in one group if new splits are later created; first inspect split-level barcode duplication.
2. If cross-dataset integration is a firm requirement, re-query Open Food Facts using the 3,083 exact annotated barcodes instead of broadening the current 50k sample blindly; audit only a bounded set of exact matches.
3. Preserve the HF annotations and map their BIO nutrient/value classes to the project schema; add relation/table-row supervision only if the downstream task needs nutrient names linked to values.

## Reproduction and output files

Run from the project root: `python scripts/analyze_dataset_overlap.py`. The script reads only the existing enriched CSV and public HF dataset-server JSON rows; it does not change source data, EDA artifacts, or the dashboard.
- `outputs/dataset_overlap_summary.csv` — metrics with denominators.
- `outputs/overlapping_products.csv` — one record per overlapping annotation image (product-level multiplicity is explicitly included); if no overlap exists, a header-only CSV is still emitted.
- This report records the HF revision so later reruns can identify repository changes.
