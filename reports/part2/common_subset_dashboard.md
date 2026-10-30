# Common-subset dashboard integration

> **Historical original-app integration revision — context added 2026-10-30.**
> Navigation and validation counts below describe that revision of the protected
> original app (8501), not today's clean navigation or a freshly verified test
> total. [Dashboard2](../../app_clean_README.md), port 8502, is cache-only with
> eight sections, View Raw Data first, and fixed evaluation denominators.
> The 3,075 matched rows come from the full **4,881,991-row** source scan, not
> the 50K EDA sample; 3,083 original annotation images include eight unmatched.
> See the [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> Historical evidence below is preserved; no integration/dashboard code changed.

## Read-only sources

The existing broad 50,000-row enriched product sample remains the default. Neither original sample CSV, notebook, integration artifact nor official annotation is modified by the dashboard.

The additional source uses outputs/part1_part2_integration/integrated_products.parquet, selecting only `barcode_matched_image_unverified` rows. At the measured snapshot this is 3,075 products. Unmatched placeholders are excluded from common-subset EDA, but all 3,083 official images remain in the linked image explorer and model evaluation is not filtered by matches.

Product metadata is adapted to the existing Part 1 EDA schema: barcode → code (string), brand → brands, category → categories, country → countries, nutrition_image_url → image_nutrition_url. Original basis-qualified nutrient keys/raw values are expanded from structured_nutrients for EDA only. Existing charts and source-specific country/category filters work for both datasets.

## Navigation

- Data source → Project sample → Sample: choose the broad enriched sample, original compact sample, or Part 1 + Part 2 common product subset.
- Part 1 + Part 2: Common product subset: full-snapshot overlap explanation, actual saved counts and denominators, split/match audit, matched metadata/nutrient coverage, descriptive comparison against broad 50K, product/image/BIO explorer, unmatched products, provenance, downloads.
- Part 2 Dataset EDA: new linked-context explorer for official train or test images; original EDA artifacts remain shown unchanged.
- BIO Entity Tagging: selected evaluation image has a linked product nutrition context expander. It performs an exact string lookup after inference only. No product nutrients are passed into the model, converted to gold labels, or used to select the official evaluation set.

## Interpretation

The common subset is selected by annotation availability. Its distributions must not replace the broad EDA or be claimed representative of the complete OFF database. The existing broad sample is also a first-50K non-random extract. Source export dates may differ; cross-subset charts are descriptive comparisons, not causal inference.

Barcode agreement is not packaging/pixel identity. Product nutrients are contextual metadata rather than image-verified truth. Original image IDs/URLs, match status and official split are displayed explicitly. Remote images are opt-in; token previews use predicate-filtered Parquet reads and preserve original gold BIO labels.

## Validation

67 regression tests passed, including string barcode preservation, unmatched exclusion from common EDA, complete official-split image retention and gold token previews. The Streamlit UI check exercised broad default → common source → nutrient EDA → common audit → Part 2 EDA → cached BIO evaluation → broad source restoration, without running OCR/model inference or rescanning the complete export. Browser inspection confirmed the common audit renders without Streamlit exceptions.
