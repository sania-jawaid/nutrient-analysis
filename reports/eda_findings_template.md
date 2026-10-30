# EDA Findings — Open Food Facts Nutrition-Image Extraction

> **Report template populated with the 2026-10-28 EDA snapshot — context added
> 2026-10-30.** Counts, provisional splits, no-OCR/no-checkpoint statements and
> the remaining-work list below describe that Part 1 run, not today's project
> state. Later Part 2 uses original official annotations, local pretrained ONNX
> inference and actual one-image EasyOCR checks; see the [current workflow](../README.md#current-workflow--documentation-verified-2026-10-30),
> [BIO measurements](part2/bio_pipeline_findings.md) and [manual review](part2/bio_remaining_tasks_findings.md).
> Preserve historical measurements; refresh/re-date the report explicitly after
> a deliberate EDA rerun, not by substituting full-cohort counts. No notebook,
> sample, report tables or numerical evidence was regenerated here.

> **Scope:** Refreshed 50,000-row sample from the first rows of the Open Food Facts English compressed TSV export, acquired on 2026-10-28. This is a bounded, non-random sample, not a population estimate. The public file can change; rerun the notebooks before presenting these figures. Structured product nutrition fields are metadata—not image-grounded labels by default.

## Dataset and scope

- **Source:** `https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz` (gzip-compressed TSV).
- **Sample:** First 50,000 records. The current source schema has 211 columns; this enriched sample retains 44 fields for product identity/text, country and serving context, image URLs, and selected nutrition attributes.
- **Why these fields:** Bound memory and focus on fields needed to identify products, assess image/metadata candidate coverage, analyze country/serving context, and profile common nutrition targets. This is task-driven selection, not statistical feature selection.
- **Still omitted:** 167 source fields, including many specialist nutrient components, contributor/provenance information, tags/translations, packaging/origin/allergen fields, derived scores, and other metadata. The complete categorized list is in `reports/eda/excluded_source_columns.csv`. The older 47-field sample remains at `data/openfoodfacts_sample.csv`; the enriched sample is `data/openfoodfacts_eda_enriched.csv`.

## Data quality and coverage

| Check | Result | Interpretation |
|---|---:|---|
| Rows / selected columns | 50,000 / 44 | First-row bounded sample. |
| Unique product codes | 50,000 | No duplicate `code` values observed. |
| Duplicate code rows | 0 | Does not detect near-duplicate products or images. |
| Exact duplicate rows | 0 | Checked across the 44 retained fields. |
| Blank product names | 261 (0.522%) | Blank or missing name. |
| Rows with at least one selected structured nutrition value | 47,775 (95.55%) | Does not imply complete nutrition labels. |
| Rows with any image URL | 13,694 (27.388%) | URL presence only; content may be inaccessible or wrong view. |
| Rows with nutrition-image URL | 8,803 (17.606%) | Nutrition image reference only. |
| Image URL + at least one selected target | 12,771 | Metadata-based review candidates, not verified alignment. |
| Nutrition image URL + at least one target | 8,527 | Higher-priority review candidates, still not automatically gold. |
| Visually verified panel/value matches | 1 | Product `241013128` inspected; values agreed with the structured per-100-g fields. One case is insufficient for a benchmark. |
| OCR-ready token annotations with boxes/BIO labels | 0 | The reviewed sample has record-level transcription only; no word-level boxes/tags yet. |

### Annotation scouting

- The scout ranked the 8,527 nutrition-image/target candidates by target completeness plus availability of product name, brand, country, and serving-size context. The score is a review-order heuristic, not confidence or verified alignment.
- 8,294 candidates have at least 7 of the 9 target fields; 6,550 have all 9. These counts indicate structured metadata completeness only.
- The nutrition-image URL filename hints at 24 locales; the most common hints are `en` (7,426), `fr` (907), and `es` (101). URL-derived locale is not guaranteed to match the actual image language.
- A 100-record pilot queue was selected to include multiple URL-locale hints, then prioritized by completeness. The full top-ranked queue is capped to 500 records for review. Review candidates are not counted as usable annotated training examples until manually checked and labeled.
- Scout artifacts: `reports/annotation_scout/summary.json`, `annotation_review_queue.csv`, and `annotation_pilot_batch.csv`.

### Common target coverage

| Field | Non-missing | Coverage |
|---|---:|---:|
| Energy (kcal/100g) | 47,748 | 95.496% |
| Fat (g/100g) | 47,704 | 95.408% |
| Carbohydrates (g/100g) | 47,687 | 95.374% |
| Proteins (g/100g) | 47,685 | 95.370% |
| Salt (g/100g) | 47,293 | 94.586% |
| Sodium (g/100g field convention) | 47,293 | 94.586% |
| Sugars (g/100g) | 45,681 | 91.362% |
| Saturated fat (g/100g) | 42,428 | 84.856% |
| Fiber (g/100g) | 39,535 | 79.070% |

Full coverage table: `reports/eda/enriched_nutrition_coverage.csv`.

### Locale and serving context

- `countries` is non-empty on 42,799 records (85.598%). Top comma-split label occurrences: United States 41,212; France 958; World 473; United Kingdom 211; États-Unis 109; Royaume-Uni 101; Estados Unidos 54. These are occurrences of labels, not normalized country counts; multilingual duplicates and the first-row sample bias remain.
- `serving_size` is non-empty on 47,600 records (95.2%); `quantity` is non-empty on 32,493 (64.986%). Example serving-size formats include `1g`, `1.61 ONZ (45 g)`, and `40g`.
- The current export/sample did not provide `serving_quantity` or `nutrition_data_per`, so those fields cannot be quantified from this run. The visually inspected French panel showed separate per-100-g and 90-g serving columns, illustrating why basis must be captured from the image itself.

### Numeric data-quality flags

- One negative protein value and one negative fiber value were observed. Tiny negative floating-point artifacts may also occur in micro-nutrient fields.
- Extreme maxima include fat 1,000,000; carbohydrates 40,000; sugars 40,000; energy 47,000 kcal/100g; protein 1,430; and fiber 900. Treat these as likely unit/scale/entry inconsistencies for review, not automatic deletion rules.
- Distributions are strongly right-skewed; report medians/quantiles and investigate extremes before training. Do not silently clip.
- A macro subtotal check over fat/carbohydrates/protein (excluding fiber) exceeds 100 for 11,909 rows; a separate plot-cell check that also included fiber flagged 16,461. These are exploratory flags only: fields/units may be inconsistent or overlapping and require manual review before calling them impossible records.

## Image quality and image-text alignment

- A previous bounded audit of 10 nutrition-image URLs decoded all 10. Dimensions were 192–400 px wide and 74–400 px high, with aspect ratios 0.48–5.41. This small, non-random audit cannot estimate dataset-wide readability, OCR quality, blur, or corruption.
- Product `241013128` (Eclairs) image was visually inspected at 400×253 px. The nutrition table was legible and displayed per-100-g and per-90-g serving values. The per-100-g values matched its product metadata. This is one manually verified example, not a general alignment result.
- Record-level manual transcription is saved in `data/evaluation/gold_ocr.txt`; normalized values in `data/evaluation/gold_records.jsonl`; image in `data/manual_samples/241013128_nutrition.jpg`; review note in `reports/eda/manual_verification_log.csv`.
- The transcription was made by visual inspection; **no OCR engine produced it**. It is not word-box/BIO annotation and must not be used as LayoutLMv3 training data.

## Candidate selection and splits

- Candidate: image URL plus at least one selected nutrition value. Strict candidate: nutrition-image URL plus at least one target. Candidate counts above must not be presented as confirmed training-pair counts.
- Human review should reject/flag inaccessible images, wrong image views, blurred/occluded tables, product mismatch, unclear language, ambiguous serving basis, and missing target fields.
- A provisional metadata-candidate 80/10/10 split by product `code` has 10,216/1,277/1,278 entries and no product-code overlap. It is exploratory only. No final supervised split exists because verified word-level annotations are not yet available.
- After annotation, group by product/SKU, stratify where feasible by language, country, nutrient presence, layout and image quality, freeze test data, and optionally report unseen-brand/country stress tests.

## OCR, entity tagging, and LayoutLMv3 status

- OCR backend adapters and a lexical/spatial entity-tagging baseline are implemented. The unit parser recognizes numeric values, units, and common nutrient aliases; spatial row pairing remains heuristic and needs human review for complex/multi-column labels.
- Synthetic smoke test passes for Energy 200 kcal and Fat 8 g; this is a code-path check, **not an accuracy result**.
- At the time of this run, neither the Tesseract executable nor PaddleOCR Python package was available, so OCR was not executed on the verified panel. Install/configure one OCR backend before generating predictions.
- LayoutLMv3 training/inference code and PyTorch Dataset are implemented, but no model weights were downloaded and no model was trained. Training requires many manually verified word/box/BIO annotations, not just one record-level gold transcription. CPU-only fine-tuning may be slow.

## Evaluation and next steps

No real OCR or extraction metric is reported yet. The code supports CER, WER, exact-span entity precision/recall/F1, field/record exact match, numeric MAE, unit/basis correctness and latency. Meaningful metrics require predicted outputs against manually verified references; use the same frozen product-group test set for OCR rules and any LayoutLMv3 experiment. Define unit conversions and numeric tolerances before evaluation.

**Remaining work:**
1. Install/configure Tesseract or PaddleOCR and run the verified panel; compare OCR output to `gold_ocr.txt`.
2. Annotate word-level boxes and BIO tags for that panel, then create a substantially larger, double-reviewed set across layouts/languages/quality levels.
3. Generate final group-aware splits from annotated data; the existing splits are exploratory only.
4. Run OCR baseline inference, record error cases, then optionally approve/download LayoutLMv3 weights and train on CPU only if feasible.
5. Report aggregate test metrics only after sufficient independent gold records exist.

## Artifacts

- EDA notebook: `eda_openfoodfacts.ipynb`
- OCR/model workflow: `ocr_extraction_workflow.ipynb`
- Enriched dataset: `data/openfoodfacts_eda_enriched.csv`
- Enriched summary: `reports/eda/enriched_summary.json`
- Country counts: `reports/eda/country_counts.csv`
- Manual review candidates: `reports/eda/enriched_manual_review_candidates.csv`
- One-image verification log: `reports/eda/manual_verification_log.csv`
- Excluded source columns: `reports/eda/excluded_source_columns.csv`
