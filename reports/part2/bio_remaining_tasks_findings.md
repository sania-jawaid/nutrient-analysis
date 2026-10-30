# Completion of Part 2 tasks 5–8

> **Historical follow-up review — context added 2026-10-30.** All numerical,
> timing, browser and test statements below are evidence from the described
> follow-up, **not a current-suite test total or a new run**. Seeded-20
> annotation-derived fields, one-image manual-reference scores and weak API
> comparisons have different denominators and remain separate. Earlier
> [OCR-only review](../part2_ocr_validation/review_summary.md) used a different
> run/scoring convention; do not overwrite or pool either result. The
> [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30)
> documents the paused, partial full cohort. Dashboard actions below describe
> the original app (8501), not [cache-only Dashboard2](../../app_clean_README.md)
> on 8502. No model, OCR, tests or notebook cells were run for this doc update.

## Provenance

The 20-image ONNX benchmark was **not rerun** for this follow-up. Its saved predictions were re-analysed for numerical accuracy and parser/error categories, with the input bundle SHA-256 recorded in [the completion report](bio_remaining_tasks_review.json). The original BIO/time/memory results remain historical measurements.

An additional **real EasyOCR → local ONNX run** was completed on the project's existing manually reviewed image 241013128. This is the same one-image fixture already documented in the OCR validation notebook, not a new held-out benchmark. Manual annotations and transcription were not changed. Its results and SHA-256 hashes of the image and both reference files are saved in [the manual-reference review](bio_manual_reference_review.json).

## Task 5 — field and numerical accuracy

The evaluator counts exact nutrient/basis/unit/value/qualifier tuples, preserving duplicate occurrences. Compatible mass units are normalized; kJ and kcal remain distinct declarations. Exact value accuracy has its own denominator: predictions and references sharing a nutrient/basis/unit key, paired exactly first and then in source order. No nearest-value matching or tolerance is used. Missing fields reduce recall separately. Empty denominators are reported as unavailable, not invented scores.

### Re-analysis of the saved 20-image subset

| Measure | Result |
|---|---:|
| Exact field matches | 174 |
| Predicted parseable fields | 179 |
| Parseable gold fields | 187 |
| Field precision / recall / F1 | 0.9721 / 0.9305 / 0.9508 |
| Conditional exact numerical accuracy | 174/178 = 97.75% |
| Uncertain gold spans excluded | 24 |
| Uncertain predicted spans | 38 |

These references are decoded official BIO annotations on supplied corpus OCR, not independent visual transcription.

### Actual existing manual-image evaluation

| Method on the same EasyOCR detections | Exact matches | Predictions | Gold fields | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ONNX model + conservative JSON parser | 5 | 5 | 16 | 1.0000 | 0.3125 | 0.4762 |
| Existing spatial-rule extractor | 1 | 2 | 16 | 0.5000 | 0.0625 | 0.1111 |

- Model numerical accuracy on comparable fields: **5/5**; **11 missing fields** and **14 uncertain predicted spans** prevent interpreting this as perfect panel extraction.
- The five exact fields were 237 kcal/100g, 213 kcal/serving, 893 kJ/serving, 0.36 g salt/100g and 21.7 g sugars/100g.
- **84/84** EasyOCR words were covered. No token-level BIO metric was calculated because the manual reference has no aligned BIO tags.
- Raw-text CER: **0.1009**; raw whitespace-word WER: **0.4615**. These include punctuation and reading-order differences on one image only. Case-folded token diffs are diagnostic displays, not the inputs to these raw metrics.
- OCR and model cache hits were both **false** for this run; measured stage times were approximately 1.72 s OCR and 2.75 s processor/model. Future cached reports retain their original timing flags.

## Task 6 — actual error examples

### Confirmed text differences on the existing manual transcription

- The image/manual reference has `992 kJ`; EasyOCR returned `992k/`. The parser flagged unexplained unit text rather than guessing kJ.
- `4,2 g` was returned as `4,2 9`, and `4,6 g` as `4,69`. These illustrate gram-glyph confusion and explain missing-unit warnings.
- The serving text containing `4,1 g - Sel 0,32 g` was merged as `4,19-Sel0,32g`, leaving a multi-number predicted span uncertain.

The saved manual report contains all reference/OCR differences, predicted raw spans and missing-field rows. Text diffs can also reflect punctuation or reading order; they are not claimed to be spatial word alignment.

### Saved official-test prediction diagnostics

- Actual BIO disagreements: **22 tokens** — 15 nutrient/basis-label disagreements, six false-positive tokens, one BIO-boundary mismatch.
- Boundary example: barcode 4099100042832, `1g` began a new added-sugars serving span where gold continued one.
- Duplicate example: barcode 3760027470152 had conflicting carbohydrate predictions `9,77 g` and `57,74 g`, plus conflicting salt quantities `1,80 g` and `120 g`. Duplicates remain uncertain rather than being silently deduplicated.
- Parser reasons: eight missing-unit flags, nine no-number flags, eight conflicting-duplicate flags, eight unexplained-letter flags, seven multi-number flags. A span may have several reasons; the total reason count is not the span count.
- Unit diagnostics in corpus text are **suspicions**, not independent proof of OCR errors. The manual-image examples above provide the separate verified reference.
- Truncation history: the original three images had 391/453 covered words. The prior real overlapping-window regression achieved **453/453**; the 20-image saved subset has **1,551/1,551**, with no uncovered position silently tagged O. Per-image coverage/window diagnostics are displayed.

## Task 7 — separate product API integration

The first 10 sampled products already had actual Open Food Facts API responses cached. This follow-up reused those 10 caches and recomputed compatible field comparisons; it did not claim 10 fresh network fetches. Barcodes remain strings, preserving leading zeros. Each comparison includes compatible-reference support, exact agreement and metadata-only tuple PRF. Missing/failed references are not invented as gold.

API nutrition is mutable product metadata, not visual gold. Its scores and missing-model-field counts remain separate from both official BIO and manual-image evaluation. The dashboard exposes per-product rows and JSON download, including cache flags.

## Task 8 — completed dashboard review

The existing BIO Entity Tagging view now includes:

- Original fine-grained/coarse BIO metrics and visualizations.
- Micro field precision/recall/F1 and conditional numerical accuracy with explicit denominators and parsing exclusions.
- Actual BIO error-category chart, duplicate/unit/parser evidence, and per-image source coverage.
- Downloadable per-image JSON and combined field/error evaluation.
- Separate 10-product API comparisons and download.
- Provenance-validated saved manual-image review, OCR differences, model-versus-rule comparison and download; optional explicitly requested rerun.
- Existing uploaded-image inference and JSON download.

The dashboard stays on port 8501. Cached evidence is visible without loading another ONNX session. Manual reports are rejected if their image/reference hashes or model/schema change. Part 1 charts, source datasets and both existing notebooks are preserved; only new review cells were appended to the BIO notebook.

## Validation and remaining limitations

- **58 project tests passed**, including denominator, duplicate, qualifier, basis/unit, empty-reference, memory-gate and provenance checks. Editor diagnostics reported no errors in changed Python files.
- Browser verification confirmed the numerical metric, saved 5/16 manual-image warning, API results and all new report downloads with no Streamlit exception.
- One manual panel is insufficient for a general OCR/extraction accuracy claim. More independent reviewed images, verified word boxes and adjudicated references would be needed. No training, quantization or checkpoint download was performed.