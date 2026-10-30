# Dashboard layout and filter revision

> **Historical original-app UI revision — context added 2026-10-30.** The
> eleven-page navigation and 71-test/browser statements below are records of
> that revision, not the clean dashboard's navigation or a new suite result.
> Original [app.py](../../app.py), port 8501, was not altered. Current
> [Dashboard2](../../app_final_README.md), port 8502, is eight-section,
> raw-data-first and cache-only, with separate basic/detailed task explanations
> and micro-score interpretation. The current [full-cohort summary](../../outputs/full_cohort_bio/summary.json)
> is paused/partial, not evidence of an active worker. See the
> [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> Original metrics, gold and numerical validation history below are preserved.

## Navigation

Data Overview → Data Understanding → Part 1 + Part 2: Common product subset → Results → Nutrients → Data quality → Images & alignment → Annotation scout → Part 2 Dataset EDA → BIO Entity Tagging → View Raw Data.

Data Understanding contains the requested three-dataset description table, with explicit provenance that the current product source is the official TSV export and nutrition-table-detection is optional/not loaded. The common-subset page also shows the descriptions, retained summary-card layout and archived Part 2 overview picture in a fixed-snapshot expander. It does not duplicate unrelated Part 1 summary cards above the Part 2 EDA page.

## Filters

Country and category filters are OR within a field, AND across fields. They select product rows, not individual labels; co-occurring country/category labels may still appear on bars for selected products. All available labels are selectable, no top-100 restriction. Modes and numeric statistics use filtered, unclipped numeric values.

Common-subset coverage/comparison charts use filtered common products and a separately filtered broad sample. Part 2 exploratory charts are recomputed by selecting original annotation rows via linked product metadata, preserving official split labels. BIO support, image shape and token-length histograms now use the filtered inventory instead of precomputed unfiltered aggregates. Product/error record tables gain View image URLs without row fan-out or arbitrary image selection. Static parsed-value aggregates without product keys are hidden when filters are active.

Fixed full-inventory snapshot counters, the archived dataset picture, source-schema column chart and official evaluation results are explicitly labelled unfiltered audits. They are intentionally not recalculated as if they were selected-product EDA. Unmatched products remain in official evaluation; product metadata is never inserted as gold labels or model features.

## Tables and metrics

- Nutrients: Mode card with frequency; all tied modes available; all-unique data is labelled no unique mode. Full nutrient statistics now also include mode, mode frequency and mode count.
- Part 2 EDA: product/image table with View image column; same-page actual entity-tagging input table with OCR words, original boxes, gold label IDs/BIO labels and image links.
- View Raw Data: existing source table plus a separate image-grain product+image left join, with search, filters, row cap and CSV download. Gold tokens remain in their normalized separate table.
- Results: saved official-test scores and evaluated product/image links, without loading model weights.
- RAM/RSS telemetry is removed from dashboard metrics/captions. Existing loading and continuation safety gates remain active; detailed resource messages are replaced with a concise safety pause. Historical JSON artifacts and notebooks retain their original recorded measurements.

## Validation

71 regression tests passed, including country+Snacks intersection, missing-metadata nonmatches, modes/ties, image-link disambiguation, no product-row multiplication and split-preserving filtered BIO counts. Interactive checks passed each page under France + Snacks (31 products in the actual common subset), including linked gold support, mode card, merged raw table and unchanged cached official-test coverage. Browser checks confirmed renamed navigation and original entity-tagging input table without Streamlit exceptions. No existing notebook, product CSV, gold annotation or evaluation source was edited.
