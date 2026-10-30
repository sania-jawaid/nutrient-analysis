# Final nutrition dashboard

## Cached image comparison (2026-10-31)

The **BIO Entity Tagging** page now shows **Bounding boxes · dataset gold vs
cached predictions**, with two images side by side. Both use the same supplied
OCR coordinates, converted from the original 0–1000 frame. The model predicts
labels, not box locations. Non-O labels are boxed: green means an exact BIO-label
match; amber means a mismatch. Small numbers link boxes to zero-based token
indices in the retained Advanced table. Sample label counts include O and do
not replace the existing span metrics. Images retain their original orientation.

Gold is the original official-dataset annotation, not independently checked
package truth. Uploads do not acquire gold labels from this feature. No OCR,
inference, model download or training is performed.

Set `NUTRITION_DATA_DIR` to the existing local data directory when data lives
outside this project. The image resolver first tries the saved basename under
that directory's image cache, then a saved existing path only within the project
or explicitly configured data directory. Saved image SHA-256 values are checked
when present. Missing or hash-mismatched images produce a notice, not a download.

The [standalone HTML](main_output/index.html) embeds the same PNG comparisons.
The [exporter](scripts/export_final_dashboard.py) supports `--refresh-bio` to
refresh only existing BIO examples, preserving frozen charts, summaries,
downloads, FINAL branding, overlap explanation and the removed sidebar timestamp.
The [overlay tests](tests/test_bio_visualization.py) cover geometry, nonmutation,
cache path/hash restrictions, embedded images and the cached-only Streamlit UI.

> **Current guide — 2026-10-30.** See the [project workflow](README.md#current-workflow--documentation-verified-2026-10-30)
> for dataset denominators, optional processing and historical reports. This
> documentation-only update did not alter the original app, this app or any notebook.

The separate entry point is [app_final.py](app_final.py). The original
[app.py](app.py) is neither imported nor changed. Existing source modules,
notebooks, configuration, models, annotation tables and worker outputs remain read-only.

## Launch ownership

The parent/user launches this entry point with Streamlit on **port 8502**.
No server is started by implementation or tests. Use an existing environment
containing Streamlit, Plotly, pandas, NumPy and PyArrow; validation used the
workspace's existing `.venv-2` environment. No dependency installation was needed.

## Navigation

Launch from the project root using an existing compatible environment:

```powershell
python -m streamlit run app_final.py --server.port 8502
```

The original [app.py](app.py) uses port **8501** and remains a separate interface.
Required dashboard runtimes include pandas, NumPy, Streamlit, Plotly, Pillow and
PyArrow. The `dashboard` extra in [pyproject.toml](pyproject.toml) does not declare
PyArrow; see [setup](README.md#setup). No environment or package was changed here.

1. View Raw Data (initial page)
2. Data Overview
3. Data Understanding
4. Nutrients
5. Data Quality
6. Part 2 Dataset Info
7. BIO Entity Tagging
8. Results

## Scope and evidence

- **50K Dataset:** the local enriched product CSV; barcodes stay strings.
- **Part1 + Part2 Overlapped Dataset:** existing `products_for_eda` transforms
  matched product metadata from the complete Part 1 snapshot, not from the 50K sample.
  The current artifacts contain 3,075 matched product/image records. Product and
  image granularity remain separate even if those counts happen to coincide.
- Country/category labels use the verified existing `filter_products` API:
  OR within each dimension, AND across dimensions. Dataset changes clear stale filters.
- Every exploratory page uses the shared product filters. Part 2 defaults to
  the **full 3,083-image annotation inventory**, including unmatched images;
  any active filter instead intersects exact barcode strings with filtered products.
- Original per-image annotation rows retain `bounding_box`, `gold_bio_id` and
  `gold_bio_label`; predictions never replace or repair the reference tags.
- Cached predictions are selected from **Full matched cohort** or **Seeded 20 test images**.
  Nothing runs OCR, loads models, performs inference, downloads images or trains.
  Only locally present original images are rendered; source image links open externally
  only when clicked. Missing images do not trigger downloads.
- Full-cohort status is a **published checkpoint**, not proof the worker is alive.
  Completed/pending/failed images and the fixed target are explicit. Refresh the page
  to read a newer snapshot; no timer interferes with selection or worker state.
- Results use saved **per-split micro metrics**, not mean per-image scores.
  EDA filters and product dataset selection do not alter evaluation membership,
  processed-image support or target denominator. Unprocessed splits show blanks.
- Reference truth is original dataset BIO annotation, not independent image
  transcription or current package truth. Train performance is descriptive,
  not held-out generalization. Supplied OCR means BIO metrics are not end-to-end OCR metrics.
- Missingness, duplicate counts and numeric anomaly flags are review signals only.
  Source units and nutritional bases are retained, with no imputation or conversions.

## Reading the latest charts and results

- Nutrients explains histogram counts, box-plot spread and violin density;
  statistics retain finite extremes and zeros, with no imputation/clipping.
  Mode lists ties and distinguishes all-unique data. Correlation is pairwise
  Pearson with at least three usable pairs and initially six fields, not cause.
- The fixed 50K examples are independent of active dataset/filters. Source
  energy_100g has median 1,350 and maximum 680,000; carbohydrate median is 25.5 g
  with 7,857 values above 100. Energy-field correlation is about 0.8307 on
  47,748 pairs. Missing names (261), any finite nutrient coverage (47,775/50,000
  across 27 fields), and vitamin-D missingness (42,242) are sample-specific.
- **How the Methods Were Compared** always uses the same saved seeded-20
  official-test bundle. **Basic Nutrient Detection** fairly compares rules and
  model under shared VALUE/BASIS/O spans. `SERVING_SIZE` projects to BASIS;
  nutrient values project to VALUE. A literal per100g phrase is not automatically
  annotated gold BASIS. **Detailed Nutrient Identification** is model-only and
  shown separately rather than compared to an incompatible rule taxonomy.
- **Understanding the Results** explains exact spans, TP/FP/FN, precision,
  recall, F1, token accuracy and micro aggregation. O-heavy token accuracy does
  not establish good entity extraction. Full-cohort means more images for the
  same model, not another method. Dataset/filters do not change scored membership.

### Published full-cohort checkpoint

The [saved summary](outputs/full_cohort_bio/summary.json), timestamp
2026-10-30T15:01:33.063231+00:00, records **paused_model_load_error**, not an active
worker: **441/3,075 completed**, **195 test / 246 train**, **1 failed** and
**2,633 pending**. The target is 199 test / 2,876 train. These are partial scores,
not a completed full-cohort benchmark. Refreshing this app reads files only;
resuming inference is a separate explicit [workflow](README.md#full-cohort-checkpoint-and-safe-resume).

### Offline Dashboard2

Open [the eight-section HTML](main_output/index.html) or read
its [capability guide](outputs/presentation/dashboard2_README.md). The
[exporter](scripts/export_final_dashboard.py) calls this app's render functions
and embeds plots/results/downloads/thumbnails. It does **not** export live
country/category EDA filters, uploads, inference or status refresh. The full
correlation matrix is embedded rather than the live six-field default. The
[earlier seven-section demo](outputs/presentation/README.md) remains separate
and optional; its exporter is not replaced for original-app compatibility.

The `_clean` entry-point, exporter and test filenames were renamed to `_final`.
The viewer snapshot's BIO examples were selectively refreshed on 2026-10-31;
its other frozen data and custom shell were preserved. Its source fingerprint
now identifies the final app used for that refresh. Internal CSS classes and
the offline shell's existing filename are unchanged.

## Validation scope

[tests/test_app_final.py](tests/test_app_final.py) uses Streamlit AppTest without a server:
all sections on both datasets, shared filters, exact barcode scope, raw annotation
preservation, split/image selection for both cache sources, empty results,
fixed evaluation denominators, missing/incompatible caches, coverage/gold mismatch
rejection, review rules, and saved micro-metric count reconstruction. A fixture
checks protected original files' size/mtime and the original app's SHA-256 before/after.

No notebooks or existing configuration files are edited by these additions.