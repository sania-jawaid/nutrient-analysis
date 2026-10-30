# Dashboard2 — offline HTML

> **Artifact companion guide — updated 2026-10-31.** See the [current workflow](../../README.md#current-workflow--documentation-verified-2026-10-30)
> and [final dashboard guide](../../app_final_README.md). This guide describes the
> final Dashboard 2 app and the saved HTML snapshot. The separate original app is
> kept intact. In the current workspace, the final app runs on port 8503; the
> offline HTML is [index.html](index.html). Re-exporting HTML does not automatically
> regenerate this Markdown companion or notebook outputs.

Open [index.html](index.html) directly in a current Edge,
Chrome or Firefox browser. No Python, server or internet is needed to view it.
Plotly JavaScript, plot data, saved results, thumbnails and downloads are embedded.
A Content Security Policy explicitly blocks network connections and remote assets.

The HTML embeds Plotly, charts and saved results in one file. It includes all
nutrient/chart-type variants and cached image examples; only the selected plots
are displayed at a time. The exact file size can change whenever it is regenerated.

## Source fidelity

The exporter is [scripts/export_final_dashboard.py](../../scripts/export_final_dashboard.py),
with [the offline shell](../../src/nutrition_extraction/clean_dashboard_offline.html).
It **calls the actual render functions** from
[app_final.py](../../app_final.py), using a small output adapter instead of a
Streamlit server or AppTest tree. This preserves current explanations, table
labels, Plotly layouts and colors. Unsupported output calls fail rather than
silently dropping content. Source apps, notebooks, models, annotation files and
the inference worker are not modified. No model module is imported.

The earlier seven-section presentation is a different export and is not reused.
It remains [optional original-app compatibility content](README.md), not an
unused duplicate merely because Dashboard2 exists.

The exact eight sections are:

1. View Raw Data (initial page)
2. Data Overview
3. Data Understanding
4. Nutrients
5. Data Quality
6. Part 2 Dataset Info
7. BIO Entity Tagging
8. Results

## Supported interactions

- Switch between **50K Dataset** and **Part1 + Part2 Overlapped Dataset**;
  both are unfiltered. Raw previews retain the source's 200-row cap, while
  embedded product downloads contain all 50,000 or 3,075 rows respectively.
- Every available nutrient has **Histogram**, **Box plot** and **Violin plot**.
  These retain every finite observation, zero and extreme; there is no sampling,
  clipping, unit conversion or imputation. Plotly hover, zoom, pan, reset,
  selection and PNG export are available as appropriate for the plot type.
- Country/category/coverage charts, missingness chart and the original numeric
  flag tables retain their source colors and definitions. The background stays
  black, including Plotly paper and plotting areas.
- Correlation uses the original Pearson/minimum-three-pairs calculation,
  RdBu_r scale, cell labels and large adaptive height. **All available fields**
  are shown initially, intentionally extending the live dashboard's default
  six-field selection. Correlation field-subsetting is not implemented.
- Original annotation selector contains representative original train/test
  records, retaining all original words, boxes, numeric label IDs and gold tags
  for each selected image. Inventory counts still cover all 3,083 images.
- Switch cached results between **Full matched cohort** and **Seeded 20 test
  images**. All 20 saved seeded images are browsable. Full-cohort examples are
  bounded to the first three saved records per original split. Split/image
  selectors, expandable token gold/prediction tables, saved structured fields,
  JSON downloads and original-image thumbnails work offline.
- In each cached BIO image example, **Dataset gold · original annotations** and
  **Cached model predictions** show the same original image side by side. Boxes
  use the supplied OCR word locations; they are not predicted object locations.
  Green indicates an exact BIO-label match; amber indicates a label mismatch.
  The concise panel captions remain, while the long overlay explanation is
  omitted from the HTML. The live app retains its fuller comparison explanation.
  These labels are dataset references, not independently checked visual truth.
- Results preserve train/test micro scores, fixed targets, method comparisons,
  partial-progress warnings, failure details and the saved per-image results
  table. Dataset switching never changes the scores.
- **How the Methods Were Compared** retains the separate **Basic Nutrient
  Detection** rules/model tables on the same saved 20 test images and shared
  VALUE/BASIS/O projection, plus **Detailed Nutrient Identification** model-only
  scores. The basic-task explanation now uses practical food-label examples:
  nutrient amounts such as 12 g are VALUE; dataset-annotated serving amounts
  such as 30 g are BASIS. The serving-size entity maps to BASIS, not every
  literal “per 100 g” phrase. **Understanding the Results** explains exact boundaries/types,
  precision/recall/F1, O-inclusive token accuracy, micro counts and coverage.
  Results-source switching does not change this fixed seeded comparison.
- Thumbnails are resized copies of cached source images; original SHA-256 and
  thumbnail SHA-256 are shown. No URL is fetched and originals are unchanged.
- CSV/JSON downloads are losslessly gzip-compressed inside the HTML and use the
  browser's `DecompressionStream`. Use a current browser; older browsers can
  view the dashboard but may not support these download buttons.

## Snapshot limits

This is not a running Streamlit application. No country/category product
filtering, upload, inference, OCR, training, external image links or live refresh
is simulated. The timestamp and saved run status describe the captured files,
not proof that any worker is running. Re-export to include later checkpoints.
Full-cohort annotation/extraction examples are **partial representative browsing
content**, not a random quality sample or an all-image gallery.

The generated snapshot currently records **441 / 3,075** completed full-cohort
images, **2,633 pending**, **1 failed**, saved status **paused_model_load_error**.
The completed images are **195 test** and **246 train**. Seeded-20 results remain
a separate saved official-test subset. The embedded timestamp and counts are
saved in the artifact, although the visible export timestamp was removed. The
counts are a frozen export snapshot, not a live display.
The currently read [source summary](../full_cohort_bio/summary.json) has timestamp
**2026-10-30T15:01:33.063231+00:00** and the same paused counts; neither that file
nor the embedded timestamp is evidence of an active worker. The original target
is **199 test + 2,876 train**. HTML cannot refresh itself from later source files.
Of the target, test has **195 scored, 3 pending and 1 failed**; train has
**246 scored, 2,630 pending and 0 failed**. The failed test attempt stopped at
the RAM safety check: **0.71 GiB** was available, below the **2.5 GiB** model-load
minimum. The run is paused, not active. Completed results remain saved; pending
images have not been scored and are not treated as incorrect.

Gold labels are supplied dataset annotations, not independent visual truth.
Train scores do not establish held-out performance. Cached BIO scores measure
labeling on supplied OCR, not end-to-end reading or verified package nutrition.
Pending and failed images are not scored as zero. Original units/bases and
reference tags are never repaired. See the in-dashboard explanations.

## Regeneration and validation

Use the existing dashboard Python environment with pandas, NumPy, Streamlit,
Plotly, PyArrow, Pillow and **Markdown** (the added lightweight Markdown renderer).
Run the exporter as a normal Python script. It writes [main_output/index.html](index.html)
and prints the exact size, chart count and checkpoint totals. Streamlit may print harmless
“No runtime found” warnings when its decorators are imported; no server is run.

From the project root, after selecting an existing compatible environment:

```powershell
python scripts/export_final_dashboard.py
```

The `dashboard` extra in [pyproject.toml](../../pyproject.toml) supplies Streamlit
and Plotly, but does **not** declare PyArrow or Markdown. Verify those separately;
see [setup](../../README.md#setup). No package installation/export was run here.
Regenerate only deliberately: embedded data/plots/results are a saved snapshot,
not live EDA filters or a full Streamlit implementation.

Focused checks are in
[tests/test_export_final_dashboard.py](../../tests/test_export_final_dashboard.py).
They cover strict JSON/script escaping, lossless compressed downloads, adapter
state restoration, exact section order, two dataset denominators, all nutrient
plot types/colors, full correlation, original annotation schemas, split metrics,
inline thumbnails/hashes, no external asset references and Node JavaScript
syntax validation without launching a browser or server. Browser interaction
validation is a separate parent-owned step.
This documentation review did not rerun these tests or browser checks and makes
no newly verified test-total claim. Existing recorded validation remains historical.

## Attribution

Open Food Facts product database: ODbL / Database Contents License. Dataset:
openfoodfacts/nutrient-detection-layout, CC-BY-SA 3.0. Original product image
contributors and their applicable image licenses still apply (typically
CC-BY-SA). Pretrained model: openfoodfacts/nutrition-extractor,
CC-BY-NC-SA-4.0. This research snapshot is not medical or nutrition advice.