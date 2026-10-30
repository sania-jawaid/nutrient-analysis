# Cleanup review — approved cleanup completed

Reviewed **2026-10-30** against actual files and first-party references. The user
explicitly approved removal of the empty log and seven XML test reports listed
below. **Only those eight files were deleted.** Nothing was staged or committed.
Any further deletion requires separate confirmation. This review is not permission to clear directories.
See the [current workflow](../README.md#current-workflow--documentation-verified-2026-10-30).

## Approved deletions — historical record

The following are execution logs, not application inputs. Exact-name searches
across first-party Python, Markdown, notebook, configuration, JSON and PowerShell
sources (including ignored artifacts) found **no references before this review
was added**. Their only new documentation references are this review's links.
Absence of references does not establish that audit evidence is unwanted.

| Actual path | Bytes | Evidence / reproducibility | Risk and required decision |
|---|---:|---|---|
| outputs/clean_dashboard_test_stderr.log | 0 | Empty stderr capture | User approved; deleted |
| outputs/clean_dashboard_bio_results_tests.xml | 593 | Historical JUnit result | User approved; deleted |
| outputs/clean_dashboard_filters_tests.xml | 553 | Historical JUnit result | User approved; deleted |
| outputs/clean_dashboard_metric_tests.xml | 1,048 | Historical JUnit result | User approved; deleted |
| outputs/clean_dashboard_safety_tests.xml | 1,782 | Historical JUnit result | User approved; deleted |
| outputs/clean_dashboard_sections_tests.xml | 523 | Historical JUnit result | User approved; deleted |
| outputs/clean_dashboard_tests.xml | 2,707 | Historical JUnit result | User approved; deleted |
| outputs/dashboard2_export_tests.xml | 2,932 | Historical export-validation result | User approved; deleted |

**Eight approved files, 10,138 bytes total** (seven XML reports and one empty
log). The user chose not to retain these historical reports. No legacy
generated export was confirmed both unreferenced and redundant, so none is
proposed for deletion. No tests were rerun to recreate historical evidence.

## Explicit keep decisions — corrections to earlier exploration

- **Keep [scripts/eda_extraction_impact.py](../scripts/eda_extraction_impact.py).**
  Actively imported by [impact_dashboard.py](../src/nutrition_extraction/impact_dashboard.py),
  [impact notebook](../part1_eda_impact_on_part2.ipynb),
  [tests](../tests/test_eda_extraction_impact.py) and
  [dashboard validation](../scripts/validate_impact_dashboard.py). It is not unused
  just because generated findings already exist.
- **Keep [scripts/export_demo_dashboard.py](../scripts/export_demo_dashboard.py)**,
  its [HTML template](../src/nutrition_extraction/demo_dashboard.html),
  [original export](../outputs/presentation/nutrition_dashboard_demo.html) and
  [tests](../tests/test_demo_export.py). Original [app.py](../app.py) still exposes
  that HTML download. The **6,818,537-byte** original HTML and **27,375,505-byte**
  [Dashboard2 HTML](../main_output/index.html) are distinct
  seven-/eight-section products, not proven duplicates. Compatibility removal
  would need a separate user-approved code/UI change.
- **Keep [data/openfoodfacts_sample.csv](../data/openfoodfacts_sample.csv)** and
  its provenance sidecar. References remain in the original app, [EDA notebook](../eda_openfoodfacts.ipynb),
  [integration notebook](../part1_part2_dataset_integration.ipynb), acquisition
  defaults and [integration support](../scripts/dataset_integration_support.py).
  Being older than the enriched sample is not evidence that it is unnecessary.
- **Keep [scripts/check_full_off_barcode_overlap.py](../scripts/check_full_off_barcode_overlap.py)**
  as a reproducibility utility. A standalone audit script need not be imported
  to be useful; no obsolete-equivalence evidence was established.
- **Keep all source, tests, notebooks, report templates/findings/dictionaries,
  generated EDA/integration/readiness artifacts, both exports, original datasets,
  annotation files, image/OCR/prediction caches and local model assets.** They
  support reproducibility or cached operation; cleanup must not trigger weight
  downloads, destroy gold, discard the paused run's progress or force re-inference.
  Model/cache/data directories are not cleanup candidates.

## Documentation scope and checks

- Inventory: **17 existing first-party Markdown files updated**, plus this new
  review = **18 documentation files**. Tool-owned pytest-cache documentation,
  environments, Git internals, weights, models/dependencies and package metadata
  are not first-party narrative docs and were excluded.
- Historical/generated reports received scope banners and current-workflow
  links; original measured findings and historical test totals remain intact.
  Regeneration can overwrite generated banners; reapply context on a future
  deliberate report refresh. No generation scripts were changed here.
- Source signatures and package declarations were read statically; no Python,
  dependency install, model, inference, export, notebook or dashboard server was
  executed for these documentation changes. Validation is documentation/link
  and protected-file checking, not an assertion that the whole test suite passed.
- Local-link validation checked **147 Markdown links** across the 18 documents:
  **zero missing targets and zero unresolved Markdown heading anchors** at the
  time of this review. Non-Markdown file size/mtime comparison against the
  pre-edit baseline found **zero changes**; SHA-256 checks for both apps and
  the package declaration were unchanged. External web availability and UI/model
  execution were not part of this check.
- The saved full-cohort summary is `paused_model_load_error`, **441 completed /
  3,075**, **1 failed**, **2,633 pending** (195 test / 246 train), timestamp
  2026-10-30T15:01:33.063231+00:00. Do not describe it as an active run.
- Original [app.py](../app.py) and the [cache-only app (now app_final.py)](../app_final.py) were not altered during that review.
  The only subsequent non-Markdown changes were the eight explicitly approved deletions above; no Git-index changes were made.
- Outside this project, the parent [exercise.md](../../exercise.md) and
  [Pune_Derby_Case_Study.md](../../Pune_Derby_Case_Study.md) are preserved original
  assignment/case-study material, not rewritten implementation documentation.
  The parent [exercise_summary_action_items.md](../../exercise_summary_action_items.md)
  is also unchanged. Any assignment-versus-implementation reconciliation is left
  to the parent/user; this review does not certify those external documents as current.