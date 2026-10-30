# Offline presentation demo

> **Original seven-section demo — guide reviewed 2026-10-30.** This remains an
> optional compatibility export linked by the protected original [app](../../app.py)
> on port 8501; it is **not automatically obsolete**. For the newer eight-section,
> raw-data-first clean app export use the separate [Dashboard2 guide](dashboard2_README.md).
> See the [current project workflow](../../README.md#current-workflow--documentation-verified-2026-10-30).
> Neither HTML export was regenerated in this documentation-only update.

Open [nutrition_dashboard_demo.html](nutrition_dashboard_demo.html) in a modern desktop browser. It is a single self-contained file: copy it to another computer, email it, or carry it on a USB drive. No Python, Streamlit server, internet connection or model weights are required. The existing file is **6,818,537 bytes**; it is not a byte-for-byte duplicate of Dashboard2.

## Presentation controls
- Use the sidebar or Previous / Next buttons to move between seven sections.
- Use left/right arrow keys to navigate, Presentation mode to hide the sidebar, and Esc to restore it.
- Hover, zoom and export chart PNGs using the Plotly toolbar.
- Search the embedded product inventory and select saved BIO examples.
- Print / save PDF uses the browser's print dialog. HTML is recommended for interactive presentations.
- The live Streamlit sidebar offers **Download offline presentation demo**.

## Snapshot, not live inference
The export records completed/failed/pending counts and its own export timestamp;
these do **not** establish that a worker is alive. The newer saved
[checkpoint](../full_cohort_bio/summary.json), timestamp
2026-10-30T15:01:33.063231+00:00, is **paused_model_load_error**, with **441/3,075
completed, 1 failed, 2,633 pending** (195 test / 246 train completed). The older
HTML has its own earlier snapshot and does not inherit these newer values.
The HTML never runs inference/OCR, uploads files, fetches images or refreshes
from the worker. Train and official-test scores remain separate; seeded-20 is
separately labelled. Six bounded saved examples are embedded, not all images
or every prediction JSON. It does not reproduce live EDA country/category filters.

To refresh this particular snapshot, use [scripts/export_demo_dashboard.py](../../scripts/export_demo_dashboard.py)
with pandas, PyArrow, Pillow and Plotly available in the chosen environment:

```powershell
python scripts/export_demo_dashboard.py
```

Run from the project root. It reads saved artifacts only; it does not start or
interrupt inference. Regeneration replaces this HTML snapshot, not original
dataset/model files. The distinct [Dashboard2 exporter](../../scripts/export_clean_dashboard.py)
does not replace this script's original-app download contract. No export was run here.

## Attribution
Source: Open Food Facts and its contributors. Product database: ODbL / Database Contents License; annotated layout dataset: CC-BY-SA 3.0; pretrained nutrition-extractor model: CC-BY-NC-SA-4.0. Embedded packaging images retain their original contributor attribution and applicable image licenses. Preserve attribution when sharing the demo.