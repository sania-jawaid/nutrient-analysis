"""Interactive Streamlit dashboard for Open Food Facts EDA findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nutrition_extraction.eda import NUTRITION_PATTERN, _nonempty, analyze_export, nutrition_statistics  # noqa: E402
from nutrition_extraction.linked_data import load_linked, products_for_eda, linked_product, annotation_preview, MATCHED
from nutrition_extraction.dashboard_data import filter_products, merged_image_products, with_image_links, annotation_rows, label_support, numeric_modes

LINKED_DIR = ROOT / "outputs" / "part1_part2_integration"
COMMON_SECTION = "Part 1 + Part 2: Common product subset"
SECTION_NAMES = ["Data Overview", "Data Understanding", COMMON_SECTION, "Results", "Pre-Inference EDA & Readiness", "Post-Inference Diagnostics", "Nutrients", "Data quality", "Images & alignment", "Annotation scout", "Part 2 Dataset EDA", "BIO Entity Tagging", "View Raw Data"]
DATASET_DESCRIPTIONS = """
| Dataset | What it contains | Best use |
|---|---|---|
| [product-database](https://huggingface.co/datasets/openfoodfacts/product-database) | Product records, metadata, structured nutritional values and image references | **Part 1: EDA** |
| [nutrient-detection-layout](https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout) | Nutrition images, OCR tokens, bounding boxes and BIO nutrient labels | **Part 2: Nutrient extraction** |
| [nutrition-table-detection](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection) | Images with annotations identifying nutrition-table regions | **Optional: Detect/crop the nutrition table** |
"""

st.set_page_config(
    page_title="Open Food Facts | Nutrition EDA",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

demo_export = ROOT / "outputs/presentation/nutrition_dashboard_demo.html"
if demo_export.is_file():
    st.sidebar.download_button(
        "Download offline presentation demo", demo_export.read_bytes(),
        file_name="nutrition_dashboard_demo.html", mime="text/html", key="offline_demo_download",
    )
    st.sidebar.caption("Standalone HTML snapshot; works offline. Live inference continues separately.")

PALETTE = {
    "ink": "#182230",
    "muted": "#65758B",
    "blue": "#276EF1",
    "teal": "#00A890",
    "orange": "#F59E0B",
    "red": "#E5484D",
    "purple": "#7950F2",
    "panel": "#F5F8FC",
}
TARGET_FIELDS = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g",
]

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Arial', sans-serif; }
        html, body, [data-testid="stApp"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background-color: #000000 !important; }
      .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
        div[data-testid="stHorizontalBlock"]:has(.metric-card) { gap: 1rem !important; }
    h1, h2, h3 { font-family: 'Arial', sans-serif; color: #E8EEF5; letter-spacing: -0.035em; }
      .hero { padding: 1.35rem 1.55rem; border-radius: 22px; background: linear-gradient(120deg,#102A43 0%,#174A74 58%,#197A86 100%); color: white; margin-bottom: 1.1rem; }
      .hero h1 { color: white; font-size: 2rem; margin: 0 0 .35rem 0; }
      .hero p { color: #D8E8F4; margin: 0; font-size: .98rem; }
      .eyebrow { text-transform: uppercase; letter-spacing: .13em; color: #8FE3D0; font-size: .72rem; font-weight: 700; margin-bottom: .45rem; }
            .metric-card { background: #171F2B; border: 1px solid #293648; border-radius: 16px; padding: .85rem 1rem; min-height: 92px; box-shadow: 0 5px 16px rgba(0,0,0,.14); }
            .metric-card { margin-bottom: .5rem; }
            .metric-label { font-size: .78rem; color: #B7C4D3; font-weight: 600; }
            .metric-value { font-family: 'Arial',sans-serif; color: #F7FAFF; font-size: clamp(1.1rem, 1.8vw, 1.55rem); font-weight: 800; line-height: 1.3; margin-top: .25rem; white-space: nowrap; letter-spacing: -.025em; }
            .metric-note { color: #A0AEC0; font-size: .72rem; margin-top: .14rem; }
            .section-kicker { color: #6CA8FF; text-transform: uppercase; font-size: .72rem; letter-spacing: .11em; font-weight: 700; margin: .35rem 0 .25rem; }
            .info-strip { padding: .8rem 1rem; border-radius: 12px; background: #152B43; border-left: 4px solid #4F8EF7; color: #DCEBFA; font-size: .88rem; }
            .warning-strip { padding: .8rem 1rem; border-radius: 12px; background: #332916; border-left: 4px solid #F2B84B; color: #F4E6C7; font-size: .88rem; }
        section[data-testid="stSidebar"] { background: #000000 !important; color: #E8EEF5 !important; min-width: 21rem !important; width: 21rem !important; }
        section[data-testid="stSidebar"] > div { width: 21rem !important; background: #000000 !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label *,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #C5D0DC !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] input,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: #202B38 !important; color: #F0F5FA !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span { background-color: #202B38 !important; color: #F0F5FA !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] *,
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #F0F5FA !important; }
    div[data-baseweb="popover"] { background: #202B38 !important; }
    div[data-baseweb="popover"] * { color: #F0F5FA !important; }
    section[data-testid="stSidebar"] button { background: #202B38 !important; color: #F0F5FA !important; border: 1px solid #45566B !important; }
      div[data-testid="stTabs"] button { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str = "") -> None:
    """Render a compact dashboard KPI card."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def friendly_name(column: str) -> str:
    """Return a readable name for OFF nutrient field identifiers."""
    label = column.replace("_100g", " / 100 g").replace("_100ml", " / 100 ml")
    return label.replace("-", " ").replace("_", " ").title()


def safe_fraction(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


@st.cache_data(show_spinner="Loading product sample…")
def load_local_csv(path_text: str, modified_time: float) -> pd.DataFrame:
    del modified_time  # included in cache key to invalidate when the file changes
    return pd.read_csv(path_text, dtype={"code": "string"}, low_memory=False)


@st.cache_data(show_spinner="Reading linked product/image metadata…")
def cached_linked_data(fingerprint: tuple):
    del fingerprint
    return load_linked(LINKED_DIR)


def get_linked_data():
    names = ["integrated_products.parquet", "integrated_images.parquet", "provenance.json", "integrated_annotations.parquet"]
    fingerprint = tuple((name, (LINKED_DIR / name).stat().st_mtime_ns, (LINKED_DIR / name).stat().st_size) for name in names)
    return cached_linked_data(fingerprint)


@st.cache_data(show_spinner="Reading filtered gold annotation rows…")
def cached_annotation_rows(image_keys: tuple, modified_ns: int):
    del modified_ns
    return annotation_rows(LINKED_DIR, image_keys)


def product_table(table, *, images=None, **options):
    """Every product/image table exposes clickable source references when known."""
    if images is None:
        try:
            _, images, _ = get_linked_data()
        except (OSError, ValueError, ImportError):
            images = None
    view = with_image_links(table, images) if images is not None else table
    config = {name: st.column_config.LinkColumn(name.replace("_", " ").title(), display_text="View image")
              for name in view if name in {"image_url", "nutrition_image_url", "image_nutrition_url", "dataset_image_url"}}
    st.dataframe(view, width="stretch", hide_index=True, column_config=config, **options)


def resource_safe_message(error):
    """Resource checks remain active; detailed memory telemetry is not a UI result."""
    text = str(error)
    return "Inference paused by the local resource safety check. Close other workloads and retry; no unsafe model load was attempted." if any(word in text.casefold() for word in ["ram", "gib", "rss", "memory"]) else text


def show_linked_product_context(barcode: str, *, key: str) -> None:
    """Show metadata only; never inject product nutrients into model/gold inputs."""
    try:
        products, images, _ = get_linked_data()
    except (OSError, ValueError, ImportError) as error:
        st.info(f"Linked product metadata unavailable: {error}")
        return
    product = linked_product(products, barcode)
    if product is None or product["match_status"] != MATCHED:
        st.info("This annotated barcode is not matched to the complete product snapshot. Its image/annotations remain available.")
        return
    st.caption("Linked product snapshot — metadata context only, not image-verified nutrition gold or model input.")
    product_table(pd.DataFrame([{name: product.get(name) for name in ["barcode", "product_name", "brand", "category", "country", "serving_size", "nutrition_image_url", "match_status"]}]), images=images)
    nutrients = json.loads(product["structured_nutrients"])
    st.dataframe(pd.DataFrame([{"source nutrient field": name, "product value (raw)": value} for name, value in nutrients.items()]), width="stretch", hide_index=True)
    related = images.loc[images["barcode"].eq(str(barcode))]
    product_table(related[["image_id", "dataset_split", "image_url", "image_reference_status"]], images=images)
    st.download_button("Download linked product metadata", json.dumps({"product": product, "warning": "Barcode match only; product values may differ from annotated packaging."}, ensure_ascii=False, indent=2, default=str), file_name=f"linked_product_{barcode}.json", mime="application/json", key=key)


@st.cache_data(show_spinner="Reading uploaded CSV…")
def load_uploaded_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO

    suffix = Path(filename).suffix.lower()
    sep = "\t" if suffix == ".tsv" else None
    if sep:
        return pd.read_csv(BytesIO(file_bytes), sep=sep, dtype={"code": "string"}, low_memory=False)
    return pd.read_csv(BytesIO(file_bytes), sep=None, engine="python", dtype={"code": "string"})


@st.cache_data(show_spinner="Computing dashboard statistics…")
def compute_dashboard_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute reusable EDA summaries for a selected subset; cache by dataframe content."""
    summary, missingness, coverage = analyze_export(frame)
    nutrition_cols = [column for column in frame.columns if NUTRITION_PATTERN.search(str(column))]
    numeric_nutrients = frame[nutrition_cols].apply(pd.to_numeric, errors="coerce") if nutrition_cols else pd.DataFrame(index=frame.index)
    nutrition_stats = nutrition_statistics(frame, nutrition_cols) if nutrition_cols else pd.DataFrame()
    image_cols = [column for column in frame.columns if "image" in str(column).lower() and str(column).lower().endswith("_url")]
    nutrition_image_cols = [column for column in image_cols if "nutrition" in str(column).lower()]
    any_image_mask = pd.concat([_nonempty(frame[column]) for column in image_cols], axis=1).any(axis=1) if image_cols else pd.Series(False, index=frame.index)
    nutrition_image_mask = pd.concat([_nonempty(frame[column]) for column in nutrition_image_cols], axis=1).any(axis=1) if nutrition_image_cols else pd.Series(False, index=frame.index)
    selected_targets = [column for column in TARGET_FIELDS if column in frame]
    target_mask = pd.concat([pd.to_numeric(frame[column], errors="coerce").notna() for column in selected_targets], axis=1).any(axis=1) if selected_targets else pd.Series(False, index=frame.index)
    candidate_mask = nutrition_image_mask & target_mask
    code_col = "code" if "code" in frame else None
    return {
        "summary": summary,
        "missingness": missingness,
        "coverage": coverage,
        "nutrition_cols": nutrition_cols,
        "numeric_nutrients": numeric_nutrients,
        "nutrition_stats": nutrition_stats,
        "image_cols": image_cols,
        "nutrition_image_cols": nutrition_image_cols,
        "any_image_mask": any_image_mask,
        "nutrition_image_mask": nutrition_image_mask,
        "selected_targets": selected_targets,
        "target_mask": target_mask,
        "candidate_mask": candidate_mask,
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_code_rows": int(frame[code_col].duplicated(keep=False).sum()) if code_col else None,
        "unique_codes": int(frame[code_col].nunique(dropna=True)) if code_col else None,
        "blank_product_names": int((~_nonempty(frame["product_name"])).sum()) if "product_name" in frame else None,
    }


def country_counts(frame: pd.DataFrame, column: str = "countries") -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="int64")
    values = frame[column].dropna().astype("string").str.split(",").explode().str.strip()
    values = values[~values.str.casefold().isin({"", "undefined", "nan", "none", "null", "n/a", "na"})]
    return values.value_counts()


def _normalized_labels(value: Any) -> set[str]:
    """Split comma-delimited OFF labels consistently for sidebar filtering."""
    if value is None or pd.isna(value):
        return set()
    return {part.strip().casefold() for part in str(value).split(",") if part.strip()}


def filtered_frame(frame: pd.DataFrame, selected_countries: list[str], selected_categories: list[str]) -> pd.DataFrame:
    """Apply OR within each label filter and AND between country/category groups."""
    return filter_products(frame, selected_countries, selected_categories)


def plot_layout(height: int = 360) -> dict[str, Any]:
    return {
        "height": height,
        "margin": {"l": 12, "r": 12, "t": 38, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Arial, sans-serif", "color": "#E8EEF5"},
    }


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Open Food Facts · Research dashboard</div>
      <h1>Nutrition image data, at a glance.</h1>
      <p>Explore coverage, data quality, nutrient distributions, image availability, and annotation readiness.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Data source and filters ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Dashboard controls")
    section_names = SECTION_NAMES
    active_section = st.radio("Navigate dashboard", section_names, index=0)
    st.divider()
    enriched_path = ROOT / "data" / "openfoodfacts_eda_enriched.csv"
    original_path = ROOT / "data" / "openfoodfacts_sample.csv"
    common_count_label = "matched products"
    common_provenance_path = LINKED_DIR / "provenance.json"
    if common_provenance_path.exists():
        try:
            measured_count = json.loads(common_provenance_path.read_text(encoding="utf-8"))["summary"]["overlapping_barcodes"]
            common_count_label = f"{int(measured_count):,} matched products"
        except (ValueError, KeyError, TypeError):
            pass  # The validated loader below reports invalid provenance.
    local_options = {
        "Part 1 Dataset · Open Food Facts Products export (50,000-row enriched sample)": enriched_path,
        "Part 1 Dataset · Open Food Facts Products export (original compact sample)": original_path,
        f"Part 1 + Part 2 · Common product subset ({common_count_label})": LINKED_DIR / "integrated_products.parquet",
    }
    available_options = {label: path for label, path in local_options.items() if path.exists()}
    source_mode = st.radio("Data source", ["Project sample", "Upload CSV / TSV"], horizontal=False)
    uploaded = None
    if source_mode == "Project sample":
        if not available_options:
            st.error("No local sample found under data/.")
            st.stop()
        selected_label = st.selectbox("Sample", list(available_options), index=0)
        selected_path = available_options[selected_label]
        if selected_path.suffix == ".parquet":
            try:
                linked_products, _, _ = get_linked_data()
                frame_all = products_for_eda(linked_products)
            except (OSError, ValueError, ImportError) as error:
                st.error(f"Cannot load linked common subset: {error}")
                st.stop()
            st.warning("Annotation-selected common subset; not representative of the full product database. The broad 50K EDA remains available.")
        else:
            frame_all = load_local_csv(str(selected_path), selected_path.stat().st_mtime)
        st.caption(f"Loaded: {selected_path.relative_to(ROOT)}")
    else:
        uploaded = st.file_uploader("Upload an OFF CSV or TSV export", type=["csv", "tsv"])
        if uploaded is None:
            st.info("Choose a CSV/TSV file to display the dashboard.")
            st.stop()
        frame_all = load_uploaded_csv(uploaded.getvalue(), uploaded.name)
        st.caption(f"Uploaded: {uploaded.name}")

    st.divider()
    st.markdown("### Filters")
    countries_available = country_counts(frame_all)
    country_choices = countries_available.index.tolist()
    filter_scope = selected_label if source_mode == "Project sample" else "uploaded"
    selected_countries = st.multiselect("Country labels", country_choices, key=f"off_country_filter_{filter_scope}", help="Country strings are not normalized; multilingual variants may be separate. Matching ignores case and whitespace around comma-separated labels.") if country_choices else []
    category_series = frame_all.get("categories", pd.Series(dtype="string")).dropna().astype("string").str.split(",").explode().str.strip()
    category_series = category_series[~category_series.str.casefold().isin({"", "undefined", "nan", "none", "null", "n/a", "na"})]
    category_choices = category_series.value_counts().index.tolist()
    selected_categories = st.multiselect("Category labels", category_choices, key=f"off_category_filter_{filter_scope}", help="Category matching ignores case and whitespace around comma-separated labels.") if category_choices else []
    filter_preview = filtered_frame(frame_all, selected_countries, selected_categories)
    st.caption(f"Filtered products: {len(filter_preview):,} / {len(frame_all):,} · {len(selected_countries)} countries · {len(selected_categories)} categories selected. Country/category filters apply to exploratory charts and linked tables. Official model evaluation and saved full-dataset audit counts remain fixed and are labelled separately.")

frame = filter_preview
if frame.empty and active_section in {"Data Overview", "Nutrients", "Data quality", "Images & alignment", "Annotation scout"}:
    st.warning("No records match the selected filters. Clear one or more filters.")
    st.stop()

stats = compute_dashboard_stats(frame) if not frame.empty else compute_dashboard_stats(frame_all)
summary = stats["summary"]
missingness = stats["missingness"]
coverage = stats["coverage"]
nutrition_cols = stats["nutrition_cols"]
numeric_nutrients = stats["numeric_nutrients"]
nutrition_stats = stats["nutrition_stats"]
image_cols = stats["image_cols"]
nutrition_image_cols = stats["nutrition_image_cols"]
any_image_mask = stats["any_image_mask"]
nutrition_image_mask = stats["nutrition_image_mask"]
selected_targets = stats["selected_targets"]
target_mask = stats["target_mask"]
candidate_mask = stats["candidate_mask"]
verified_path = ROOT / "data" / "evaluation" / "gold_records.jsonl"
verified_n = sum(1 for line in verified_path.read_text(encoding="utf-8").splitlines() if line.strip()) if verified_path.exists() else 0

# ── Summary ribbon ─────────────────────────────────────────────────────────
source_label = f"Uploaded dataset · {uploaded.name}" if uploaded else selected_label
if active_section not in {"Part 2 Dataset EDA", "BIO Entity Tagging", "Results", "Post-Inference Diagnostics", "Pre-Inference EDA & Readiness"}:
    st.markdown(f'<div class="section-kicker">{source_label}</div>', unsafe_allow_html=True)
    metrics_top = st.columns(3)
    with metrics_top[0]: metric_card("Products in view", f"{len(frame):,}", f"of {len(frame_all):,} loaded rows")
    with metrics_top[1]: metric_card("Nutrition fields", str(len(nutrition_cols)), "detected from column names")
    with metrics_top[2]: metric_card("Any image URL", f"{int(any_image_mask.sum()):,}", f"{safe_fraction(any_image_mask):.1%} of filtered rows")
    metrics_bottom = st.columns(2)
    with metrics_bottom[0]: metric_card("Nutrition-image URL", f"{int(nutrition_image_mask.sum()):,}", f"{safe_fraction(nutrition_image_mask):.1%} of filtered rows")
    with metrics_bottom[1]: metric_card("Verified gold records", str(verified_n), "record-level; not token BIO labels")
    st.markdown("<div class='warning-strip'>Image URL + nutrition metadata identifies review candidates only. It does not prove that the linked image contains those exact values. One manually reviewed example is not a model benchmark.</div>", unsafe_allow_html=True)

# ── Overview ───────────────────────────────────────────────────────────────
if active_section == "Pre-Inference EDA & Readiness":
    from nutrition_extraction.readiness_dashboard import render as render_readiness

    render_readiness(ROOT)

if active_section == "Post-Inference Diagnostics":
    from nutrition_extraction.impact_dashboard import render as render_impact_insights

    render_impact_insights(ROOT)

if active_section == COMMON_SECTION:
    st.subheader(COMMON_SECTION)
    try:
        common_products, common_images, common_provenance = get_linked_data()
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Integration artifacts are unavailable or invalid: {error}. Run the separate dataset-integration notebook; the dashboard never substitutes the 50K sample for full-source overlap.")
    else:
        overlap = common_provenance["summary"]
        common_frame = products_for_eda(common_products)
        common_frame = filtered_frame(common_frame, selected_countries, selected_categories)
        common_codes = set(common_frame["code"])
        visible_common_images = common_images.loc[common_images["barcode"].isin(common_codes)]
        st.caption(f"Common-subset exploratory charts/tables: {len(common_frame):,} products after the country/category filters. Full overlap audit counts below are fixed snapshot measurements.")
        st.markdown(DATASET_DESCRIPTIONS)
        part2_snapshot_path = ROOT / "reports/part2_eda/part2_eda_summary.json"
        if part2_snapshot_path.exists():
            snapshot = json.loads(part2_snapshot_path.read_text(encoding="utf-8"))
            st.subheader("Part 2 dataset snapshot · full official inventory")
            snapshot_cols = st.columns(4)
            for col, title, value in zip(snapshot_cols, ["Annotation examples", "Train examples", "Test examples", "BIO labels"], [snapshot.get("total_rows", 0), snapshot.get("split_counts", {}).get("train", 0), snapshot.get("split_counts", {}).get("test", 0), snapshot.get("label_count", 0)]):
                with col: metric_card(title, f"{value:,}", "full inventory; not filtered")
            snapshot_cols = st.columns(4)
            for col, title, value in zip(snapshot_cols, ["Unique barcodes", "Train/test barcode overlap", "Sequence alignment failures", "Review flags"], [snapshot.get("unique_valid_barcodes", 0), snapshot.get("barcode_leakage_across_train_test", 0), snapshot.get("quality_summary", {}).get("sequence_alignment_failures", 0), snapshot.get("suspicious_rows_flagged", 0)]):
                with col: metric_card(title, f"{value:,}", "full inventory; not filtered")
            st.caption("Dataset summary layout retained here from the Part 1 + Part 2 view. Fixed inventory counters are not filter-dependent chart results.")
        overview_picture = ROOT / "reports/part2_eda/part2_eda_overview.png"
        if overview_picture.exists():
            with st.expander("Original dataset picture · fixed full-inventory snapshot"):
                st.image(str(overview_picture), caption="Archived full Part 2 dataset picture, intentionally unfiltered. Current exploratory charts use the filters separately.", width="stretch")
        st.warning("This is an additional annotation-selected analysis, not a replacement for Part 1. Annotation availability determines membership, so its distributions are not representative of the broader product database.")
        cols = st.columns(4)
        cols[0].metric("Complete Part 1 rows scanned", f"{overlap['part1_product_rows']:,}")
        cols[1].metric("Annotated product barcodes", f"{overlap['part2_unique_barcodes']:,}")
        cols[2].metric("Common products", f"{overlap['overlapping_barcodes']:,}", f"{overlap['overlap_percent_of_part2']:.2f}% of Part 2")
        cols[3].metric("Unmatched products retained", f"{overlap['unmatched_barcodes']:,}")
        st.markdown("**How the calculation works**\n\n1. Scan the complete compressed OFF export through EOF in 10,000-row chunks; count exact string barcodes using a disk-backed index.\n2. Inventory every official Part 2 train/test row, preserving image IDs, OCR boxes and original gold BIO labels.\n3. Match exact barcodes after trimming outer whitespace only—no numeric conversion, zero-padding or fuzzy matching.\n4. Left-join from Part 2: unmatched images/annotations remain in the linked tables. Products, images and tokens are stored separately to avoid repeated product metadata.")
        st.caption("Saved full-snapshot measurements, not a new scan or live API lookup. The integration audit stays fixed; all exploratory charts below use the country/category filters. Choose the common subset under Data source → Sample to use Data Overview/Nutrients/Data quality on it.")
        split_counts = visible_common_images.groupby(["dataset_split", "match_status"]).size().rename("images").reset_index()
        st.subheader("Official splits and match status")
        st.dataframe(split_counts, width="stretch", hide_index=True)
        st.info("Official train/test splits are preserved. Model evaluation still uses official test annotations, including unmatched test products; neither common-subset membership nor product nutrients changes model inputs, labels or evaluation selection.")
        st.subheader("Common-subset nutrition and metadata coverage")
        common_statistics = compute_dashboard_stats(common_frame) if not common_frame.empty else {"coverage": pd.DataFrame()}
        coverage_rows = common_statistics["coverage"]
        st.dataframe(coverage_rows, width="stretch", hide_index=True)
        meta_columns = [name for name in ["product_name", "brands", "categories", "countries", "serving_size"] if name in common_frame]
        st.dataframe(pd.DataFrame([{"field": name, "products_with_value": int(_nonempty(common_frame[name]).sum()), "fraction_of_common_products": safe_fraction(_nonempty(common_frame[name]))} for name in meta_columns]), width="stretch", hide_index=True)
        st.subheader("Compare with the broad 50K sample")
        st.caption("Descriptive comparison only: both are selected datasets from potentially different export dates. This is not a representative-population or causal comparison.")
        if enriched_path.exists():
            broad_frame = filtered_frame(load_local_csv(str(enriched_path), enriched_path.stat().st_mtime), selected_countries, selected_categories)
            comparison_rows = []
            for name in TARGET_FIELDS:
                for label, data in [("Broad 50K sample", broad_frame), ("Annotation-selected common subset", common_frame)]:
                    values = pd.to_numeric(data[name], errors="coerce") if name in data else pd.Series(dtype=float)
                    comparison_rows.append({"dataset": label, "nutrient": name, "products": len(data), "numeric_support": int(values.notna().sum()), "numeric_coverage": float(values.notna().sum()/len(data)) if len(data) else None, "median": values.median() if values.notna().any() else None})
            comparison_frame = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_frame, width="stretch", hide_index=True)
            fig = px.bar(comparison_frame, x="nutrient", y="numeric_coverage", color="dataset", barmode="group", labels={"numeric_coverage": "Numeric coverage", "nutrient": "Source nutrient field"})
            fig.update_layout(**plot_layout(390))
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig, width="stretch")
        st.subheader("Connect product nutrition to annotated images")
        inspect_split = st.selectbox("Official image split to inspect", ["All", "train", "test"], key="common_inspect_split")
        inspect_images = visible_common_images.loc[visible_common_images["match_status"].eq(MATCHED)]
        if inspect_split != "All":
            inspect_images = inspect_images.loc[inspect_images["dataset_split"].eq(inspect_split)]
        choices = inspect_images["barcode"].dropna().drop_duplicates().tolist()
        if choices:
            names_by_code = common_products.set_index("barcode")["product_name"].fillna("").to_dict()
            selected_code = st.selectbox("Common product barcode", choices, format_func=lambda code: f"{code} · {names_by_code.get(code, '')}", key=f"common_product_{inspect_split}")
            show_linked_product_context(selected_code, key="common_product_download")
            associated = inspect_images.loc[inspect_images["barcode"].eq(selected_code)]
            chosen_key = st.selectbox("Annotated image", associated["image_key"].tolist(), key=f"common_image_{selected_code}_{inspect_split}")
            image_record = associated.loc[associated["image_key"].eq(chosen_key)].iloc[0]
            st.caption(f"Image {image_record['image_id']} · official split {image_record['dataset_split']} · {image_record['image_width']}×{image_record['image_height']} · {image_record['token_count']} OCR tokens. {image_record['image_reference_status']}.")
            url = image_record["image_url"]
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                st.link_button("Open original annotated nutrition image", url)
                if st.checkbox("Load this annotated image from its source URL", key="common_load_image"):
                    st.image(url, caption="External annotated image; barcode linkage does not verify current packaging identity.", width="stretch")
            st.dataframe(annotation_preview(LINKED_DIR, chosen_key, limit=500), width="stretch", hide_index=True)
            st.caption("Token preview capped at 500 rows; original OCR words, boxes and gold BIO labels are unchanged. Product nutrients above are context, not replacement gold labels.")
        with st.expander("Snapshot provenance, unmatched barcodes and image-match caveats"):
            st.json({"source": common_provenance["part1"]["source"], "dataset_revision": common_provenance["part2"]["revision"], "summary": overlap, "quality": common_provenance["quality"]})
            st.dataframe(common_products.loc[~common_products["match_status"].eq(MATCHED), ["barcode", "match_status"]], width="stretch", hide_index=True)
            st.warning("Barcode equality links product records only. URL differences do not prove different pixels, and URL equality would not prove packaging/nutrient-version identity. No image was pixel-verified by the integration.")
        st.download_button("Download common-subset EDA data", common_frame.to_csv(index=False).encode("utf-8"), file_name="common_product_subset.csv", mime="text/csv", key="common_eda_csv")
        for artifact in ["dataset_overlap_summary.csv", "data_dictionary.md"]:
            path = LINKED_DIR / artifact
            if path.exists():
                st.download_button(f"Download {artifact}", path.read_bytes(), file_name=artifact, key=f"common_{artifact}")

if active_section == "Data Overview":
    st.subheader("Data Overview")
    st.caption(f"All charts on this page use the selected dataset and country/category filters ({len(frame):,} products).")
    if selected_countries or selected_categories:
        st.caption("Filters select products (OR within each filter, AND across country and category). Bars may include other labels attached to those same selected products—for example a Snacks product can also be labelled Sweet snacks.")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Where the sample comes from")
        counts = country_counts(frame).head(12)
        if not counts.empty:
            fig = px.bar(counts.sort_values(), orientation="h", labels={"value": "Label occurrences", "index": "Country label"}, color=counts.sort_values().values, color_continuous_scale=["#B8D7FF", PALETTE["blue"]])
            fig.update_layout(**plot_layout(390), showlegend=False, coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No country column in this source.")
    with right:
        st.subheader("Most common categories")
        if "categories" in frame:
            cats = frame["categories"].dropna().astype("string").str.split(",").explode().str.strip()
            cats = cats[~cats.str.casefold().isin({"", "undefined", "nan", "none", "null", "n/a", "na"})].value_counts().head(12).sort_values()
            fig = px.bar(cats, x=cats.values, y=cats.index, orientation="h", labels={"x": "Product-field occurrences", "y": "Category"}, color_discrete_sequence=[PALETTE["teal"]])
            fig.update_layout(**plot_layout(390), yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No categories field in this source.")

    st.subheader("Selected nutrition-field coverage")
    if not coverage.empty:
        cov = coverage.sort_values("coverage_fraction", ascending=True)
        fig = px.bar(cov, x="coverage_fraction", y="field", orientation="h", color="coverage_fraction", color_continuous_scale=["#F6C667", PALETTE["teal"]], labels={"coverage_fraction": "Non-missing fraction", "field": "Nutrition field"}, hover_data={"non_missing_count": True, "coverage_fraction": ":.1%"})
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(**plot_layout(max(380, len(cov) * 22)), coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    serving_fields = [column for column in ["countries", "quantity", "serving_size"] if column in frame]
    if serving_fields:
        st.subheader("Locale and serving metadata coverage")
        serving_rows = pd.DataFrame([
            {"field": column, "non_missing_count": int(_nonempty(frame[column]).sum()), "coverage_fraction": safe_fraction(_nonempty(frame[column]))}
            for column in serving_fields
        ])
        fig = px.bar(serving_rows, x="coverage_fraction", y="field", orientation="h", color="coverage_fraction", color_continuous_scale=["#F6C667", PALETTE["blue"]], labels={"coverage_fraction": "Non-missing fraction", "field": "Context field"}, hover_data={"non_missing_count": True, "coverage_fraction": ":.1%"})
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(**plot_layout(230), coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")

# ── Nutrients ──────────────────────────────────────────────────────────────
if active_section == "Nutrients":
    st.subheader("Nutrient distribution explorer")
    available_numeric = [column for column in nutrition_cols if pd.to_numeric(frame[column], errors="coerce").notna().any()]
    if available_numeric:
        controls = st.columns([2, 1, 1])
        with controls[0]: selected_nutrient = st.selectbox("Field", available_numeric, index=available_numeric.index("energy-kcal_100g") if "energy-kcal_100g" in available_numeric else 0, format_func=friendly_name)
        with controls[1]: chart_type = st.selectbox("Chart", ["Histogram", "Box plot", "Violin"])
        with controls[2]: clip_display = st.checkbox("Clip display at 99th percentile", value=True, help="Visual-only clipping; statistics retain raw values.")
        values = pd.to_numeric(frame[selected_nutrient], errors="coerce").dropna()
        display_values = values.clip(upper=values.quantile(.99)) if clip_display and len(values) else values
        if chart_type == "Histogram":
            fig = px.histogram(display_values, nbins=50, labels={"value": "Stored numeric value"}, color_discrete_sequence=[PALETTE["blue"]])
        elif chart_type == "Box plot":
            fig = px.box(display_values, points="outliers", labels={"value": "Stored numeric value"}, color_discrete_sequence=[PALETTE["purple"]])
        else:
            fig = px.violin(display_values, box=True, points="outliers", labels={"value": "Stored numeric value"}, color_discrete_sequence=[PALETTE["teal"]])
        fig.update_layout(**plot_layout(360), title=friendly_name(selected_nutrient), xaxis_title="Value as stored in OFF export")
        st.plotly_chart(fig, width="stretch")
        if selected_nutrient in nutrition_stats["field"].values:
            row = nutrition_stats.loc[nutrition_stats["field"].eq(selected_nutrient)].iloc[0]
            stat_cols = st.columns(7)
            for col, label, key in zip(stat_cols, ["Records", "Median", "P01", "P99", "Maximum", "Negative"], ["valid_numeric_count", "median", "p01", "p99", "max", "negative_count"]):
                val = row.get(key)
                col.metric(label, f"{val:,.4g}" if pd.notna(val) else "—")
            mode = numeric_modes(values)
            mode_text = "No unique mode" if mode["status"] == "all_values_unique" else ", ".join(f"{value:,.4g}" for value in mode["values"][:3]) or "—"
            stat_cols[6].metric("Mode", mode_text, f"frequency {mode['frequency']}")
            st.caption("Mode uses un-clipped numeric values after filters. All tied modes are retained; all-unique data has no unique mode.")
            if len(mode["values"]) > 1 and mode["status"] != "all_values_unique":
                with st.expander("All tied modes"):
                    st.dataframe(pd.DataFrame({"mode": mode["values"], "frequency": mode["frequency"]}), width="stretch", hide_index=True)
        st.caption("Values are reported in Open Food Facts stored units; energy may appear in kcal or kJ fields. Review outliers and units before interpreting.")
    else:
        st.info("No numeric nutrient columns detected in this source.")

    st.subheader("Nutrient correlations")
    corr_cols = [column for column in available_numeric if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 20]
    corr_cols = st.multiselect("Fields for correlation matrix", corr_cols, default=[c for c in ["energy-kcal_100g", "fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g"] if c in corr_cols])
    if len(corr_cols) >= 2:
        corr = frame[corr_cols].apply(pd.to_numeric, errors="coerce").corr(min_periods=2)
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", labels={"color": "Pearson r"}, x=[friendly_name(c) for c in corr_cols], y=[friendly_name(c) for c in corr_cols])
        fig.update_layout(**plot_layout(max(420, len(corr_cols) * 52)), title="Available-case Pearson correlation")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Select at least two fields.")

# ── Data quality ────────────────────────────────────────────────────────────
if active_section == "Data quality":
    st.subheader("Missingness and record integrity")
    q1, q2, q3, q4 = st.columns(4)
    code_col = "code" if "code" in frame else None
    q1.metric("Exact duplicate rows", f"{int(frame.duplicated().sum()):,}")
    q2.metric("Duplicate product-code rows", f"{int(frame[code_col].duplicated(keep=False).sum()):,}" if code_col else "N/A")
    q3.metric("Blank product names", f"{int((~_nonempty(frame['product_name'])).sum()):,}" if "product_name" in frame else "N/A")
    q4.metric("Rows with any nutrient", f"{int(summary.get('rows_with_any_structured_nutrition') or 0):,}")

    miss = missingness.head(30).sort_values("missing_fraction")
    fig = px.bar(miss, x="missing_fraction", y="column", orientation="h", color="missing_fraction", color_continuous_scale=["#F6C667", PALETTE["red"]], labels={"missing_fraction": "Missing fraction", "column": "Field"}, hover_data={"missing_count": True})
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(**plot_layout(max(400, len(miss) * 20)), title="Fields with highest missingness", coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Potential numeric anomalies")
    if not nutrition_stats.empty:
        anomaly = nutrition_stats.copy()
        anomaly["above_100_count"] = [int((pd.to_numeric(frame[row.field], errors="coerce") > 100).sum()) if row.field in frame else 0 for row in anomaly.itertuples()]
        anomaly = anomaly.sort_values(["negative_count", "iqr_outlier_count"], ascending=False)
        st.dataframe(anomaly, width="stretch", hide_index=True)
        st.caption("IQR flags and values >100 are prompts to review, not automatic data-error labels. Units, vitamins, overlaps, and per-serving conventions need context.")
        with st.expander("Full numeric summary for every detected nutrient"):
            st.dataframe(nutrition_stats, width="stretch", hide_index=True)
            st.download_button("Download nutrient statistics", nutrition_stats.to_csv(index=False).encode("utf-8"), file_name="nutrition_statistics.csv", mime="text/csv", key="download_nutrient_stats")

# ── Images and alignment ────────────────────────────────────────────────────
if active_section == "Images & alignment":
    st.subheader("Image availability vs nutrition metadata")
    presence_data = pd.DataFrame({
        "Group": ["Any image URL", "Nutrition image URL", "Any selected nutrition target", "Nutrition image + target candidate"],
        "Records": [int(any_image_mask.sum()), int(nutrition_image_mask.sum()), int(target_mask.sum()), int(candidate_mask.sum())],
    })
    fig = px.bar(presence_data, x="Group", y="Records", color="Group", color_discrete_sequence=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]], text_auto=True)
    fig.update_layout(**plot_layout(350), showlegend=False, xaxis_title="", yaxis_title="Product records")
    st.plotly_chart(fig, width="stretch")
    st.markdown('<div class="warning-strip">A candidate means an image URL and metadata coexist. It does not mean the image has been checked or that its visible table matches those values.</div>', unsafe_allow_html=True)

    audit_path = ROOT / "reports" / "eda" / "image_resolution_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, dtype={"product_id": "string", "code": "string"})
        if selected_countries or selected_categories:
            audit_code = "product_id" if "product_id" in audit else "code" if "code" in audit else None
            audit = audit.loc[audit[audit_code].astype("string").isin(frame["code"].astype("string"))] if audit_code and "code" in frame else audit.iloc[0:0]
        st.subheader("Bounded image-resolution audit")
        if not audit.empty and {"width", "height"}.issubset(audit.columns):
            st.caption(f"Small audit only: {int(audit.get('readable', pd.Series(dtype=bool)).sum())}/{len(audit)} decoded images. Not representative of all images.")
            fig = px.scatter(audit, x="width", y="height", color="readable", hover_data=[c for c in ["product_id", "aspect_ratio", "content_type"] if c in audit], color_discrete_map={True: PALETTE["teal"], False: PALETTE["red"]})
            fig.update_layout(**plot_layout(340), title="Audited nutrition-image dimensions", xaxis_title="Width (px)", yaxis_title="Height (px)")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(audit.drop(columns=["image_url"], errors="ignore"), width="stretch", hide_index=True)
    verified_image = ROOT / "data" / "manual_samples" / "241013128_nutrition.jpg"
    if verified_image.exists():
        st.subheader("Manually reviewed panel example")
        st.image(str(verified_image), caption="Product 241013128 · visually checked reference example", width=520)
    manual_log = ROOT / "reports" / "eda" / "manual_verification_log.csv"
    if manual_log.exists():
        st.subheader("Manual alignment checks")
        st.dataframe(pd.read_csv(manual_log), width="stretch", hide_index=True)
        st.caption("Only explicitly documented visual checks count as verified; this is not an accuracy estimate.")
    manual_gold_path = ROOT / "data" / "evaluation" / "gold_records.jsonl"
    if manual_gold_path.exists():
        gold_records = [json.loads(line) for line in manual_gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.subheader("Manually transcribed record-level example")
        selected_gold = st.selectbox("Verified product ID", [record.get("product_id", "unknown") for record in gold_records])
        record = next(record for record in gold_records if record.get("product_id") == selected_gold)
        st.dataframe(pd.DataFrame(record.get("fields", [])), width="stretch", hide_index=True)
        st.caption("This is record-level transcription only. It is not word-level OCR annotation and is insufficient for training or aggregate evaluation.")

# ── Annotation scouting ────────────────────────────────────────────────────
if active_section == "Annotation scout":
    st.subheader("Annotation review queue")
    scout_path = ROOT / "reports" / "annotation_scout" / "annotation_review_queue.csv"
    scout_summary_path = ROOT / "reports" / "annotation_scout" / "summary.json"
    if scout_path.exists():
        scout = pd.read_csv(scout_path, dtype={"code": "string"})
        if "code" in scout and "code" in frame:
            scout = scout.loc[scout["code"].isin(frame["code"])]
        if scout_summary_path.exists():
            scout_meta = json.loads(scout_summary_path.read_text(encoding="utf-8"))
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Nutrition image candidates", f"{scout_meta.get('nutrition_image_url_candidates', 0):,}")
            sm2.metric("Metadata-ranked candidates", f"{scout_meta.get('ranked_candidates_with_any_target', 0):,}")
            sm3.metric("Record-level verified", str(scout_meta.get("record_level_alignment_verified_count", 0)))
            st.caption("Saved scout counters are a fixed full-queue audit; the queue table below follows the selected dataset and country/category filters.")
        status_filter = st.multiselect("Review status", sorted(scout["review_status"].dropna().unique()) if "review_status" in scout else [], default=sorted(scout["review_status"].dropna().unique()) if "review_status" in scout else [])
        view = scout.loc[scout["review_status"].isin(status_filter)] if status_filter and "review_status" in scout else scout
        st.dataframe(view.head(250), width="stretch", hide_index=True, column_config={"nutrition_image_url": st.column_config.LinkColumn("Nutrition image", display_text="Open image")})
        previewable = view.loc[view["nutrition_image_url"].notna()] if "nutrition_image_url" in view else pd.DataFrame()
        if not previewable.empty:
            preview_options = {
                f"{row.get('code', '')} · {str(row.get('product_name', 'Unnamed'))[:55]}": row["nutrition_image_url"]
                for _, row in previewable.head(100).iterrows()
            }
            preview_label = st.selectbox("Preview one nutrition image (manual selection)", list(preview_options))
            st.image(preview_options[preview_label], caption=preview_label, width=520)
        st.download_button("Download visible queue as CSV", view.to_csv(index=False).encode("utf-8"), file_name="annotation_review_queue.csv", mime="text/csv")
        st.caption("Ranking helps select records for review only. Update review_status, reviewer, review_notes, and alignment_verified after checking each image manually.")
    else:
        st.info("No scout output yet. Run `nutrition-scout-annotations` or the extraction workflow notebook.")

    pilot_path = ROOT / "reports" / "annotation_scout" / "annotation_pilot_batch.csv"
    if pilot_path.exists():
        st.subheader("Locale-diverse pilot")
        pilot = pd.read_csv(pilot_path, dtype={"code": "string"})
        if "code" in pilot and "code" in frame:
            pilot = pilot.loc[pilot["code"].isin(frame["code"])]
        st.dataframe(pilot.head(100), width="stretch", hide_index=True, column_config={"nutrition_image_url": st.column_config.LinkColumn("Nutrition image", display_text="Open image")})
        st.download_button("Download pilot CSV", pilot.to_csv(index=False).encode("utf-8"), file_name="annotation_pilot_batch.csv", mime="text/csv")

# ── Part 2 annotated dataset EDA (cached metadata artifacts; no image fetch) ──
if active_section == "Part 2 Dataset EDA":
    st.subheader("Part 2 Dataset · Open Food Facts Nutrient Detection Layout")
    st.info("Country/category filters affect Part 2 exploratory tables and charts through linked product metadata, not official model evaluation. Original train/test labels remain unchanged. Product nutrients are context only—not image-verified targets.")
    context_images = None
    part2_tokens = None
    try:
        context_products, all_context_images, context_provenance = get_linked_data()
        context_images = all_context_images
        if selected_countries or selected_categories:
            filtered_context = filtered_frame(products_for_eda(context_products), selected_countries, selected_categories)
            context_images = all_context_images.loc[all_context_images["barcode"].isin(filtered_context["code"])]
        st.caption(f"Part 2 exploratory selection: {len(context_images):,}/{len(all_context_images):,} images. Active country/category filters exclude images whose linked metadata cannot match; they never drop examples from official model evaluation.")
        part2_tokens = cached_annotation_rows(tuple(context_images["image_key"]), (LINKED_DIR / "integrated_annotations.parquet").stat().st_mtime_ns)
        st.subheader("Part 2 products and annotated images")
        joined_context = merged_image_products(context_products, context_images)
        product_columns = [name for name in ["barcode", "product_name", "brand", "category", "country", "image_id", "dataset_split", "image_url", "match_status", "token_count"] if name in joined_context]
        product_table(joined_context[product_columns], images=context_images)
        st.download_button("Download filtered Part 2 product/image table", joined_context[product_columns].to_csv(index=False).encode("utf-8"), file_name="part2_filtered_products_images.csv", mime="text/csv", key="part2_product_table_download")
    except (OSError, ValueError, ImportError) as error:
        st.warning(f"Filtered linked annotation charts unavailable: {error}")
        if selected_countries or selected_categories:
            st.stop()  # Never silently show unfiltered charts for an active filter.
    with st.expander("Inspect annotated images with linked product metadata", expanded=True):
        if context_images is not None and not context_images.empty:
            context_split = st.selectbox("Official split for linked context", ["train", "test"], key="part2_context_split")
            context_rows = context_images.loc[context_images["dataset_split"].eq(context_split)]
            st.caption(f"{len(context_rows):,} {context_split} images in the exploratory selection. This explorer does not select or filter evaluation records.")
            context_labels = context_rows.set_index("image_key")["barcode"].to_dict()
            if not context_rows.empty:
                context_key = st.selectbox("Part 2 image record", context_rows["image_key"].tolist(), format_func=lambda key: f"{key} · {context_labels[key]}", key=f"part2_context_image_{context_split}")
                context_image = context_rows.loc[context_rows["image_key"].eq(context_key)].iloc[0]
                show_linked_product_context(str(context_image["barcode"]), key="part2_context_download")
                context_url = context_image["image_url"]
                if isinstance(context_url, str) and context_url.startswith(("http://", "https://")):
                    st.link_button("View image", context_url)
                st.subheader("Entity-tagging input table · original Part 2 annotations")
                product_table(annotation_preview(LINKED_DIR, context_key, limit=500), images=context_images)
                st.caption("Original token index, OCR token, normalized bounding box, gold BIO ID and label—the aligned inputs used for tagging. Preview capped at 500; no annotations are changed.")
            else:
                st.info("No images in this official split match the current exploratory filters.")
        else:
            st.info("No linked images match the country/category filters. Clear a filter to inspect tagging inputs.")
    if context_images is not None and context_images.empty:
        st.info("No Part 2 chart data for the current selection. Full-inventory statistics remain available in Part 1 + Part 2; official test results remain available in Results.")
        st.stop()
    part2_dir = ROOT / "reports" / "part2_eda"
    summary_path = part2_dir / "part2_eda_summary.json"
    if not summary_path.exists():
        st.info("Part 2 EDA artifacts are not available yet. Run part2/part2_dataset_eda.ipynb to fetch bounded annotation metadata and generate this view. The notebook does not download image payloads.")
    else:
        part2_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        split_counts = part2_summary.get("split_counts", {})
        st.markdown(
            f'<div class="info-strip">Dataset revision {part2_summary.get("revision", "unknown")} · '
            f'{part2_summary.get("license", "license not recorded")} · metadata-only EDA; '
            'no images downloaded. BIO value labels are not complete nutrient-name/value row supervision.</div>',
            unsafe_allow_html=True,
        )
        kpis = st.columns(4)
        with kpis[0]: metric_card("Annotation examples", f"{part2_summary.get('total_rows', 0):,}", "official HF splits")
        with kpis[1]: metric_card("Train examples", f"{split_counts.get('train', 0):,}", "published train split")
        with kpis[2]: metric_card("Test examples", f"{split_counts.get('test', 0):,}", "published test split")
        with kpis[3]: metric_card("BIO labels", str(part2_summary.get("label_count", 0)), "ClassLabel mapping")

        quality = part2_summary.get("quality_summary", {})
        flagged_record_count = int(part2_summary.get("suspicious_rows_flagged", 0))
        qcols = st.columns(4)
        with qcols[0]: metric_card("Unique barcodes", f"{part2_summary.get('unique_valid_barcodes', 0):,}", "leading zeros preserved")
        with qcols[1]: metric_card("Train/test barcode overlap", str(part2_summary.get("barcode_leakage_across_train_test", 0)), "checked by barcode")
        with qcols[2]: metric_card("Sequence alignment failures", str(quality.get("sequence_alignment_failures", 0)), "tokens / boxes / tags")
        with qcols[3]: metric_card("Review flags", f"{flagged_record_count:,}", "automated review flags")
        st.caption(f"Full-inventory snapshot counters above remain fixed. Charts and record tables below use the exploratory filters. Full-inventory review rule: a numeric OCR token has |value| > 10,000; {flagged_record_count:,} records flagged, not confirmed errors. Original annotations are preserved.")

        review_category_path = part2_dir / "part2_review_flag_categories.csv"
        flagged_tokens_path = part2_dir / "part2_flagged_tokens.csv"
        if review_category_path.exists():
            st.subheader("Investigation of flagged records")
            st.caption("Exclusive record categories use conservative context heuristics. They are not manually image-verified; ambiguous and conflicting evidence remains Uncertain. Category counts should sum to the flagged-record total.")
            if flagged_tokens_path.exists():
                flagged_tokens = pd.read_csv(flagged_tokens_path, dtype={"barcode": "string", "image_id": "string"}, keep_default_na=False)
                if context_images is not None:
                    allowed_pairs = set(zip(context_images["barcode"].astype(str), context_images["image_id"].astype(str)))
                    flagged_tokens = flagged_tokens.loc[[pair in allowed_pairs for pair in zip(flagged_tokens["barcode"].astype(str), flagged_tokens["image_id"].astype(str))]]
                review_categories = flagged_tokens.drop_duplicates(["barcode", "image_id", "category"]).groupby("category").size().rename("flagged_records").reset_index()
                st.dataframe(review_categories, width="stretch", hide_index=True)
                st.caption(f"Inspected {len(flagged_tokens):,} flagged OCR token occurrences across {flagged_tokens['barcode'].nunique():,} product barcodes. These are source OCR strings; no annotations were corrected.")
                example_columns = ["category", "barcode", "image_id", "original_ocr_token", "parsed_value", "surrounding_original_ocr_text", "evidence_note", "manual_visual_verification"]
                available_example_columns = [column for column in example_columns if column in flagged_tokens]
                examples = flagged_tokens.groupby("category", group_keys=False).head(2)
                product_table(examples[available_example_columns], images=context_images)
                if not flagged_tokens.empty:
                    record_options = {
                        f"{row['category']} · {row['barcode']} · image {row['image_id']} · token {row['original_ocr_token']}": index
                        for index, row in examples.iterrows()
                    }
                    selected_flag_example = st.selectbox("Show full original OCR text for a representative flagged token", list(record_options), key="part2_flagged_ocr_example")
                    original_text = flagged_tokens.loc[record_options[selected_flag_example], "full_original_ocr_text"]
                    st.text_area("Full original OCR token sequence (unchanged)", str(original_text), height=150, disabled=True, key="part2_flagged_full_ocr")
                st.download_button("Download flagged token investigation", flagged_tokens.to_csv(index=False).encode("utf-8"), file_name="part2_flagged_tokens.csv", mime="text/csv", key="part2_flagged_tokens_download")
        else:
            st.info("Flagged-token investigation artifacts are missing. Rerun the Part 2 EDA notebook to generate category counts and original OCR contexts.")

        st.caption("Dataset ID: openfoodfacts/nutrient-detection-layout")
        if context_images is not None and not context_images.empty:
            st.subheader("Image shape · current exploratory filters")
            shapes = context_images.copy()
            shapes["aspect_ratio"] = pd.to_numeric(shapes["image_width"], errors="coerce") / pd.to_numeric(shapes["image_height"], errors="coerce").replace(0, np.nan)
            st.plotly_chart(px.histogram(shapes, x="aspect_ratio", color="dataset_split", nbins=35), width="stretch")

        label_path = part2_dir / "part2_label_support.csv"
        if label_path.exists():
            support = label_support(part2_tokens) if part2_tokens is not None else pd.read_csv(label_path)
            non_o_labels = support.loc[support["label"].ne("O")].copy()
            non_o_labels["total_support"] = non_o_labels["train_token_support"] + non_o_labels["test_token_support"]
            st.subheader("BIO label support")
            st.caption(f"Filtered O-class support: train {int(support.loc[support['label'].eq('O'), 'train_token_support'].sum()):,}; test {int(support.loc[support['label'].eq('O'), 'test_token_support'].sum()):,}. All bars are recomputed from selected images; the chart excludes O so nutrient entity counts remain legible.")
            top_labels = non_o_labels.nlargest(20, "total_support")
            melted = top_labels.melt(id_vars=["label"], value_vars=["train_token_support", "test_token_support"], var_name="split", value_name="token_support")
            melted["split"] = melted["split"].str.replace("_token_support", "", regex=False)
            fig = px.bar(melted, x="token_support", y="label", color="split", barmode="group", orientation="h", labels={"token_support": "Tagged token support", "label": "BIO label", "split": "Official split"}, color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"]])
            fig.update_layout(**plot_layout(max(480, len(top_labels) * 24)), title="Top 20 nutrient BIO labels (O excluded)", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
            st.caption("Token support means the number of OCR token positions assigned to a BIO label—not products or complete nutrient entities. The O label means outside any annotated entity; its much larger count would flatten the nutrient-label bars, so O is excluded from the main chart and counted separately above.")
            st.dataframe(pd.DataFrame([
                {"OCR token": "Fat", "BIO label": "O", "Support contribution": "1 token to O"},
                {"OCR token": "18", "BIO label": "B-FAT", "Support contribution": "1 token to B-FAT"},
                {"OCR token": "g", "BIO label": "I-FAT", "Support contribution": "1 token to I-FAT"},
            ]), width="stretch", hide_index=True)
            with st.expander("Inspect all label IDs and support"):
                st.dataframe(support, width="stretch", hide_index=True)
                st.download_button("Download Part 2 label support", support.to_csv(index=False).encode("utf-8"), file_name="part2_label_support.csv", mime="text/csv", key="part2_label_support_download")

        sequence_path = part2_dir / "part2_sequence_summary.csv"
        if sequence_path.exists():
            sequence_data = pd.read_csv(sequence_path)
            if not {"token_count", "split"}.issubset(sequence_data.columns):
                metadata_path = part2_dir / "part2_example_metadata.csv"
                sequence_data = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
            if context_images is not None:
                sequence_data = context_images.rename(columns={"dataset_split": "split"})
            st.subheader("OCR token sequence lengths")
            if {"token_count", "split"}.issubset(sequence_data.columns):
                fig = px.histogram(sequence_data, x="token_count", color="split", nbins=45, barmode="overlay", opacity=.65, labels={"token_count": "OCR tokens per image", "split": "Official split"}, color_discrete_sequence=[PALETTE["blue"], PALETTE["orange"]])
                fig.update_layout(**plot_layout(340), title="Sequence length distribution")
                st.plotly_chart(fig, width="stretch")
                st.caption("Each image contributes one OCR-token count. The histogram shows how those per-image sequence lengths are distributed in the train and test splits; the table gives the split-level mean, median, quartiles, and range.")
                st.dataframe(sequence_data.groupby("split")["token_count"].describe().reset_index().round(2), width="stretch", hide_index=True)
            else:
                st.info("Per-example token counts are unavailable. Rerun the Part 2 EDA notebook to regenerate the sequence artifact.")

        value_path = part2_dir / "part2_tagged_value_summary.csv"
        if value_path.exists():
            value_summary = pd.read_csv(value_path)
            if selected_countries or selected_categories:
                value_summary = pd.DataFrame()  # This static aggregate has no product keys; never present it as filtered.
            st.subheader("Parsed numeric mentions from tagged BIO spans")
            st.caption("These are token mentions grouped by annotated nutrient/value class and unit—not row-linked product nutrition facts. Values with unit_not_read remain unnormalized.")
            if value_summary.empty:
                st.caption("Full-snapshot parsed-value aggregate is hidden while filters are active because it has no image-level keys. Use the filtered tagging input table above for actual source values.")
            else:
                st.dataframe(value_summary.sort_values("mention_count", ascending=False).head(30), width="stretch", hide_index=True)
            st.download_button("Download tagged-value summary", value_summary.to_csv(index=False).encode("utf-8"), file_name="part2_tagged_value_summary.csv", mime="text/csv", key="part2_value_summary_download")

        with st.expander("Metadata, limitations, and quality checks"):
            st.write({
                "dataset_id": part2_summary.get("dataset_id"),
                "last_modified": part2_summary.get("last_modified"),
                "feature_names": part2_summary.get("feature_names"),
                "license": part2_summary.get("license"),
                "quality_checks": quality,
                "images_downloaded": part2_summary.get("images_downloaded"),
                "caveat": part2_summary.get("caveat"),
            })
            missing_path = part2_dir / "part2_metadata_missingness.csv"
            if missing_path.exists():
                st.dataframe(pd.read_csv(missing_path).head(30), width="stretch", hide_index=True)
            review_path = part2_dir / "part2_review_flags.csv"
            if review_path.exists():
                review_rows = pd.read_csv(review_path, dtype={"barcode": "string", "image_id": "string"})
                if context_images is not None and "barcode" in review_rows:
                    review_rows = review_rows.loc[review_rows["barcode"].isin(context_images["barcode"])]
                product_table(review_rows.head(100), images=context_images)
            for artifact_name in ["part2_eda_summary.json", "part2_example_metadata.csv", "part2_review_flags.csv"]:
                artifact = part2_dir / artifact_name
                if artifact.exists():
                    st.download_button(f"Download {artifact_name}", artifact.read_bytes(), file_name=artifact_name, mime="application/json" if artifact.suffix == ".json" else "text/csv", key=f"download_{artifact_name}")

# ── Pretrained Open Food Facts LayoutLMv3 BIO tagging ───────────────────────
if active_section == "BIO Entity Tagging":
    from part2.bio_review import (
        summarize_field_evaluations, prediction_diagnostics,
        load_manual_reference_review, run_manual_reference_review,
    )
    from part2.layoutlm_bio import (
        INFERENCE_SCHEMA_VERSION,
        MODEL_CACHE_DIR,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        InsufficientResourcesError,
        analyze_bio_errors,
        compare_fields_to_off_metadata,
        compare_fields_to_gold,
        draw_prediction_boxes,
        fetch_openfoodfacts_product,
        evaluate_sequences,
        load_pretrained_model,
        predict_uploaded_image,
        resource_snapshot,
        rule_baseline_coarse_bio,
        to_coarse_bio,
        predict_record,
        save_prediction_bundle,
    )

    st.subheader("BIO Entity Tagging · pretrained by Open Food Facts")
    from nutrition_extraction.full_cohort_dashboard import render as render_full_cohort
    render_full_cohort(ROOT)
    st.markdown(
        f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · revision `{MODEL_REVISION}` · "
        f"license `{MODEL_LICENSE}`. This is the Open Food Facts fine-tuned LayoutLMv3-large checkpoint; "
        "the local ONNX export is preferred and runs with ONNX Runtime CPU only. No training, paid API, "
        "or automatic model-weight download is used.")
    st.info("BIO example: `Fat → O`, `18 → B-FAT_100G`, `g → I-FAT_100G`. B starts a labeled span, I continues it, and O means outside the annotated value span. The dataset usually leaves nutrient-name words tagged O.")
    st.caption("The model requires the original image, dataset-supplied OCR tokens, and matching normalized [0,1000] boxes. It does not run OCR or alter ground-truth annotations. Inference starts with one official test image; optional extension remains safety-gated.")

    resource_info = resource_snapshot(ROOT / MODEL_CACHE_DIR)
    st.caption(f"Weights are read from {ROOT / MODEL_CACHE_DIR}. ONNX Runtime uses CPU sequential execution with one intra-op thread. The code never fetches weights; missing small sidecars or runtime packages produce a specific error.")

    @st.cache_resource(show_spinner=False)
    def _cached_openfoodfacts_layoutlm():
        return load_pretrained_model(ROOT / MODEL_CACHE_DIR)

    cached_bundle_path = ROOT / "data" / "part2_cache" / "layoutlm_bio" / "latest_test_bundle.json"
    if cached_bundle_path.is_file() and "part2_bio_predictions" not in st.session_state:
        try:
            cached_bundle = json.loads(cached_bundle_path.read_text(encoding="utf-8"))
            if cached_bundle.get("inference_schema_version") != INFERENCE_SCHEMA_VERSION:
                raise ValueError("Legacy partial-coverage results were preserved on disk but are not shown as current. Run inference again to create coverage-complete results.")
            cached_records, cached_predictions, cached_rules = [], [], []
            for sample in cached_bundle["samples"]:
                image_path = Path(sample["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(f"Cached image for barcode {sample['barcode']} is missing: {image_path}")
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
                cached_records.append({
                    "source_index": sample.get("source_index"),
                    "barcode": sample["barcode"], "image_path": str(image_path), "image": image,
                    "image_size": sample["image_size"], "tokens": sample["tokens"],
                    "bboxes": sample["bboxes"],
                })
                cached_predictions.append({
                    "predicted_labels": sample["predicted_labels"], "gold_labels": sample["gold_labels"],
                    "covered_token_count": sample["covered_token_count"], "token_count": sample["token_count"],
                    "coverage_mask": sample["coverage_mask"],
                    "inference_windows": sample.get("inference_windows"),
                    "inference_seconds": sample["inference_seconds"],
                    "process_rss_after_mb": sample.get("process_rss_after_mb"),
                    "process_rss_delta_mb": sample.get("process_rss_delta_mb"),
                    "available_ram_delta_mb": sample.get("available_ram_delta_mb"),
                    "cache_hit": sample["cache_hit"], "extracted_entities": sample["extracted_entities"],
                    "structured_output": sample.get("structured_output"),
                    "runtime": cached_bundle["runtime"]["runtime"],
                })
                cached_rules.append(sample["coarse_rule_labels"])
            st.session_state["part2_bio_sample_records"] = cached_records
            st.session_state["part2_bio_predictions"] = cached_predictions
            st.session_state["part2_bio_coarse_rules"] = cached_rules
            st.session_state["part2_bio_runtime_info"] = cached_bundle["runtime"]
            st.session_state["part2_bio_fine_metrics"] = cached_bundle["fine_grained_metrics"]
            st.session_state["part2_bio_coarse_model_metrics"] = cached_bundle["coarse_model_metrics"]
            st.session_state["part2_bio_coarse_rule_metrics"] = cached_bundle["coarse_rule_metrics"]
            st.session_state["part2_bio_evaluation_metadata"] = cached_bundle.get("evaluation_metadata", {})
            st.session_state["part2_bio_evaluation_failures"] = cached_bundle.get("evaluation_metadata", {}).get("failures", [])
            st.success("Loaded cached token predictions and evaluation; model weights were not loaded.")
        except ValueError as error:
            st.warning(str(error))
        except Exception as error:
            st.error(f"Could not display cached BIO results: {type(error).__name__}: {error}")

    if not resource_info["safe_to_load"] and "part2_bio_predictions" not in st.session_state:
        st.warning("Local inference is paused by the resource safety check. Saved results can still be viewed; retry after closing other workloads.")

    def run_bio_predictions(sample_records: list[dict[str, Any]]) -> None:
        model, processor, model_id2label, runtime_info = _cached_openfoodfacts_layoutlm()
        predictions = [
            predict_record(
                sample_record, model=model, processor=processor, id2label=model_id2label,
                cache_dir=ROOT / "data" / "part2_cache" / "layoutlm_bio",
            )
            for sample_record in sample_records
        ]
        shared_gold, fine_predictions, coarse_gold, coarse_model, coarse_rules = [], [], [], [], []
        for sample_record, prediction in zip(sample_records, predictions):
            gold_labels = prediction["gold_labels"]
            predicted_labels = prediction["predicted_labels"]
            shared_gold.append(gold_labels)
            fine_predictions.append(predicted_labels)
            coarse_gold.append(to_coarse_bio(gold_labels))
            coarse_model.append(to_coarse_bio(predicted_labels))
            coarse_rules.append(rule_baseline_coarse_bio(
                sample_record["tokens"], sample_record["bboxes"], sample_record["image"].size,
            ))

        st.session_state["part2_bio_sample_records"] = sample_records
        st.session_state["part2_bio_predictions"] = predictions
        st.session_state["part2_bio_coarse_rules"] = coarse_rules
        st.session_state["part2_bio_runtime_info"] = runtime_info
        coverage_masks = [prediction["coverage_mask"] for prediction in predictions]
        st.session_state["part2_bio_fine_metrics"] = evaluate_sequences(shared_gold, fine_predictions, aligned=True, coverage_masks=coverage_masks)
        st.session_state["part2_bio_coarse_model_metrics"] = evaluate_sequences(coarse_gold, coarse_model, aligned=True, coverage_masks=coverage_masks)
        st.session_state["part2_bio_coarse_rule_metrics"] = evaluate_sequences(coarse_gold, coarse_rules, aligned=True, coverage_masks=coverage_masks)
        st.session_state.pop("part2_bio_evaluation_metadata", None)
        st.session_state.pop("part2_bio_evaluation_failures", None)
        st.session_state["part2_bio_error_rows"] = analyze_bio_errors(sample_records, predictions)
        st.session_state.pop("part2_bio_off_comparisons", None)
        st.session_state["part2_bio_bundle_path"] = str(save_prediction_bundle(
            sample_records, predictions,
            runtime_info=runtime_info,
            fine_metrics=st.session_state["part2_bio_fine_metrics"],
            coarse_model_metrics=st.session_state["part2_bio_coarse_model_metrics"],
            coarse_rule_metrics=st.session_state["part2_bio_coarse_rule_metrics"],
            coarse_rule_labels=coarse_rules,
            output_path=ROOT / "data" / "part2_cache" / "layoutlm_bio" / "latest_test_bundle.json",
        ))

    if st.button("Run seeded evaluation on up to 20 official test images", key="run_part2_bio_20", help="Uses the cached ONNX session when available; otherwise the loader enforces its resource safety check before opening weights."):
        try:
            from part2.layoutlm_bio import evaluate_official_test_subset

            with st.spinner("Sampling official test metadata with seed 42; processing images sequentially on CPU…"):
                model, processor, model_id2label, runtime_info = _cached_openfoodfacts_layoutlm()
                evaluation = evaluate_official_test_subset(
                    model=model,
                    processor=processor,
                    id2label=model_id2label,
                    sample_count=20,
                    seed=42,
                    image_cache=ROOT / "data" / "part2_cache" / "images",
                    inference_cache=ROOT / "data" / "part2_cache" / "layoutlm_bio",
                    min_available_ram_gb=0.75,
                )
            if not evaluation["predictions"]:
                st.error(f"No sample completed inference. Failures: {evaluation['failures']}")
            else:
                st.session_state["part2_bio_sample_records"] = evaluation["records"]
                st.session_state["part2_bio_predictions"] = evaluation["predictions"]
                st.session_state["part2_bio_coarse_rules"] = evaluation["rule_labels"]
                st.session_state["part2_bio_runtime_info"] = runtime_info
                st.session_state["part2_bio_fine_metrics"] = evaluation["metrics"]["fine_grained"]
                st.session_state["part2_bio_coarse_model_metrics"] = evaluation["metrics"]["coarse_model"]
                st.session_state["part2_bio_coarse_rule_metrics"] = evaluation["metrics"]["coarse_rule"]
                st.session_state["part2_bio_error_rows"] = evaluation["error_rows"]
                st.session_state["part2_bio_evaluation_failures"] = evaluation["failures"]
                st.session_state["part2_bio_evaluation_metadata"] = {
                    "seed": evaluation["seed"],
                    "requested_images": evaluation["requested_images"],
                    "official_test_rows": evaluation["official_test_rows"],
                    "selected_source_indices": evaluation["selected_source_indices"],
                    "successful_images": evaluation["successful_images"],
                    "failures": evaluation["failures"],
                }
                save_prediction_bundle(
                    evaluation["records"], evaluation["predictions"], runtime_info=runtime_info,
                    fine_metrics=evaluation["metrics"]["fine_grained"],
                    coarse_model_metrics=evaluation["metrics"]["coarse_model"],
                    coarse_rule_metrics=evaluation["metrics"]["coarse_rule"],
                    coarse_rule_labels=evaluation["rule_labels"],
                    evaluation_metadata=st.session_state["part2_bio_evaluation_metadata"],
                    output_path=cached_bundle_path,
                )
                st.success(f"Completed {evaluation['successful_images']} of {evaluation['requested_images']} sampled official test images; saved results and actual failures locally.")
        except Exception as error:
            st.error(f"Expanded evaluation failed: {resource_safe_message(error)}")

    if st.button("Run local CPU model on 1 official test image", type="primary", disabled=not resource_info["safe_to_load"]):
        try:
            from part2.config import DatasetConfig
            from part2.dataset import HuggingFaceNutritionDataset, BIO_LABELS

            with st.spinner("Opening the local ONNX CPU session, then fetching one official test image…"):
                model, processor, model_id2label, runtime_info = _cached_openfoodfacts_layoutlm()
                dataset = HuggingFaceNutritionDataset(DatasetConfig(
                    split="test", subset_size=1, batch_size=1,
                    image_cache=ROOT / "data" / "part2_cache" / "images",
                ))
                run_bio_predictions([dataset[0]])
            st.success(f"One-image inference completed with {runtime_info['runtime']}. Prediction cache is local.")
        except InsufficientResourcesError as error:
            st.error(resource_safe_message(error))
        except Exception as error:
            st.error(f"BIO inference could not complete: {error}")

    if "part2_bio_predictions" in st.session_state:
        sample_records = st.session_state["part2_bio_sample_records"]
        predictions = st.session_state["part2_bio_predictions"]
        coarse_rules = st.session_state["part2_bio_coarse_rules"]
        fine_metrics = st.session_state["part2_bio_fine_metrics"]
        coarse_model_metrics = st.session_state["part2_bio_coarse_model_metrics"]
        coarse_rule_metrics = st.session_state["part2_bio_coarse_rule_metrics"]

        if len(sample_records) == 1:
            current_ram = resource_snapshot(ROOT / MODEL_CACHE_DIR)["available_ram_gb"]
            can_extend = current_ram >= 1.0
            if st.button("Optionally extend to 3 test images", disabled=not can_extend, help="The additional images run sequentially after the local safety check."):
                try:
                    from part2.config import DatasetConfig
                    from part2.dataset import HuggingFaceNutritionDataset

                    latest_ram = resource_snapshot(ROOT / MODEL_CACHE_DIR)["available_ram_gb"]
                    if latest_ram < 1.0:
                        raise InsufficientResourcesError("Memory safety check paused continuation.")
                    with st.spinner("Fetching and processing only test images 2–3, sequentially…"):
                        dataset = HuggingFaceNutritionDataset(DatasetConfig(
                            split="test", subset_size=3, batch_size=1,
                            image_cache=ROOT / "data" / "part2_cache" / "images",
                        ))
                        model, processor, model_id2label, _ = _cached_openfoodfacts_layoutlm()
                        run_bio_predictions([*sample_records, dataset[1], dataset[2]])
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not extend inference: {resource_safe_message(error)}")
            if not can_extend:
                st.caption("Three-image extension paused by the resource safety check.")

        st.subheader("Fine-grained model BIO entity metrics")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Precision", f"{fine_metrics['precision']:.3f}")
        metric_cols[1].metric("Recall", f"{fine_metrics['recall']:.3f}")
        metric_cols[2].metric("F1", f"{fine_metrics['f1']:.3f}", f"{fine_metrics['true_positive_spans']} exact spans")
        metric_cols[3].metric("Token accuracy", f"{fine_metrics['token_accuracy']:.3f}", f"{fine_metrics['token_support']:,} tokens")
        mean_latency = sum(item["inference_seconds"] for item in predictions) / len(predictions)
        metric_cols[4].metric("Inference time", f"{mean_latency:.2f} s/image", "processor + CPU forward pass")
        covered_total = sum(item["covered_token_count"] for item in predictions)
        source_total = sum(item["token_count"] for item in predictions)
        st.caption(f"Measured on {len(sample_records)} official test images with {st.session_state['part2_bio_runtime_info']['runtime']}: {covered_total:,}/{source_total:,} source tokens covered by overlapping windows. Gold entity coverage: {fine_metrics.get('gold_entity_coverage', 0):.1%}. Metrics use explicit coverage masks; these are subset results, not full-test benchmarks.")
        eval_metadata = st.session_state.get("part2_bio_evaluation_metadata", {})
        if eval_metadata:
            st.caption(f"Fixed seed {eval_metadata['seed']} · sampled from {eval_metadata['official_test_rows']} official test records · selected indices: {eval_metadata['selected_source_indices']}.")
        failures = st.session_state.get("part2_bio_evaluation_failures", [])
        if failures:
            with st.expander(f"Evaluation failures ({len(failures)})"):
                st.dataframe(pd.DataFrame([{**failure, "error": resource_safe_message(failure.get("error", ""))} for failure in failures]), width="stretch", hide_index=True)

        st.subheader("Comparable rule-baseline comparison")
        st.caption("For a fair shared-label comparison, fine-grained nutrient BIO labels are projected to coarse VALUE BIO labels, and serving-size labels to BASIS BIO labels. The existing lexical rule baseline runs on the same supplied OCR tokens and boxes; nutrient-name detections are mapped to O because nutrient names are generally not annotated in this corpus. Both methods are scored against the same projected gold spans.")
        comparison = pd.DataFrame([
            {"method": "Open Food Facts pretrained LayoutLMv3", "precision": coarse_model_metrics["precision"], "recall": coarse_model_metrics["recall"], "F1": coarse_model_metrics["f1"], "scope": "coarse VALUE/BASIS spans"},
            {"method": "Existing rule-based OCR tagger", "precision": coarse_rule_metrics["precision"], "recall": coarse_rule_metrics["recall"], "F1": coarse_rule_metrics["f1"], "scope": "same tokens, boxes, gold projection"},
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)

        sample_labels = [f"{item['barcode']} · {len(item['tokens'])} OCR tokens" for item in sample_records]
        selected_sample = st.selectbox("Inspect a test image", range(len(sample_records)), format_func=lambda index: sample_labels[index], key="part2_bio_selected_sample")
        selected_record = sample_records[selected_sample]
        selected_prediction = predictions[selected_sample]
        selected_rule_labels = coarse_rules[selected_sample]
        with st.expander("Linked product nutrition context for this evaluated image"):
            show_linked_product_context(str(selected_record["barcode"]), key="bio_linked_product_download")
            st.caption("This is a read-only metadata lookup after inference. The displayed gold BIO tags and metrics still come only from the official test annotations; unmatched examples remain in evaluation.")
        image_col, note_col = st.columns([1.2, 1])
        with image_col:
            overlay = draw_prediction_boxes(selected_record["image"], selected_record["bboxes"], selected_prediction["predicted_labels"])
            st.image(overlay, caption="Green boxes = model-predicted non-O BIO tokens", width="stretch")
        with note_col:
            st.subheader("Extracted model entities")
            decoded = selected_prediction.get("structured_output")
            if decoded is None:
                from part2.layoutlm_bio import decode_bio_entities
                decoded = decode_bio_entities(selected_record["tokens"], selected_prediction["predicted_labels"], barcode=selected_record.get("barcode"), image_path=selected_record.get("image_path"))
            st.dataframe(pd.DataFrame(decoded.get("parsed_fields", []), columns=["nutrient", "basis", "value", "unit", "normalized_value", "normalized_unit", "raw_text", "parse_status"]), width="stretch", hide_index=True)
            if decoded.get("uncertain_fields"):
                st.warning(f"{len(decoded['uncertain_fields'])} predicted spans need review because of ambiguous parsing, missing units, or duplicate fields.")
                st.dataframe(pd.DataFrame(decoded["uncertain_fields"]), width="stretch", hide_index=True)
            with st.expander("Complete structured JSON (raw labels retained)"):
                st.json(decoded)
            st.download_button("Download selected image JSON", json.dumps(decoded, ensure_ascii=False, indent=2), file_name=f"nutrition_{selected_record['barcode']}.json", mime="application/json", key="part2_selected_json")
            st.caption(f"Device: CPU · {selected_prediction.get('runtime', 'local model')} · cache {'hit' if selected_prediction['cache_hit'] else 'miss'} · {selected_prediction['inference_seconds']:.2f} seconds")

        token_rows = []
        for token_index, (token, predicted, truth) in enumerate(zip(selected_record["tokens"], selected_prediction["predicted_labels"], selected_prediction["gold_labels"])):
            token_rows.append({"index": token_index, "OCR token": token, "predicted BIO": predicted, "ground-truth BIO": truth, "rule baseline (coarse)": selected_rule_labels[token_index] if token_index < len(selected_rule_labels) else "O"})
        st.subheader("Token-level predictions vs original annotations")
        st.dataframe(pd.DataFrame(token_rows), width="stretch", hide_index=True, height=440)

        st.subheader("Token-level error analysis")
        diagnostics = prediction_diagnostics(sample_records, predictions)
        with st.expander("Parsing failures, duplicate predictions, units and coverage", expanded=False):
            st.caption("These are diagnostics from actual saved predictions. Parser warnings are not confirmed OCR transcription errors. No uncovered position is silently scored as O.")
            if diagnostics["examples"]:
                st.dataframe(pd.DataFrame([{"reason": reason, "count": count} for reason, count in diagnostics["reason_counts"].items()]), width="stretch", hide_index=True)
                product_table(pd.DataFrame(diagnostics["examples"]))
            else:
                st.info("No parser uncertainty was recorded for these predictions.")
            product_table(pd.DataFrame(diagnostics["coverage_by_image"]))
        error_rows = analyze_bio_errors(sample_records, predictions)
        if error_rows:
            error_frame = pd.DataFrame(error_rows)
            st.bar_chart(error_frame["error_type"].value_counts().rename("token disagreements"))
            st.caption("Evaluation audit: this chart covers the saved official-test predictions and intentionally ignores exploratory country/category filters. Filtered EDA never changes evaluation scores.")
            st.caption("These are actual disagreements between model BIO tags and dataset `ner_tags`. OCR text itself is supplied corpus OCR, not independently re-transcribed truth; OCR errors cannot be confirmed from this comparison alone.")
            product_table(error_frame)
        else:
            st.success("No BIO tag disagreements on the currently displayed sample. This does not certify full-test accuracy.")

        st.subheader("Field comparison against official annotations")
        field_comparisons = [
            {"barcode": record.get("barcode"), **compare_fields_to_gold(record["tokens"], prediction["gold_labels"], prediction["predicted_labels"])}
            for record, prediction in zip(sample_records, predictions)
        ]
        st.caption("Exact normalized nutrient/basis/unit/value/qualifier tuples decoded from aligned corpus ner_tags. This measures annotation decoding, not independent visual transcription. Uncertain gold spans are disclosed and excluded.")
        field_summary = summarize_field_evaluations(field_comparisons)
        field_cols = st.columns(4)
        for column, title, key in zip(field_cols, ["Field precision", "Field recall", "Field F1", "Exact numerical accuracy"], ["field_precision", "field_recall", "field_f1", "exact_value_accuracy"]):
            value = field_summary[key]
            column.metric(title, f"{value:.3f}" if value is not None else "Not established")
        st.caption(f"Micro counts: {field_summary['true_positive']} exact fields / {field_summary['predicted_fields']} predictions / {field_summary['gold_fields']} parseable gold fields. Numerical accuracy: {field_summary['exact_value_matches']}/{field_summary['comparable_value_fields']} on matching nutrient/basis/unit keys, including qualifiers. Excluded uncertain gold spans: {field_summary['gold_uncertain_spans']}; predicted uncertain spans: {field_summary['predicted_uncertain_spans']}. Missing fields reduce recall, not the conditional numerical denominator.")
        st.download_button("Download field and error evaluation", json.dumps({"provenance": "analysis of currently displayed cached or newly computed predictions; no new model latency claimed", "field_summary": field_summary, "fields": field_comparisons, "bio_errors": error_rows, "diagnostics": diagnostics}, ensure_ascii=False, indent=2), file_name="bio_field_error_review.json", mime="application/json", key="part2_review_download")
        product_table(pd.DataFrame([{key: value for key, value in item.items() if not isinstance(value, list)} for item in field_comparisons]))
        with st.expander("Missing and unmatched field tuples"):
            st.json(field_comparisons)

        st.subheader("Separate Open Food Facts API reference check")
        st.caption("This is separate from BIO model evaluation. Product API nutriments are mutable product metadata, not image-verified gold. Only exact unit/basis-compatible values are compared; API-missing fields are excluded rather than counted as errors.")
        if st.button("Fetch/cache API references for up to 10 sample products", key="part2_off_reference_lookup"):
            reference_rows = []
            for sample_record, prediction in zip(sample_records[:10], predictions[:10]):
                barcode = str(sample_record.get("barcode", ""))
                try:
                    reference = fetch_openfoodfacts_product(barcode, cache_dir=ROOT / "data" / "part2_cache" / "openfoodfacts_products")
                    decoded = prediction.get("structured_output")
                    if decoded is None:
                        from part2.layoutlm_bio import decode_bio_entities
                        decoded = decode_bio_entities(sample_record["tokens"], prediction["predicted_labels"], barcode=barcode, image_path=sample_record.get("image_path"))
                    comparison_result = compare_fields_to_off_metadata(decoded, reference)
                    reference_rows.append({"barcode": barcode, "product_name": comparison_result.get("product_name"), "status": comparison_result.get("status", "compared"), "api_reference_fields": comparison_result.get("api_reference_fields_available", 0), "compatible_model_fields": comparison_result.get("model_fields_with_compatible_api_reference", 0), "exact_value_accuracy": comparison_result.get("exact_value_accuracy"), "metadata_tuple_precision": comparison_result.get("field_tuple_precision"), "metadata_tuple_recall": comparison_result.get("field_tuple_recall"), "metadata_tuple_f1": comparison_result.get("field_tuple_f1"), "from_cache": reference.get("from_cache"), "comparisons": comparison_result.get("comparisons", [])})
                except Exception as error:
                    reference_rows.append({"barcode": barcode, "status": f"{type(error).__name__}: {error}", "compatible_fields": 0, "comparisons": []})
            st.session_state["part2_bio_off_comparisons"] = reference_rows
        if st.session_state.get("part2_bio_off_comparisons"):
            st.download_button("Download separate API comparisons", json.dumps({"reference_is_visual_gold": False, "products": st.session_state["part2_bio_off_comparisons"]}, ensure_ascii=False, indent=2), file_name="off_api_reference_comparison.json", mime="application/json", key="part2_api_download")
            product_table(pd.DataFrame([{key: value for key, value in row.items() if key != "comparisons"} for row in st.session_state["part2_bio_off_comparisons"]]))
            with st.expander("Inspect per-product API value comparisons"):
                for comparison_row in st.session_state["part2_bio_off_comparisons"]:
                    st.markdown(f"**{comparison_row['barcode']}** · {comparison_row.get('product_name') or comparison_row['status']}")
                    if comparison_row.get("comparisons"):
                        st.dataframe(pd.DataFrame(comparison_row["comparisons"]), width="stretch", hide_index=True)
                    else:
                        st.caption("No compatible returned reference values for this product.")

    st.subheader("Existing manually reviewed image: separate end-to-end accuracy")
    st.caption("One existing reviewed panel (241013128), not a new held-out benchmark. Manual fields and transcription stay unchanged. Saved review can be inspected without loading any weights.")
    if st.button("Run manual-reference OCR + ONNX review", key="part2_manual_review_run"):
        try:
            with st.spinner("Checking resources, then evaluating the existing manual panel with cached local weights…"):
                model, processor, model_id2label, _ = _cached_openfoodfacts_layoutlm()
                run_manual_reference_review(ROOT, model=model, processor=processor, id2label=model_id2label)
        except Exception as error:
            st.error(f"Manual reference review did not complete: {resource_safe_message(error)}")
    try:
        manual_review = load_manual_reference_review(ROOT)
        if manual_review:
            st.caption(f"Saved review: {manual_review['generated_at_utc']} · OCR cache hit: {manual_review.get('ocr_cache_hit')} · model cache hit: {manual_review.get('model_cache_hit')}. Recorded stage times are not claimed as freshly measured cache-hit latency.")
            if manual_review.get("ocr_error"):
                st.error(manual_review["ocr_error"])
            elif manual_review.get("field_metrics"):
                model_fields = manual_review["field_metrics"]
                st.warning(f"Only {model_fields['true_positive']}/{model_fields['gold_fields']} manual fields were recovered exactly. Conditional exact-value accuracy on {model_fields['comparable_value_fields']} comparable fields is not full-panel accuracy. Parsing failures/uncertain spans: {manual_review['parse_failure_count']}.")
                st.dataframe(pd.DataFrame([{"method": name, **{key: value for key, value in manual_review[name].items() if not isinstance(value, list)}} for name in ["field_metrics", "rule_field_metrics_same_ocr"]]), width="stretch", hide_index=True)
                with st.expander("Actual OCR transcription differences and missing manual fields"):
                    st.write(manual_review["ocr_metrics"])
                    st.caption("CER/WER use raw text, including punctuation and reading order, on this one manual transcription only. Word diffs below are case-folded diagnostics, not gold BIO alignment.")
                    st.dataframe(pd.DataFrame(manual_review["ocr_difference_examples"]), width="stretch", hide_index=True)
                    st.dataframe(pd.DataFrame(model_fields["comparisons"]), width="stretch", hide_index=True)
                    st.text_area("Manual reference text", manual_review["reference_text"], disabled=True)
                    st.text_area("Actual EasyOCR text", manual_review["ocr_text"], disabled=True)
                    st.json(manual_review["structured_output"])
                st.download_button("Download manual-reference review", json.dumps(manual_review, ensure_ascii=False, indent=2), file_name="bio_manual_reference_review.json", mime="application/json", key="part2_manual_download")
        else:
            st.info("No saved manual-reference model evaluation yet. Run it explicitly after the resource safety check.")
    except (ValueError, OSError) as error:
        st.warning(f"Saved manual review unavailable: {error}")

    st.subheader("Unseen image: EasyOCR → pretrained BIO tags → normalized JSON")
    st.caption("EasyOCR uses only already-cached local model files (`download_enabled=False`). Its OCR tokenization differs from the annotated corpus, so no gold BIO metrics are computed for uploads. Ambiguous values stay flagged in JSON.")
    uploaded_label_image = st.file_uploader("Upload one nutrition-label image", type=["jpg", "jpeg", "png", "webp"], key="part2_bio_uploaded_image")
    if uploaded_label_image is not None:
        if uploaded_label_image.size > 15 * 1024 * 1024:
            st.error("Image exceeds the 15 MiB upload safety limit.")
        elif st.button("Run local EasyOCR + ONNX inference", key="run_part2_uploaded_inference"):
            try:
                import hashlib
                upload_dir = ROOT / "data" / "part2_cache" / "uploaded_images"
                upload_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(uploaded_label_image.name).suffix.lower() or ".png"
                upload_digest = hashlib.sha256(uploaded_label_image.getvalue()).hexdigest()
                upload_path = upload_dir / f"{upload_digest}{suffix}"
                if not upload_path.exists():
                    upload_path.write_bytes(uploaded_label_image.getvalue())
                with st.spinner("Running cached EasyOCR and local CPU ONNX BIO inference…"):
                    model, processor, model_id2label, runtime_info = _cached_openfoodfacts_layoutlm()
                    upload_result = predict_uploaded_image(
                        upload_path, model=model, processor=processor, id2label=model_id2label,
                        languages=("fr", "en"),
                        cache_dir=ROOT / "data" / "part2_cache" / "layoutlm_bio",
                        easyocr_model_dir=ROOT / "models" / "easyocr",
                        ocr_cache_dir=ROOT / "data" / "part2_cache" / "ocr",
                    )
                st.session_state["part2_bio_upload_result"] = upload_result
            except Exception as error:
                st.error(f"Upload inference failed: {resource_safe_message(error)}")
    upload_result = st.session_state.get("part2_bio_upload_result")
    if upload_result:
        if upload_result.get("ocr_error"):
            st.error(upload_result["ocr_error"])
        else:
            upload_image_col, upload_json_col = st.columns([1, 1])
            with upload_image_col:
                upload_prediction = upload_result["prediction"]
                st.image(draw_prediction_boxes(upload_result["image"], upload_result["boxes"], upload_prediction["predicted_labels"]), caption="Green boxes = predicted non-O BIO spans", width="stretch")
                st.dataframe(pd.DataFrame({"OCR token": upload_result["tokens"], "EasyOCR confidence": upload_result["ocr_confidences"], "Predicted BIO": upload_prediction["predicted_labels"]}), width="stretch", hide_index=True)
            with upload_json_col:
                st.caption(upload_result["warning"])
                if not upload_result["structured_output"]["parsed_fields"]:
                    st.warning("No unambiguous nutrition fields were extracted. Review the OCR text, image orientation and uncertain spans; successful execution does not imply successful extraction.")
                st.json(upload_result["structured_output"])
                st.download_button("Download extracted JSON", json.dumps(upload_result["structured_output"], ensure_ascii=False, indent=2), file_name="nutrition_extraction.json", mime="application/json", key="part2_upload_json_download")
    else:
        st.caption("No model predictions are shown until inference completes. Metrics are never filled with model-card scores or illustrative values.")

# ── Findings and provenance ─────────────────────────────────────────────────
if active_section == "Results":
    st.subheader("Results · saved official-test evaluation")
    st.info("These are fixed saved model results, not filtered EDA. Country/category filters and the selected EDA dataset never change the official test cohort or gold labels. No model is loaded on this page.")
    results_path = ROOT / "data/part2_cache/layoutlm_bio/latest_test_bundle.json"
    if results_path.exists():
        result_bundle = json.loads(results_path.read_text(encoding="utf-8"))
        result_rows = [{"task": task, **result_bundle[key]} for task, key in [("Fine-grained BIO model", "fine_grained_metrics"), ("Coarse model projection", "coarse_model_metrics"), ("Coarse rule baseline", "coarse_rule_metrics")]]
        st.caption(f"Official test subset: {result_bundle['sample_count']} images. Full-image token coverage and gold entity coverage are included; not a full-test benchmark.")
        st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)
        result_examples = pd.DataFrame([{"barcode": sample["barcode"], "tokens": sample["token_count"], "covered_tokens": sample["covered_token_count"], "inference_seconds_recorded": sample["inference_seconds"]} for sample in result_bundle["samples"]])
        product_table(result_examples)
        st.caption("For original images, token-level predictions, normalized JSON, error analysis and opt-in inference, use BIO Entity Tagging.")
    else:
        st.info("No saved official-test results are available. Run opt-in inference in BIO Entity Tagging.")

if active_section == "Data Understanding":
    st.subheader("Data Understanding")
    st.markdown(DATASET_DESCRIPTIONS)
    st.caption("Part 1 currently uses the official OFF product TSV export and local 50K EDA extracts; product-database is the corresponding product-record resource, not a claim that this dashboard downloaded that Hugging Face version. The table-detection dataset is optional and is not loaded or used for current model metrics.")
    enriched_summary_path = ROOT / "reports" / "eda" / "enriched_summary.json"
    if enriched_summary_path.exists():
        saved_summary = json.loads(enriched_summary_path.read_text(encoding="utf-8"))
        st.json({key: saved_summary.get(key) for key in ["source_url", "acquired_at_utc", "row_limit", "selection_note", "columns_examined", "columns", "rows_examined"] if key in saved_summary})
    else:
        st.info("Saved enriched summary not found. The dashboard calculates the metrics currently visible from the selected data source.")
    st.markdown('<div class="info-strip">The broad project sample contains the first 50,000 rows, not a random sample. The separate common subset is selected by annotation availability. Neither should be presented as representative of all Open Food Facts products; choosing the common subset does not replace or rewrite Part 1.</div>', unsafe_allow_html=True)

    code_col = "code" if "code" in frame else None
    unique_codes = int(frame[code_col].nunique(dropna=True)) if code_col else None
    blank_names = int((~_nonempty(frame["product_name"])).sum()) if "product_name" in frame else None
    st.subheader("Data quality snapshot")
    st.write({
        "rows_in_filtered_view": int(len(frame)),
        "retained_fields": int(frame.shape[1]),
        "unique_product_codes": unique_codes,
        "duplicate_code_rows": int(frame[code_col].duplicated(keep=False).sum()) if code_col else None,
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "blank_product_names": blank_names,
        "rows_with_any_structured_nutrition": int(summary.get("rows_with_any_structured_nutrition") or 0),
    })

    st.subheader("Annotation and model readiness")
    st.write({
        "nutrition_image_metadata_candidates": int(candidate_mask.sum()),
        "record_level_gold_examples": int(verified_n),
        "word_box_bio_annotations": 0,
        "supervised_training_ready": False,
        "why": "Candidate metadata does not verify image text. One manually transcribed panel is not enough for model training or a meaningful benchmark.",
    })
    st.markdown("**Recommended order:** visually review the pilot queue → transcribe and annotate words/boxes/BIO tags → adjudicate disagreements → create product-grouped train/validation/test splits → run OCR baseline → evaluate against held-out gold → consider LayoutLMv3 only if data and compute are sufficient.")

    exclusions_path = ROOT / "reports" / "eda" / "excluded_source_columns.csv"
    if exclusions_path.exists():
        exclusions = pd.read_csv(exclusions_path)
        st.subheader(f"Excluded source-column inventory ({len(exclusions)} columns)")
        st.caption("Schema inventory: counts columns in the original export, not product records. Country/category filters do not apply to this explicitly fixed schema chart.")
        if "category" in exclusions:
            grouped = exclusions.groupby("category", as_index=False).size().sort_values("size", ascending=False)
            st.dataframe(grouped, width="stretch", hide_index=True)
        with st.expander("Inspect every excluded column and rationale"):
            st.dataframe(exclusions, width="stretch", hide_index=True)
            st.download_button("Download excluded-column inventory", exclusions.to_csv(index=False).encode("utf-8"), file_name="excluded_source_columns.csv", mime="text/csv", key="download_exclusions")

    st.subheader("Download EDA artifacts")
    findings_path = ROOT / "reports" / "eda_findings_template.md"
    if findings_path.exists():
        findings_text = findings_path.read_text(encoding="utf-8")
        with st.expander("Read the complete written EDA findings"):
            st.markdown(findings_text)
        st.download_button("Download written EDA findings", findings_text.encode("utf-8"), file_name="eda_findings.md", mime="text/markdown", key="download_findings")
    artifact_paths = [
        ROOT / "reports" / "eda" / "enriched_summary.json",
        ROOT / "reports" / "eda" / "enriched_nutrition_coverage.csv",
        ROOT / "reports" / "eda" / "nutrition_statistics.csv",
        ROOT / "reports" / "eda" / "nutrition_correlations.csv",
        ROOT / "reports" / "eda" / "country_counts.csv",
        ROOT / "reports" / "eda" / "image_resolution_audit.csv",
        ROOT / "reports" / "eda" / "manual_verification_log.csv",
        ROOT / "reports" / "eda" / "excluded_source_columns.csv",
    ]
    for artifact in artifact_paths:
        if artifact.exists():
            data = artifact.read_bytes()
            mime = "application/json" if artifact.suffix == ".json" else "text/csv"
            st.download_button(f"Download {artifact.name}", data, file_name=artifact.name, mime=mime, key=f"download_{artifact.name}")

# ── Data table and exports ─────────────────────────────────────────────────
if active_section == "View Raw Data":
    st.subheader("Filtered product records")
    search = st.text_input("Search product, brand, or code", placeholder="e.g. chocolate, brand, 241013128")
    table = frame.copy()
    if search:
        search_cols = [column for column in ["code", "product_name", "brands", "categories"] if column in table]
        search_mask = pd.Series(False, index=table.index)
        for column in search_cols:
            search_mask |= table[column].astype("string").str.contains(search, case=False, na=False, regex=False)
        table = table.loc[search_mask]
    display_limit = st.slider("Maximum rows to display", min_value=50, max_value=1000, value=200, step=50)
    st.caption(f"Showing {min(len(table), display_limit):,} of {len(table):,} matching rows.")
    product_table(table.head(display_limit))
    st.download_button("Download filtered records", table.to_csv(index=False).encode("utf-8"), file_name="filtered_openfoodfacts_records.csv", mime="text/csv")
    st.subheader("Merged Part 1 + Part 2 table · separate linked view")
    st.caption("One row per annotated image, left-joined to product metadata. All unmatched rows are retained when filters are clear. Tokens stay in a separate annotation table to avoid multiplying product rows.")
    try:
        raw_products, raw_images, _ = get_linked_data()
        if selected_countries or selected_categories:
            raw_codes = set(filtered_frame(products_for_eda(raw_products), selected_countries, selected_categories)["code"])
            raw_images = raw_images.loc[raw_images["barcode"].isin(raw_codes)]
        merged_table = merged_image_products(raw_products, raw_images)
        if search:
            raw_mask = pd.Series(False, index=merged_table.index)
            for name in ["barcode", "product_name", "brand", "category"]:
                raw_mask |= merged_table[name].astype("string").str.contains(search, case=False, na=False, regex=False)
            merged_table = merged_table.loc[raw_mask]
        st.caption(f"Merged image/product rows after filters and search: {len(merged_table):,}.")
        product_table(merged_table.head(display_limit), images=raw_images)
        st.download_button("Download merged linked table", merged_table.to_csv(index=False).encode("utf-8"), file_name="merged_products_images.csv", mime="text/csv", key="raw_merged_download")
    except (OSError, ValueError, ImportError) as error:
        st.info(f"Merged table unavailable: {error}")

st.divider()
st.caption("Dashboard uses local sample data or a CSV/TSV you upload. EDA counts update with sidebar filters. EasyOCR and pretrained BIO inference are opt-in in the BIO Entity Tagging view and run on CPU using local weights.")