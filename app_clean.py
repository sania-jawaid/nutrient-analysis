"""Read-only, cache-only nutrition dashboard. Never imports the original app.

Run with Streamlit. Importing this module only defines helpers; no UI or worker runs.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nutrition_extraction.dashboard_data import filter_products, label_set, numeric_modes
from nutrition_extraction.linked_data import annotation_preview, load_linked, products_for_eda

LINKED = ROOT / "outputs/part1_part2_integration"
FULL = ROOT / "outputs/full_cohort_bio"
BUNDLE = ROOT / "data/part2_cache/layoutlm_bio/latest_test_bundle.json"
# Display compatibility contract, deliberately no import of the inference module.
CACHE_SCHEMA = "5-adaptive-sliding-structured"
SECTIONS = ["View Raw Data", "Data Overview", "Data Understanding", "Nutrients",
            "Data Quality", "Part 2 Dataset Info", "BIO Entity Tagging", "Results"]
DATASETS = ["50K Dataset", "Part1 + Part2 Overlapped Dataset"]
SOURCES = ["Full matched cohort", "Seeded 20 test images"]

CSS = """
<style>
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stHeader"],
section[data-testid="stSidebar"] {background:#000 !important; color:#E8EEF5;}
.block-container {max-width:1320px; padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3 {color:#E8EEF5; letter-spacing:-.03em;}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] {color:#C5D0DC;}
.clean-hero {border:1px solid #293648; border-radius:18px; padding:1.3rem 1.5rem;
 background:linear-gradient(115deg,#102A43,#0d252b); margin-bottom:1.6rem;}
.clean-hero small {color:#8FE3D0; letter-spacing:.15em;}
.clean-hero h1 {font-size:2rem; margin:.3rem 0;}
.clean-hero p {color:#B7C4D3; margin:0;}
[data-testid="stMetric"] {background:#171F2B; border:1px solid #293648;
 border-radius:14px; padding:1rem; margin-bottom:1rem;}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#E8EEF5;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="popover"] {background:#202B38 !important; color:#F0F5FA !important;}
[data-baseweb="select"] input, [data-baseweb="select"] span,
[data-baseweb="popover"] * {color:#F0F5FA !important;}
[data-testid="stRadio"] label {color:#C5D0DC !important;}
[data-testid="stExpander"] {border-color:#293648;}
button {border-radius:8px !important;}
</style>
"""


def fingerprint(paths):
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


@st.cache_data(show_spinner=False, max_entries=2)
def load_inventory(stamp):
    del stamp
    return load_linked(LINKED)


@st.cache_data(show_spinner=False, max_entries=2)
def load_sample(stamp):
    del stamp
    return pd.read_csv(ROOT / "data/openfoodfacts_eda_enriched.csv",
                       dtype={"code": "string"}, low_memory=False)


@st.cache_data(show_spinner=False, max_entries=32)
def original_annotations(image_key, stamp):
    del stamp
    # Head limit exceeds the selected record length; never reconstruct gold tags.
    return annotation_preview(LINKED, image_key, limit=1_000_000).sort_values("token_index")


def label_choices(frame, column):
    return sorted(set().union(*(label_set(v) for v in frame[column].dropna()))) if column in frame else []


def scoped_images(images, frame, filters_active):
    """Default full annotation inventory; exact barcode intersection only on filtering."""
    if not filters_active:
        return images.copy()
    return images.loc[images["barcode"].isin(frame["code"].dropna())].copy()


def numeric_nutrients(frame):
    names = [c for c in frame if c.endswith(("_100g", "_100ml", "_serving"))]
    return frame[names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def nutrient_example_facts(frame):
    """Illustrative metadata facts only; reuse the dashboard's finite-number rules."""
    numeric = numeric_nutrients(frame)
    energy = numeric.reindex(columns=["energy_100g", "energy-kcal_100g"])
    carbs = numeric.reindex(columns=["carbohydrates_100g"])["carbohydrates_100g"]
    names = frame.get("product_name", pd.Series(pd.NA, index=frame.index, dtype="string"))
    codes = frame.get("code", pd.Series(pd.NA, index=frame.index, dtype="string"))
    codes = codes.dropna().astype("string").str.strip()
    codes = codes.loc[codes.ne("")]
    micronutrients = [c for c in numeric if c.startswith(
        ("vitamin-", "potassium_", "calcium_", "iron_"))]
    return {
        "rows": len(frame),
        "nutrient_fields": len(numeric.columns),
        "energy_count": int(energy["energy_100g"].count()),
        "energy_mean": energy["energy_100g"].mean(),
        "energy_median": energy["energy_100g"].median(),
        "energy_max": energy["energy_100g"].max(),
        "carbs_count": int(carbs.count()),
        "carbs_median": carbs.median(),
        "carbs_above_100": int(carbs.gt(100).sum()),
        "energy_correlation": energy.corr(min_periods=3).iloc[0, 1],
        "energy_pairs": int(energy.notna().all(axis=1).sum()),
        "exact_duplicates": int(frame.duplicated().sum()),
        "duplicate_codes": int(codes.duplicated().sum()),
        "missing_names": int((names.isna() | names.astype("string").str.strip().eq("").fillna(False)).sum()),
        "any_nutrition": int(numeric.notna().any(axis=1).sum()),
        "micronutrient_missing": {c: int(numeric[c].isna().sum()) for c in micronutrients},
    }


@st.cache_data(show_spinner=False, max_entries=2)
def sample_nutrient_examples(stamp):
    """Only the saved enriched sample, never the active frame or full export."""
    return nutrient_example_facts(load_sample(stamp))


def render_sample_nutrient_examples():
    with st.expander("50K sample examples · not changed by filters", expanded=False):
        path = ROOT / "data/openfoodfacts_eda_enriched.csv"
        if not path.is_file():
            st.caption("The saved enriched sample is unavailable; no fixed examples are asserted.")
            return
        facts = sample_nutrient_examples(fingerprint([path]))
        rows = facts["rows"]
        coverage = 100 * facts["any_nutrition"] / rows if rows else 0
        st.caption(f"Source: saved data/openfoodfacts_eda_enriched.csv · {rows:,} rows · "
                   "first export rows, not a random sample. Independent of the selected dataset and filters; "
                   "recomputed when this saved file changes. No full-export scan.")
        st.markdown(
            f"- **Distribution:** energy_100g (kJ per 100 g, or 100 ml for liquids) has "
            f"{facts['energy_count']:,} finite values: median **{facts['energy_median']:,.0f}**, "
            f"mean **{facts['energy_mean']:,.2f}**, maximum **{facts['energy_max']:,.0f}**. "
            "The mean exceeds the median; the extreme upper tail warrants unit/basis and source checks, not automatic correction.\n"
            f"- **Plausibility:** carbohydrates_100g has {facts['carbs_count']:,} finite values and "
            f"median **{facts['carbs_median']:g} g**; **{facts['carbs_above_100']:,}** values exceed 100. "
            "For a genuine g-per-100-g record, >100 g is implausible; verify basis, units and entry errors first.\n"
            f"- **Association:** energy-kcal_100g versus energy_100g has Pearson "
            f"**r = {facts['energy_correlation']:.4f}** across **{facts['energy_pairs']:,}** finite pairs. "
            "The same energy expressed in consistent units would be almost perfectly linear; "
            "departures suggest reviewing units, basis, outliers or inconsistent entries, not causation.\n"
            f"- **Record quality:** **{facts['exact_duplicates']:,}** exact repeated rows; "
            f"**{facts['duplicate_codes']:,}** repeated nonblank product-code rows (extra copies after the first); "
            f"**{facts['missing_names']:,}** missing product names, including whitespace-only names.\n"
            f"- **Any-nutrient coverage:** **{facts['any_nutrition']:,} / {rows:,} ({coverage:.2f}%)** "
            f"have at least one finite numeric value across the **{facts['nutrient_fields']}** nutrient fields "
            "recognized by numeric_nutrients (_100g, _100ml, _serving). Numeric strings are parsed; "
            "invalid text, missing values and ±infinity are excluded, but zero and finite outliers count. "
            "This is not complete-panel coverage or proof of correctness."
        )
        vitamin_d_missing = facts["micronutrient_missing"].get("vitamin-d_100g")
        if vitamin_d_missing is not None:
            st.markdown(f"- **Micronutrient missingness:** vitamin-d_100g lacks a finite numeric value in "
                        f"**{vitamin_d_missing:,} / {rows:,}** rows. Vitamin/mineral coverage varies by field; "
                        "missing does not mean absent from the food or equal to zero.")


def chart(fig, height=350):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000",
                      font_color="#C5D0DC", margin=dict(l=12, r=12, t=40, b=20), height=height,
                      colorway=["#64B5F6", "#62D6BD", "#F2B84B"])
    st.plotly_chart(fig, width="stretch")


def table(frame, headings=None):
    # Display labels only: raw data and helper column names stay unchanged.
    st.dataframe(frame, hide_index=True, width="stretch", column_config=headings or {})


def cards(values):
    for col, (label, value) in zip(st.columns(len(values)), values):
        col.metric(label, value)


def choose_images(images, prefix):
    if images.empty:
        st.info("No Part 2 images match the active product filters.")
        return None
    split = st.selectbox("Split", ["All"] + sorted(images["dataset_split"].unique().tolist()), key=prefix + "_split")
    st.caption("Split = dataset group. Train is the learning group; test is the separate checking group. All shows both.")
    rows = images if split == "All" else images.loc[images["dataset_split"].eq(split)]
    if rows.empty:
        st.info("No images in this split.")
        return None
    key = st.selectbox("Image", rows["image_key"].tolist(), key=prefix + "_image")
    return rows.loc[rows["image_key"].eq(key)].iloc[0]


def render_original(images, prefix):
    row = choose_images(images, prefix)
    if row is None:
        return
    st.caption(f"Original image words and labels · {row['image_key']} · barcode {row['barcode']}. "
               "OCR means text read from an image; these words were supplied by the dataset, not read again here.")
    annotations = original_annotations(row["image_key"], fingerprint([LINKED / "integrated_annotations.parquet"]))
    table(annotations)
    st.caption("Column key: image_key = image ID; token_index = word position; ocr_token = supplied word; "
               "bounding_box = word location (0–1000 scale); gold_bio_id = label number; "
               "gold_bio_label = reference word label. Original names and values are unchanged.")
    st.caption("Gold/reference means the dataset's supplied labels, not independently verified labels checked against the image.")
    # Links only: no HTTP calls or automatic remote image loads.
    if pd.notna(row.get("image_url")):
        st.link_button("Open source image (external)", row["image_url"])


def render_raw(frame, images):
    st.subheader("Product data")
    st.caption(f"{len(frame):,} filtered product rows. Preview limited to 200; download contains every filtered row.")
    table(frame.head(200))
    st.caption("Column key: code = barcode; product_name = product name; countries/categories = supplied labels; "
               "serving_size = stated portion. Nutrient endings: _100g = per 100 g (sometimes 100 ml for liquids); "
               "_100ml = per 100 ml; _serving = per portion. Blank means unknown, not zero. Source field names stay unchanged.")
    st.download_button("Download filtered products", frame.to_csv(index=False).encode("utf-8"),
                       file_name="clean_filtered_products.csv", mime="text/csv")
    st.subheader("Original words and reference labels")
    render_original(images, "raw")


def render_overview(frame, images):
    numeric = numeric_nutrients(frame)
    cards([("Product rows", f"{len(frame):,}"), ("Unique barcodes", f"{frame['code'].nunique():,}"),
           ("Part 2 images in scope", f"{len(images):,}")])
    st.caption("Product rows are records; unique barcodes count different product IDs. Part 2 contains images with word labels.")
    for column, title in [("countries", "Country labels"), ("categories", "Category labels")]:
        counts = frame[column].dropna().map(lambda v: sorted(label_set(v))).explode().value_counts().head(12)
        if not counts.empty:
            chart(px.bar(x=counts.values, y=counts.index, orientation="h", title=title,
                         color_discrete_sequence=["#64B5F6" if column == "countries" else "#62D6BD"],
                         labels={"x": "Product rows with this label", "y": "Label"}))
    st.caption("A product can have several labels, so counts can exceed the number of products. A country label does not prove where a product was made.")
    if len(frame) and not numeric.empty:
        coverage = numeric.notna().mean().sort_values(ascending=False).head(12).rename("fraction").reset_index()
        chart(px.bar(coverage, x="index", y="fraction", title="How often nutrient values are available",
                     color_discrete_sequence=["#F2B84B"],
                     labels={"index": "Nutrient field", "fraction": "Share of selected product rows"}))
        st.caption("1 means every selected row has a usable number; 0.5 means half do. Only the 12 most available fields are shown.")


def render_understanding(frame, images, dataset):
    st.markdown("""
### What does one row mean?
- **Product:** one barcode record. The 50K Dataset takes the first 50,000 export rows, not a random sample.
- **Image:** one nutrition image with labels. One product may have several images.
- **Token:** one supplied OCR word. OCR reads text from images and can make mistakes. Do not count nutrition again for every word.

### Where does it come from?
- **Part 1:** product details from Open Food Facts.
- **Part 2:** nutrition images, OCR words and reference labels from its nutrient-detection-layout dataset.
- **Overlapped dataset:** barcodes matched against the complete saved Part 1 export, not just the 50K sample.
- A barcode match does not prove the same packaging, date, recipe or nutrient values. Eight Part 2 images have no match.

### Per 100 g or per serving?
- **Basis** means the amount of food a value describes: per100g = per 100 g; perServing = per portion.
- Example: 10 g fat per 100 g and 3 g fat per 30 g serving describe different amounts of food.
- `_100g` can also mean per 100 ml for liquids in Open Food Facts. `_100ml` explicitly means per 100 ml; `_serving` means per portion.
- Do not compare or combine these without checking the package basis and portion size. Missing means unknown, not zero.

### What do filters change?
- Within a filter, any selected label can match. Between country and category, both filters must match.
- With filters, Part 2 shows only images with an exact remaining barcode match. Leading zeros are kept.
- Without filters, Part 2 shows all original images, whichever product dataset is selected.
- Results keep the chosen saved image group and fixed target count; product filters do not change scores.
""")
    with st.expander("Details · units and dataset groups", expanded=False):
        st.markdown("- Fat, protein and similar nutrients are usually in grams. Energy uses kJ or kcal as named; "
                    "the older energy_100g field uses kJ. Check source definitions for vitamin/mineral units.\n"
                    "- Train/test are the dataset's original learning/checking groups; this dashboard does not reshuffle them.\n"
                    "- No units are converted and no reference labels are rewritten here.")
    cards([("Selected dataset", dataset), ("Filtered product rows", f"{len(frame):,}"),
           ("Part 2 images in scope", f"{len(images):,}")])


def render_nutrients(frame):
    st.caption("Understand typical nutrition values, unusual records and relationships in the active dataset after filters. Fixed 50K examples are labelled separately below.")
    numeric = numeric_nutrients(frame)
    available = numeric.columns[numeric.notna().any()].tolist()
    if not available:
        st.info("No numeric nutrient observations in the filtered products.")
        return
    st.subheader("How nutrient values vary")
    nutrient = st.selectbox("Nutrient field (source unit and basis)", available)
    st.caption("Choose a nutrient. Check its unit and whether it is per 100 g, per 100 ml or per serving before comparing values.")
    kind = st.selectbox("Chart type", ["Histogram", "Box plot", "Violin plot"])
    st.caption("Histogram: counts in value ranges. Box plot: middle values and spread. Violin plot: wider areas mean more values.")
    if kind == "Histogram":
        fig = px.histogram(numeric, x=nutrient, nbins=35, color_discrete_sequence=["#64B5F6"])
    elif kind == "Box plot":
        fig = px.box(numeric, y=nutrient, color_discrete_sequence=["#62D6BD"])
    else:
        fig = px.violin(numeric, y=nutrient, box=True, color_discrete_sequence=["#B699FF"])
    fig.update_layout(title="Spread of available nutrient values")
    chart(fig)
    st.markdown("**What this chart tells you**\n"
                "- See which values are common, how widely they vary, and whether a few values are unusually large.\n"
                "- Compare the average with the median below: a much higher average can mean a few high values pull it upward.\n"
                "- Extreme values may be real or reflect wrong units, decimal mistakes or a serving-size mix-up. Check the source before calling them errors.")
    st.subheader("Nutrient summary")
    statistics = numeric[available].describe().T.reset_index(names="source_field")
    def mode_text(name):
        mode = numeric_modes(numeric[name])
        if mode["status"] == "all_values_unique":
            return "No repeated mode (all values unique)"
        if not mode["values"]:
            return "No numeric values"
        return ", ".join(f"{v:g}" for v in mode["values"])
    statistics["Mode"] = statistics["source_field"].map(mode_text)
    table(statistics, {"source_field": "Nutrient field", "count": "Values available", "mean": "Average",
                       "std": "Spread (standard deviation)", "min": "Lowest", "25%": "Lower quarter (25%)",
                       "50%": "Middle (median)", "75%": "Upper quarter (75%)", "max": "Highest"})
    st.caption("Mode is the most frequent observed value; all tied modes are shown. Unique-only values have no repeated mode.")
    st.markdown("**What this table tells you**\n"
                "- **Values available** counts usable measurements, not every product. Missing values are unknown, not zero.\n"
                "- **Average, median and Mode** describe typical values in different ways. **Spread** shows how much values vary.\n"
                "- **25% and 75%** mark the middle half of the values; **Lowest and Highest** help spot records to check.\n"
                "- A product with one nutrient value does not necessarily have a complete nutrition panel. Vitamins and other micronutrients may have many missing values. Use Data Quality to check missing fields, duplicate codes/rows and missing names.")
    st.subheader("Correlation")
    st.caption("Do nutrients tend to rise or fall together? Near +1: rise together; near −1: move in opposite directions; near 0: little straight-line relationship.")
    selected = st.multiselect("Fields to compare", available, default=available[:6])
    if len(selected) >= 2:
        chart(px.imshow(numeric[selected].corr(min_periods=3), zmin=-1, zmax=1,
                        aspect="auto", text_auto=".2f",
                        color_continuous_scale="RdBu_r", title="How nutrient values move together (Pearson correlation)"),
              height=max(650, 50 * len(selected)))
        st.markdown("**What this graph tells you**\n"
                  "- Positive values mean two nutrients often rise together; negative values mean they often move in opposite directions. Near zero means little straight-line relationship.\n"
                  "- A relationship does not prove that one nutrient causes another, or that the data is correct. Food types, missing values, extremes and inconsistent units can change the pattern.\n\n"
                  "**Example: sugars and carbohydrates**\n"
                  "Sugars are part of carbohydrates. An illustrative product A has **20 g carbohydrates, including 5 g sugars**; product B has **60 g carbohydrates, including 35 g sugars**, on the same food basis. "
                  "These examples rise together, but two examples do not establish the dataset's correlation. Rice can have high carbohydrates from starch and little sugar, so the relationship is not necessarily strong or perfect across products.")
    st.caption("Only usable numbers are counted; missing or invalid values are not filled in, and extreme values are not cut off. "
               "Units and food amounts stay unchanged. A relationship does not prove cause or correct nutrition data.")
    with st.expander("Details · how correlation is calculated", expanded=False):
        st.caption("Each pair uses rows with both values present (at least three pairs). Different cells may use different rows. "
                   "A blank cell means there is not enough usable variation or data to calculate it.")
    st.info("How this helps Part 2: these findings help check whether values extracted from packaging are plausible and comparable. They do not directly improve BIO tagging or prove model accuracy; that needs separate annotated-image evaluation.")
    render_sample_nutrient_examples()


def quality_tables(frame):
    present = frame.apply(lambda col: col.notna() & col.astype("string").str.strip().ne("").fillna(False))
    missing = (~present).sum().rename("missing_rows").to_frame()
    missing["missing_fraction"] = missing["missing_rows"] / len(frame) if len(frame) else np.nan
    numeric = numeric_nutrients(frame)
    anomalies = []
    mass = {"fat", "saturated-fat", "trans-fat", "carbohydrates", "sugars", "fiber", "proteins", "salt", "sodium"}
    for name in numeric:
        raw = frame[name]
        values = numeric[name]
        rule = name.endswith("_100g") and name.removesuffix("_100g") in mass
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        iqr_flags = values.lt(q1 - 1.5 * iqr) | values.gt(q3 + 1.5 * iqr)
        anomalies.append(dict(source_field=name, negative=int(values.lt(0).sum()),
                      iqr_outliers=int(iqr_flags.sum()),
                              above_100g=int(values.gt(100).sum()) if rule else None,
                              invalid_numeric=int((present[name] & values.isna()).sum())))
    return missing.reset_index(names="source_field"), pd.DataFrame(anomalies)


def render_quality(frame):
    missing, anomalies = quality_tables(frame)
    ids = frame["code"].dropna().astype("string").str.strip()
    ids = ids.loc[ids.ne("")]
    cards([("Repeated barcode rows", f"{ids.duplicated().sum():,}"),
           ("Exact repeated rows", f"{frame.duplicated().sum():,}"),
           ("Missing barcode rows", f"{len(frame) - len(ids):,}")])
    st.subheader("Missing values")
    ordered = missing.sort_values("missing_fraction", ascending=False)
    chart(px.bar(ordered.head(25), x="missing_fraction", y="source_field", orientation="h",
                 color="missing_fraction", color_continuous_scale="Sunset",
                 title="Fields with the most missing values · top 25", hover_data=["missing_rows"],
                 labels={"missing_fraction": "Share missing", "source_field": "Field", "missing_rows": "Rows missing"}), height=650)
    table(ordered, {"source_field": "Field", "missing_rows": "Rows missing", "missing_fraction": "Share missing"})
    st.caption("Missing includes empty cells and blank text. A share of 1 means every row is missing. Repeated counts above count extra copies after the first.")
    st.subheader("Numbers worth checking")
    table(anomalies, {"source_field": "Nutrient field", "negative": "Below zero", "iqr_outliers": "Unusually low/high (IQR)",
                      "above_100g": "Over 100 g per 100 g", "invalid_numeric": "Not a usable number"})
    st.markdown("- **IQR:** the spread of the middle half of the values. Values far outside it are flagged, not proven wrong.\n"
                "- **Over 100 g:** checked only for named gram-based nutrients per 100 g, not energy, vitamins or per-serving fields.\n"
                "- Nothing is repaired or removed. These checks do not verify the package or image.")
    with st.expander("Details · the IQR rule", expanded=False):
        st.caption("Q1 is the lower quarter (25%); Q3 is the upper quarter (75%). IQR = Q3 − Q1. "
                   "Flags fall below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR using selected usable values. "
                   "Even when IQR is zero, a flag is only a reason to check. Invalid numbers include text and infinity.")


def render_part2(images, inventory):
    cards([("Images in scope", f"{len(images):,}"), ("Products in scope", f"{images['barcode'].nunique():,}"),
           ("Original words in scope", f"{images['token_count'].sum():,}")])
    st.caption(f"All original Part 2 data: {len(inventory):,} images. One product can have several images; each image has OCR words and reference labels.")
    table(images.groupby("dataset_split", as_index=False).agg(images=("image_key", "size"), tokens=("token_count", "sum")),
          {"dataset_split": "Dataset group (split)", "images": "Images", "tokens": "Supplied words"})
    render_original(images, "part2")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_results(source):
    """Read one published snapshot; do not touch locks, predictions, or worker state."""
    if source == SOURCES[0]:
        if not (FULL / "summary.json").exists():
            return None, pd.DataFrame(), None
        # Summary and CSV are individually atomic, not a cross-file transaction.
        for _ in range(3):
            summary = read_json(FULL / "summary.json")
            if summary.get("inference_schema_version") != CACHE_SCHEMA:
                raise ValueError("Full-cohort cache schema is incompatible; refusing to display scores.")
            try:
                rows = pd.read_csv(FULL / "image_results.csv", dtype={"barcode": "string"})
            except pd.errors.EmptyDataError:
                rows = pd.DataFrame()
            if summary == read_json(FULL / "summary.json") and len(rows) == summary["completed_images"]:
                return summary, rows, None
        raise ValueError("Checkpoint publication is in progress. Refresh the page to read a consistent snapshot.")
    if not BUNDLE.exists():
        return None, pd.DataFrame(), None
    bundle = read_json(BUNDLE)
    if bundle.get("inference_schema_version") != CACHE_SCHEMA:
        raise ValueError("Seeded cache schema is incompatible; refusing to display scores.")
    split = bundle["split"]
    rows = pd.DataFrame([dict(image_key=f"{split}:{s['source_index']}", barcode=str(s["barcode"]), split=split)
                         for s in bundle["samples"]])
    target = int(bundle["sample_count"])
    summary = dict(completed_images=len(rows), target_images=target, failed_images=0,
                   pending_images=max(0, target - len(rows)), status="cached_subset",
                   target_splits={split: target}, metrics_by_split={split: dict(images=len(rows), **bundle["fine_grained_metrics"])},
                   updated_at_utc="Saved seeded subset; not the full test split")
    return summary, rows, bundle


def metric_rows(summary):
    result = []
    for split in ["test", "train"]:
        m = summary.get("metrics_by_split", {}).get(split, {})
        images = m.get("images", 0)
        tp, pred, gold = (m.get(k) for k in ["true_positive_spans", "predicted_spans", "gold_spans"])
        result.append(dict(split=split, processed_images=images, target_images=summary["target_splits"].get(split),
                           precision=m.get("precision") if images else None,
                           recall=m.get("recall") if images else None, F1=m.get("f1") if images else None,
                           token_accuracy=m.get("token_accuracy") if images else None,
                           token_support=m.get("token_support") if images else None,
                           true_positives=tp if images else None,
                           false_positives=pred - tp if images and pred is not None and tp is not None else None,
                           false_negatives=gold - tp if images and gold is not None and tp is not None else None,
                           gold_span_support=gold if images else None))
    return pd.DataFrame(result)


def render_status(summary):
    cards([("Completed images", f"{summary['completed_images']:,}"),
           ("Pending images", f"{summary['pending_images']:,}"),
           ("Failed images", f"{summary['failed_images']:,}"), ("Run status", summary["status"])])
    done, target = summary["completed_images"], summary["target_images"]
    st.progress(min(1., done / target) if target else 0., text=f"{done:,} / {target:,} target images saved")
    status_text = {
        "running": "The last save said running; it does not prove a worker is still active.",
        "paused_limit": "Paused at the run's image limit.",
        "paused_low_memory": "Paused because there was not enough free memory.",
        "paused_model_load_error": "Paused because the model could not be loaded in the earlier run.",
        "interrupted": "The earlier run was interrupted before finishing.",
        "complete": "All target images have saved results.",
        "incomplete_with_failures": "The run ended with some images that failed.",
        "cached_subset": "Showing a saved small test group, not the full test dataset.",
    }
    st.caption(status_text.get(summary["status"], "Showing the status saved by the earlier run; no live worker check is made."))
    st.caption("Completed = saved results. Pending = not yet completed and not listed as failed. "
               "Failed = an attempt ended with an error; it is separate from pending.")
    st.caption(f"Last saved: {summary['updated_at_utc']}. This page only reads saved files. Refresh to see newer saves; it cannot start or resume a run.")
    if done < target:
        st.warning("Partial results: scores cover completed images only. Pending and failed images are not scored as zero. "
                   "Both remain in the fixed target count used for image progress.")


def checked_prediction(payload, row, annotations):
    """Refuse mismatched caches; never refresh/rewrite their gold annotations."""
    pred = payload.get("prediction", payload)
    tokens = payload["tokens"]
    if str(payload["barcode"]) != str(row["barcode"]):
        raise ValueError("Cached barcode does not match the selected original image.")
    if (tokens != annotations["ocr_token"].tolist()
            or payload["bboxes"] != [list(box) for box in annotations["bounding_box"]]
            or pred.get("gold_labels") != annotations["gold_bio_label"].tolist()):
        raise ValueError("Cached tokens, boxes or gold differ from original annotations; display refused, gold not rewritten.")
    mask = pred.get("coverage_mask")
    if (len(pred.get("predicted_labels", [])) != len(tokens) or mask is None
            or len(mask) != len(tokens) or not all(mask) or pred.get("covered_token_count") != len(tokens)):
        raise ValueError("Cached prediction has incomplete or unverified token alignment/coverage.")
    return pred


def extracted_table(structured):
    rows = []
    for status, key in [("Parsed", "parsed_fields"), ("Needs review", "uncertain_fields")]:
        for field in structured.get(key, []):
            rows.append({**{k: field.get(k) for k in ["nutrient", "value", "unit", "basis"]}, "status": status})
    return pd.DataFrame(rows, columns=["nutrient", "value", "unit", "basis", "status"])


def render_bio(images, summary, rows, bundle):
    st.markdown("### How word labels work (BIO)\n"
                "- **B = beginning**, **I = inside the same answer**, **O = outside any marked answer**. An entity is a group of words marked as one answer.\n"
                "- Example: with **Fat 12 g** in a per-100-g column, `12` → `B-FAT_100G` and `g` → `I-FAT_100G` mark the value **12 g**, not the nutrient-name word Fat; an unmarked word → `O`.\n"
                "- The reference labels come from the dataset; they were not independently checked against each image here.")
    if rows.empty:
        st.info("No cached predictions available yet.")
        return
    available = images.loc[images["image_key"].isin(rows["image_key"])]
    st.caption(f"{len(available):,} images with saved predictions match the browsing selection. "
               "Browsing filters do not change the results target or scores.")
    row = choose_images(available, "bio")
    if row is None:
        return
    key = row["image_key"]
    if bundle is not None:
        payload = next(s for s in bundle["samples"] if f"{bundle['split']}:{s['source_index']}" == key)
    else:
        payload = read_json(FULL / "samples" / f"{key.replace(':', '_')}.json")
        if payload.get("input_fingerprint") != summary.get("input_fingerprint") or payload.get("inference_schema_version") != CACHE_SCHEMA:
            raise ValueError("Selected prediction does not belong to this published cohort snapshot.")
    annotations = original_annotations(key, fingerprint([LINKED / "integrated_annotations.parquet"]))
    prediction = checked_prediction(payload, row, annotations)
    st.caption(f"{key} · barcode {row['barcode']} · {len(annotations):,} original OCR words · every word has a saved prediction")
    path = Path(payload.get("image_path", ""))
    # Local images only, never fetch a missing image or choose another packaging version.
    if path.is_file() and path.resolve().is_relative_to(ROOT.resolve()):
        st.image(str(path), width=450)
    else:
        st.info("Original cached image is not available locally. No image download was attempted.")
    st.subheader("Extracted nutrient values")
    structured = prediction.get("structured_output") or {}
    table(extracted_table(structured), {"nutrient": "Nutrient", "value": "Amount", "unit": "Unit", "basis": "Per what amount?", "status": "Reading status"})
    st.caption("Saved model guesses, not verified nutrition facts. Basis = per 100 g, per 100 ml or per serving (portion). "
               "Parsed means the text could be read as a value, not that it is correct. Needs review means the reading is uncertain. Units and basis are shown as saved.")
    with st.expander("Advanced · original token tags and cached JSON", expanded=False):
        st.caption("Column key: ocr_token = supplied word; token_index = word position; bounding_box = word location on a 0–1000 scale; "
               "gold_bio_id = reference label number; gold_bio_label = reference label; predicted_bio_label = model's label. "
               "image_key identifies the image. Source column names stay unchanged. JSON is the saved computer-readable output.")
        raw = annotations.copy()
        raw["predicted_bio_label"] = prediction["predicted_labels"]
        table(raw)
        st.json(structured)
        st.download_button("Download selected cached prediction", json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           file_name=f"{key.replace(':', '_')}_cached.json", mime="application/json")


def render_approaches():
    st.subheader("How the Methods Were Compared")
    st.markdown("""
1. **Rules:** predefined, hand-written Python checks detect numbers, units and serving amounts in supplied OCR text. No machine learning is used; unrelated numbers and OCR mistakes can cause errors.
2. **Pretrained LayoutLMv3:** an already-trained model uses the image, OCR words and their positions. It was not retrained here.
""")
    saved = None
    if BUNDLE.exists():
        candidate = read_json(BUNDLE)
        if candidate.get("inference_schema_version") == CACHE_SCHEMA:
            saved = candidate
        else:
            st.warning("Saved comparison uses an incompatible schema; scores are not displayed.")
    else:
        st.info("Saved approach-comparison results are unavailable.")
    if saved is not None:
        st.caption(f"Established fixed sample: {len(saved['samples'])} selected official-test images, supplied OCR and original reference annotations. "
                   "These are not independently verified image readings. The results-source selector and product filters do not change this comparison.")
    for heading, explanation, methods in [
        ("Basic Nutrient Detection",
         "Both methods are scored on the same task: finding value and measurement groups. "
         "All annotated nutrient values become VALUE; the annotated serving-size group becomes BASIS "
         "(for example, 30 g as a serving amount). An ordinary OCR phrase such as per100g is not automatically a reference BASIS entity.",
         [("Rules · Basic Nutrient Detection", "coarse_rule_metrics"),
          ("LayoutLMv3 · Basic Nutrient Detection", "coarse_model_metrics")]),
        ("Detailed Nutrient Identification",
         "The model must also identify the specific nutrient and measurement basis: for example, "
         "12 g → Fat per100g. The value entity is 12 g; per100g describes its measurement basis, "
         "not a separately tagged reference phrase. Nutrient-name words are not the value entity. "
         "This model-only score is shown separately, not compared with the rules.",
         [("LayoutLMv3 · Detailed Nutrient Identification", "fine_grained_metrics")]),
    ]:
        st.subheader(heading)
        st.markdown(explanation)
        if saved is not None:
            comparisons = []
            for title, key in methods:
                m = saved[key]
                comparisons.append({"Approach / task": title, "Precision": f"{m['precision']:.1%}",
                                    "Recall": f"{m['recall']:.1%}", "F1": f"{m['f1']:.1%}"})
            table(pd.DataFrame(comparisons))
    st.markdown("- **Precision:** share of predicted groups that exactly match the reference. **Recall:** share of reference groups found exactly. **F1:** a balanced score for both. "
                "An exact match requires the same entity type and first/last word boundaries. Compare rules and model only under Basic Nutrient Detection.\n"
                "- **Structured output:** saved groups are parsed into nutrient, value, unit and basis. Uncertain readings stay **Needs review**, with no guessing. "
                "A correct span does not prove a correct number.\n"
                "- **Full matched cohort:** the same model on more images, not a new method. Train and test scores stay separate; partial results are not final.")
    with st.expander("Details · long images and new-image reading", expanded=False):
        st.markdown("- **Overlapping windows:** a long word list is read in smaller chunks that share some words, so words near a chunk boundary are not lost. "
                    "This is one model's input method, not another model. Saved predictions must cover every original word.\n"
                    "- Basic detection reuses the same model predictions and reference labels with broader categories; no second model was trained.\n"
                    "- For new images, EasyOCR first reads words and their locations. Its mistakes add another source of error. "
                    "New uploads have no matching reference BIO labels, so these scores do not measure upload accuracy. "
                    "One manually checked example is not a general benchmark. Nothing is run on this dashboard.")


def render_results(summary):
    st.subheader("Understanding the Results")
    st.caption("Detailed Nutrient Identification scores for test and train images, kept separate.")
    st.info("Fixed results scope: dataset selection and product filters do not change these scores, the scored image count or the target count. "
            "Use Cached predictions source to choose a different saved results group.")
    table(metric_rows(summary), {"split": "Dataset group", "processed_images": "Images scored", "target_images": "Target images",
                                "precision": "Precision · answers right", "recall": "Recall · reference answers found",
                                "F1": "F1 · balanced score", "token_accuracy": "Word-label accuracy",
                                "token_support": "Words scored", "true_positives": "Exact matches (TP)",
                                "false_positives": "Wrong/extra answers (FP)", "false_negatives": "Missed answers (FN)",
                                "gold_span_support": "Reference answers scored"})
    st.caption("Scores are fractions: 0.8 means 80%. Counts are pooled across scored images, not averaged image by image. "
               "Blank means not processed or not applicable, not zero accuracy. Seeded-20 is a small test subset, not the whole test group.")
    st.markdown("""
| Score | What it tells you |
|---|---|
| Precision | How many model answers were right: the share that exactly match reference answers. |
| Recall | How many reference answers were found: the share recovered exactly by the model. |
| F1 | A balanced score for precision and recall. Both need to be high for a high F1. |
| Token accuracy | How often word labels were right, including O (outside an answer). Many O words can make this high even when nutrient answers are missed. |

- **Reference / gold:** the dataset's original word labels, not independently verified readings of each image or confirmed product facts.
- **Exact entity match:** the same entity type (specific nutrient and measurement basis here), first word and last word. Finding only part of an answer does not count.
- **Train vs test:** train is the learning group; test is the separate checking group. The existing model may have seen train examples, so train scores do not show performance on unseen data. The model's training history still matters for test scores too.
- These scores check labels on supplied OCR words, not OCR reading quality, numeric field correctness or current packaging truth.
""")
    with st.expander("Details · matches, counts and score formulas", expanded=False):
        st.markdown("""
- **True positive (TP):** one model answer exactly matches a reference answer.
- **False positive (FP):** a model answer has no exact reference match.
- **False negative (FN):** a reference answer was not found exactly. Wrong word boundaries or nutrient labels can cause both FP and FN.
- **True negative (TN):** correctly leaving something unmarked. There is no fixed list of all possible non-answer word groups, so entity scores do not use TN. Correct O word labels count in token accuracy instead.
- **Support:** how many reference answers or words were actually scored, not the pending image group.
- **Image coverage:** completed images divided by the fixed target. Pending and failed images remain in that target, not in score calculations.
- **Token coverage:** how many original words received predictions. Accepted saved images must cover every word; this does not mean every target image is complete.
- **Micro scores:** pool exact answer counts across completed images. Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2 × precision × recall / (precision + recall).
""")


def main():
    st.set_page_config(page_title="Nutrition · Clean dashboard", page_icon="◈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        products, inventory, _ = load_inventory(fingerprint([LINKED / n for n in
            ["integrated_products.parquet", "integrated_images.parquet", "provenance.json"]]))
        with st.sidebar:
            st.markdown("### Nutrition explorer")
            section = st.radio("Navigate dashboard", SECTIONS, key="section")
            st.divider()
            dataset = st.selectbox("Dataset", DATASETS, key="dataset")
            frame = (load_sample(fingerprint([ROOT / "data/openfoodfacts_eda_enriched.csv"]))
                     if dataset == DATASETS[0] else products_for_eda(products))
            # Clear stale selections when the product dataset changes.
            if st.session_state.get("filter_dataset") != dataset:
                st.session_state["countries"] = []
                st.session_state["categories"] = []
                st.session_state["filter_dataset"] = dataset
            countries = st.multiselect("Country labels", label_choices(frame, "countries"), key="countries")
            categories = st.multiselect("Category labels", label_choices(frame, "categories"), key="categories")
            st.caption("Any selected country can match; any selected category can match. If both filters are used, a product must match both.")
            st.caption("Saved local data only. No model loading, image-text reading, predictions or training are run.")
        filtered = filter_products(frame, countries, categories)
        images = scoped_images(inventory, filtered, bool(countries or categories))
        st.markdown(f'<div class="clean-hero"><small>OPEN FOOD FACTS · CLEAN VIEW</small>'
                    f'<h1>{escape(section)}</h1><p>Explore the data. Understand the evidence.</p></div>', unsafe_allow_html=True)
        st.caption(f"{dataset} · {len(filtered):,} / {len(frame):,} product rows · "
                         + ("Part 2: images with an exact barcode match to selected products." if countries or categories else
                             "Part 2: all original images (unfiltered), regardless of the product dataset."))
        if filtered.empty:
            st.info("No product rows match these filters. Clear a filter to broaden the selection.")
        if section == SECTIONS[0]:
            render_raw(filtered, images)
        elif section == SECTIONS[1]:
            render_overview(filtered, images)
        elif section == SECTIONS[2]:
            render_understanding(filtered, images, dataset)
        elif section == SECTIONS[3]:
            render_nutrients(filtered)
        elif section == SECTIONS[4]:
            render_quality(filtered)
        elif section == SECTIONS[5]:
            render_part2(images, inventory)
        else:
            source = st.selectbox("Cached predictions source", SOURCES, key="prediction_source")
            st.caption("Choose saved results: Full matched cohort = all barcode-matched target images (may be partial); "
                       "Seeded 20 test images = the same small test group chosen in advance. This does not start a run.")
            summary, rows, bundle = read_results(source)
            if summary is None:
                st.info("No saved results for this source. Nothing was executed.")
                return
            render_status(summary)
            if source == SOURCES[0] and summary["failed_images"]:
                with st.expander("Failure details · not scored as zero", expanded=False):
                    st.json(read_json(FULL / "failures.json"))
            if section == SECTIONS[6]:
                with st.expander("Approaches & comparison · brief explanation", expanded=True):
                    render_approaches()
                render_bio(images, summary, rows, bundle)
                with st.expander("Results · scores and what they mean", expanded=False):
                    render_results(summary)
            else:
                render_approaches()
                render_results(summary)
    except (OSError, ValueError, KeyError) as error:
        st.error("Saved data could not be shown safely. No new processing was started.")
        with st.expander("Details · saved-data error", expanded=False):
            st.code(str(error), language=None)
        st.caption("No source data, reference labels, model state or saved run progress was changed.")


if __name__ == "__main__":
    main()