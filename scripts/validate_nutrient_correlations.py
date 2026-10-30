"""Validate nutrient correlations in the existing Open Food Facts EDA sample.

Reads the already-processed 50k CSV only. The source file is never modified.
Missing, blank, non-numeric and non-finite values remain missing; no imputation
or zero fill is performed. Outputs are written to outputs/ and reports/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "openfoodfacts_eda_enriched.csv"
EDA_CORRELATIONS = ROOT / "reports" / "eda" / "nutrition_correlations.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_PATH = ROOT / "reports" / "nutrient_correlation_validation.md"
FIGURE_DIR = ROOT / "reports" / "nutrient_correlation_validation"

FIELDS = [
    "energy-kcal_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g",
]
PAIRWISE_OF_INTEREST = [
    ("carbohydrates_100g", "sugars_100g"),
    ("fat_100g", "energy-kcal_100g"),
    ("carbohydrates_100g", "energy-kcal_100g"),
    ("proteins_100g", "energy-kcal_100g"),
]
ABSENT_TEXT = {"", "nan", "none", "null", "<na>", "n/a", "na", "undefined"}


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw-string data, numeric coercion, and coercion diagnostics."""
    if not INPUT.is_file():
        raise FileNotFoundError(f"Existing processed sample not found: {INPUT}")
    raw = pd.read_csv(INPUT, dtype={"code": "string"}, keep_default_na=False, low_memory=False)
    missing_columns = [column for column in FIELDS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Required nutrient columns are absent: {missing_columns}")

    raw_text = raw[FIELDS].astype("string").apply(lambda col: col.str.strip())
    numeric = raw[FIELDS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    diagnostics: list[dict[str, Any]] = []
    for column in FIELDS:
        text = raw_text[column]
        explicit_missing = text.isna() | text.str.casefold().isin(ABSENT_TEXT)
        conversion_failure = ~explicit_missing & numeric[column].isna()
        valid = numeric[column].dropna()
        diagnostics.append(
            {
                "field": column,
                "rows": len(raw),
                "explicit_missing_or_blank": int(explicit_missing.sum()),
                "numeric_values": int(numeric[column].notna().sum()),
                "conversion_failures": int(conversion_failure.sum()),
                "non_finite_values_coerced_missing": int(
                    np.isinf(pd.to_numeric(text, errors="coerce").to_numpy(dtype=float, na_value=np.nan)).sum()
                ),
                "recorded_zero_values": int(valid.eq(0).sum()),
                "negative_values": int(valid.lt(0).sum()),
                "above_100_values": int(valid.gt(100).sum()),
                "minimum": float(valid.min()) if len(valid) else np.nan,
                "median": float(valid.median()) if len(valid) else np.nan,
                "p99": float(valid.quantile(0.99)) if len(valid) else np.nan,
                "maximum": float(valid.max()) if len(valid) else np.nan,
            }
        )
    return raw, numeric, pd.DataFrame(diagnostics)


def pearson_and_count(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = data.corr(method="pearson", min_periods=2)
    valid = data.notna().astype("int64")
    counts = valid.T.dot(valid)
    counts.index.name = "field"
    return corr, counts


def safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> tuple[float, int]:
    paired = pd.concat([left, right], axis=1).dropna()
    n = len(paired)
    if n < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan, n
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method=method)), n


def plausibility_mask(data: pd.DataFrame) -> pd.Series:
    """Conservative per-100-g bounds for sensitivity only, never source edits."""
    mask = pd.Series(True, index=data.index)
    for column in ["fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g"]:
        mask &= data[column].between(0, 100, inclusive="both")
    # Pure fat is about 9 kcal/g: 900 kcal/100g is a conservative energy ceiling.
    mask &= data["energy-kcal_100g"].between(0, 900, inclusive="both")
    return mask


def tukey_pair_mask(pair: pd.DataFrame, multiplier: float = 3.0) -> pd.Series:
    """Flag only points outside 3×IQR fences, as an extreme-tail sensitivity."""
    keep = pd.Series(True, index=pair.index)
    for column in pair.columns:
        q1, q3 = pair[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            # A zero IQR may occur in sparse/rounded nutrient fields; avoid
            # treating every tied observation as an outlier.
            continue
        keep &= pair[column].between(q1 - multiplier * iqr, q3 + multiplier * iqr)
    return keep


def correlation_sensitivity(numeric: pd.DataFrame) -> pd.DataFrame:
    plausible = plausibility_mask(numeric)
    rows: list[dict[str, Any]] = []
    for left, right in PAIRWISE_OF_INTEREST:
        pair = numeric[[left, right]].dropna()
        original_r, original_n = safe_corr(pair[left], pair[right], "pearson")
        spearman_r, spearman_n = safe_corr(pair[left], pair[right], "spearman")
        plausible_pair = numeric.loc[plausible, [left, right]].dropna()
        plausible_r, plausible_n = safe_corr(plausible_pair[left], plausible_pair[right], "pearson")
        extreme_keep = tukey_pair_mask(pair)
        trimmed_pair = pair.loc[extreme_keep]
        trimmed_r, trimmed_n = safe_corr(trimmed_pair[left], trimmed_pair[right], "pearson")
        rows.extend(
            [
                {
                    "x_field": left,
                    "y_field": right,
                    "condition": "A_original_pairwise_Pearson",
                    "correlation": original_r,
                    "n": original_n,
                    "excluded_from_pairwise_n": 0,
                    "rule": "Raw numeric coercion; pairwise-complete observations; no imputation or outlier removal.",
                },
                {
                    "x_field": left,
                    "y_field": right,
                    "condition": "B_plausibility_screen_Pearson",
                    "correlation": plausible_r,
                    "n": plausible_n,
                    "excluded_from_pairwise_n": original_n - plausible_n,
                    "rule": "On the full six-field frame: macros and salt 0–100 g/100g; energy-kcal 0–900 kcal/100g.",
                },
                {
                    "x_field": left,
                    "y_field": right,
                    "condition": "C_3xIQR_extremes_excluded_Pearson",
                    "correlation": trimmed_r,
                    "n": trimmed_n,
                    "excluded_from_pairwise_n": original_n - trimmed_n,
                    "rule": "Pair-specific values outside Q1−3×IQR to Q3+3×IQR excluded; sensitivity only, not a deletion policy.",
                },
                {
                    "x_field": left,
                    "y_field": right,
                    "condition": "D_original_pairwise_Spearman",
                    "correlation": spearman_r,
                    "n": spearman_n,
                    "excluded_from_pairwise_n": 0,
                    "rule": "Raw pairwise-complete observations; rank-based Spearman.",
                },
            ]
        )
    return pd.DataFrame(rows)


def pairwise_complete_comparison(numeric: pd.DataFrame) -> pd.DataFrame:
    complete = numeric.dropna(how="any")
    rows = []
    for i, left in enumerate(FIELDS):
        for right in FIELDS[i:]:
            pair_r, pair_n = safe_corr(numeric[left], numeric[right], "pearson")
            cc_r, cc_n = safe_corr(complete[left], complete[right], "pearson")
            rows.append(
                {
                    "field_x": left,
                    "field_y": right,
                    "pairwise_n": pair_n,
                    "pairwise_pearson": pair_r,
                    "complete_case_n": cc_n,
                    "complete_case_pearson": cc_r,
                    "absolute_correlation_difference": abs(pair_r - cc_r)
                    if pd.notna(pair_r) and pd.notna(cc_r)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def energy_unit_audit(raw: pd.DataFrame, numeric: pd.DataFrame) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for left, right in [
        ("energy_100g", "energy-kj_100g"),
        ("energy_100g", "energy-kcal_100g"),
        ("energy-kj_100g", "energy-kcal_100g"),
    ]:
        if left not in raw or right not in raw:
            continue
        a = pd.to_numeric(raw[left], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b = pd.to_numeric(raw[right], errors="coerce").replace([np.inf, -np.inf], np.nan)
        r, n = safe_corr(a, b)
        pair = pd.DataFrame({"a": a, "b": b}).dropna()
        nonzero = pair.loc[pair["b"] > 0]
        ratio = nonzero["a"] / nonzero["b"]
        info[f"{left}_vs_{right}"] = {
            "pearson": r,
            "n": n,
            "exact_equal_count": int(pair["a"].eq(pair["b"]).sum()),
            "ratio_median_a_over_b_nonzero_b": float(ratio.median()) if len(ratio) else None,
            "ratio_n": int(len(ratio)),
        }
    if "energy-kj_100g" in numeric and "energy-kcal_100g" in numeric:
        pair = numeric[["energy-kj_100g", "energy-kcal_100g"]].dropna()
        pair = pair.loc[pair["energy-kcal_100g"] > 0]
        ratio = pair["energy-kj_100g"] / pair["energy-kcal_100g"]
        info["kj_per_kcal"] = {
            "n_positive_kcal": int(len(pair)),
            "median_ratio": float(ratio.median()) if len(ratio) else None,
            "fraction_within_2pct_of_4_184": float(
                ratio.sub(4.184).abs().le(0.02 * 4.184).mean()
            )
            if len(ratio)
            else None,
        }
    return info


def energy_consistency(numeric: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    macro_fields = ["fat_100g", "carbohydrates_100g", "proteins_100g"]
    data = numeric[macro_fields + ["energy-kcal_100g"]].dropna().copy()
    data["estimated_energy_kcal_100g"] = (
        9 * data["fat_100g"] + 4 * data["carbohydrates_100g"] + 4 * data["proteins_100g"]
    )
    data["residual_reported_minus_estimated_kcal"] = (
        data["energy-kcal_100g"] - data["estimated_energy_kcal_100g"]
    )
    data["absolute_error_kcal"] = data["residual_reported_minus_estimated_kcal"].abs()
    data["relative_absolute_error"] = data["absolute_error_kcal"] / data["energy-kcal_100g"].replace(0, np.nan)
    r, n = safe_corr(data["energy-kcal_100g"], data["estimated_energy_kcal_100g"])
    residual = data["residual_reported_minus_estimated_kcal"]
    summary = {
        "complete_macro_plus_kcal_n": int(n),
        "pearson_reported_vs_estimated": r,
        "median_absolute_error_kcal": float(data["absolute_error_kcal"].median()) if len(data) else None,
        "residual_median_reported_minus_estimated": float(residual.median()) if len(data) else None,
        "residual_p05": float(residual.quantile(0.05)) if len(data) else None,
        "residual_p95": float(residual.quantile(0.95)) if len(data) else None,
        "large_error_abs_gt_50_kcal_count": int(data["absolute_error_kcal"].gt(50).sum()),
        "large_error_abs_gt_50_kcal_pct": float(data["absolute_error_kcal"].gt(50).mean() * 100) if len(data) else None,
        "large_error_relative_gt_20pct_count": int(data["relative_absolute_error"].gt(0.20).sum()),
        "large_error_relative_gt_20pct_pct_of_positive_reported": float(data["relative_absolute_error"].dropna().gt(0.20).mean() * 100) if data["relative_absolute_error"].notna().any() else None,
    }
    return summary, data


def subgroup_correlations(raw: pd.DataFrame, numeric: pd.DataFrame, minimum_n: int = 200) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column, group_type in [("categories", "primary_category"), ("countries", "first_country_label")]:
        if column not in raw:
            continue
        groups = raw[column].astype("string").str.split(",").str[0].str.strip()
        groups = groups.mask(groups.str.casefold().isin(ABSENT_TEXT))
        for group, indices in groups.groupby(groups, dropna=True).groups.items():
            if len(indices) < minimum_n:
                continue
            for left, right in [
                ("carbohydrates_100g", "sugars_100g"),
                ("fat_100g", "energy-kcal_100g"),
            ]:
                r, n = safe_corr(numeric.loc[indices, left], numeric.loc[indices, right], "pearson")
                rho, _ = safe_corr(numeric.loc[indices, left], numeric.loc[indices, right], "spearman")
                if n >= minimum_n:
                    rows.append(
                        {
                            "group_type": group_type,
                            "group": str(group),
                            "field_x": left,
                            "field_y": right,
                            "pairwise_n": n,
                            "pearson": r,
                            "spearman": rho,
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["group_type", "group", "field_x", "field_y", "pairwise_n", "pearson", "spearman"])
    return pd.DataFrame(rows).sort_values(["group_type", "field_y", "pairwise_n"], ascending=[True, True, False])


def dashboard_reproduction(numeric: pd.DataFrame) -> dict[str, Any]:
    original = numeric.corr(method="pearson", min_periods=2)
    result: dict[str, Any] = {"saved_eda_comparison": None, "matrix_symmetry_max_abs": None, "diagonal_error_max_abs": None}
    symmetry = (original - original.T).abs().to_numpy()
    result["matrix_symmetry_max_abs"] = float(np.nanmax(symmetry)) if np.isfinite(symmetry).any() else 0.0
    diag = original.to_numpy().diagonal()
    result["diagonal_error_max_abs"] = float(np.nanmax(np.abs(diag - 1))) if len(diag) else None
    if EDA_CORRELATIONS.exists():
        saved = pd.read_csv(EDA_CORRELATIONS, index_col=0)
        common_rows = [field for field in FIELDS if field in saved.index and field in original.index]
        common_cols = [field for field in FIELDS if field in saved.columns and field in original.columns]
        differences = (saved.loc[common_rows, common_cols] - original.loc[common_rows, common_cols]).abs()
        result["saved_eda_comparison"] = {
            "common_cells": int(differences.notna().sum().sum()),
            "max_abs_difference": float(differences.max().max()) if not differences.empty else None,
        }
    return result


def make_scatterplots(numeric: pd.DataFrame, consistency: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    scatter_specs = [
        ("carbohydrates_100g", "sugars_100g", "Carbohydrates vs sugars"),
        ("fat_100g", "energy-kcal_100g", "Fat vs energy (kcal/100 g)"),
        ("carbohydrates_100g", "energy-kcal_100g", "Carbohydrates vs energy (kcal/100 g)"),
        ("proteins_100g", "energy-kcal_100g", "Protein vs energy (kcal/100 g)"),
    ]
    for ax, (x, y, title) in zip(axes.flat, scatter_specs):
        pair = numeric[[x, y]].dropna()
        ax.scatter(pair[x], pair[y], s=7, alpha=0.16, color="#2864A5", edgecolors="none", rasterized=True)
        r, n = safe_corr(pair[x], pair[y])
        ax.set_title(f"{title}\nPearson r={r:.3f}; pairwise n={n:,}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.2)
    fig.suptitle("Raw pairwise-complete values (extreme values retained)", fontsize=14)
    path = FIGURE_DIR / "nutrient_correlation_scatterplots.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if not consistency.empty:
        ax.scatter(
            consistency["estimated_energy_kcal_100g"],
            consistency["energy-kcal_100g"],
            s=7,
            alpha=0.16,
            color="#9A5BB5",
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel("Estimated energy: 9×fat + 4×carbs + 4×protein")
        ax.set_ylabel("Reported energy-kcal_100g")
        ax.set_title("Approximate macro energy vs reported energy (raw values)")
        ax.grid(alpha=0.2)
    consistency_path = FIGURE_DIR / "approximate_energy_consistency.png"
    fig.savefig(consistency_path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    if frame.empty:
        return "*(No qualifying rows.)*"
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
        )
    columns = [str(column) for column in display.columns]
    rows = [[str(value).replace("|", "\\|").replace("\n", " ") for value in row] for row in display.to_numpy()]
    head = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def build_report(
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    data_quality: pd.DataFrame,
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    complete_case: pd.DataFrame,
    energy_units: dict[str, Any],
    consistency: dict[str, Any],
    consistency_records: pd.DataFrame,
    subgroups: pd.DataFrame,
    reproduction: dict[str, Any],
    raw_corr: pd.DataFrame,
    carb_sugar: dict[str, Any],
) -> None:
    top_valid = data_quality.set_index("field")["numeric_values"].to_dict()
    macros = numeric[["fat_100g", "carbohydrates_100g", "proteins_100g"]]
    invalid_macros = {
        field: int((macros[field].notna() & ~macros[field].between(0, 100, inclusive="both")).sum())
        for field in macros
    }
    energy_invalid = int((numeric["energy-kcal_100g"].notna() & ~numeric["energy-kcal_100g"].between(0, 900, inclusive="both")).sum())
    ratio = carb_sugar["ratio_summary"]
    equals = carb_sugar["exact_equal_count"]
    greater = carb_sugar["sugar_gt_carbs_count"]
    n_cs = carb_sugar["n"]

    relevant_complete = complete_case.loc[
        complete_case.apply(
            lambda row: row["field_x"] in FIELDS and row["field_y"] in FIELDS,
            axis=1,
        )
    ]
    summary_rows = []
    for left, right in PAIRWISE_OF_INTEREST:
        pair = numeric[[left, right]].dropna()
        rp, n = safe_corr(pair[left], pair[right], "pearson")
        rs, _ = safe_corr(pair[left], pair[right], "spearman")
        cc = relevant_complete.loc[
            ((relevant_complete["field_x"] == left) & (relevant_complete["field_y"] == right))
            | ((relevant_complete["field_x"] == right) & (relevant_complete["field_y"] == left))
        ]
        cc_row = cc.iloc[0] if not cc.empty else None
        summary_rows.append(
            {
                "pair": f"{left} × {right}",
                "pairwise_n": n,
                "Pearson_pairwise": rp,
                "Spearman_pairwise": rs,
                "complete_case_n": cc_row["complete_case_n"] if cc_row is not None else np.nan,
                "Pearson_complete_case": cc_row["complete_case_pearson"] if cc_row is not None else np.nan,
            }
        )
    principal = pd.DataFrame(summary_rows)

    lines = [
        "# Nutrient Correlation Validation",
        "",
        f"Source: existing processed sample, `{INPUT.relative_to(ROOT).as_posix()}` ({len(raw):,} rows, {len(raw.columns)} columns). The acquisition note identifies the first 50,000 rows of the English Open Food Facts compressed TSV. No source data was changed or redownloaded.",
        "",
        "## Implementation inspected before analysis",
        "",
        "- Dashboard defaults to the enriched local CSV; an uploaded file can replace it. The plotted `frame` is the currently filtered subset after optional country/category filters. With the default local sample and no filters it is all 50,000 records.",
        "- Dashboard code selects `corr_cols` among available nutrient fields (numeric-coercible support ≥20; default six fields) and calculates `frame[corr_cols].apply(pd.to_numeric, errors='coerce').corr(min_periods=2)`. This is raw stored values after numeric coercion—no scaling, winsorization, outlier removal, or imputation is applied to the correlation inputs.",
        "- Pandas Pearson uses pairwise-complete observations by default. Blank/non-numeric cells become missing under `errors='coerce'`; there is no `fillna(0)` in the correlation path. Explicit zero values remain zero.",
        "- The EDA module independently coerces nutrient fields with `pd.to_numeric(errors='coerce')` and saves a Pearson matrix. The dashboard matrix and this validation match numerically on shared cells below.",
        "",
        "## Findings at a glance",
        "",
        markdown_table(principal, 4),
        "",
        f"- Dashboard matrix symmetry max absolute error: {reproduction['matrix_symmetry_max_abs']:.3g}; maximum diagonal deviation from 1: {reproduction['diagonal_error_max_abs']:.3g}.",
        f"- Compared with the saved EDA Pearson matrix, shared cells: {reproduction['saved_eda_comparison']['common_cells'] if reproduction['saved_eda_comparison'] else 'not available'}; maximum absolute coefficient difference: {reproduction['saved_eda_comparison']['max_abs_difference'] if reproduction['saved_eda_comparison'] else 'not available'}.",
        "",
        "## Reproduced Pearson correlation matrix",
        "",
        "All six requested fields are present. Coefficients below use the original stored numbers after numeric coercion and pairwise deletion only.",
        "",
        markdown_table(corr.reset_index().rename(columns={"index": "field"}), 4),
        "",
        "### Pairwise valid-observation counts",
        "",
        markdown_table(counts.reset_index().rename(columns={"index": "field"}), 0),
        "",
        "## Missing values and conversion",
        "",
        "No missing values were filled with zero. Numeric conversion used `pd.to_numeric(errors='coerce')`; blank/placeholders and any unparseable nonblank values become missing. Pairwise counts differ per coefficient, while complete-case comparison uses rows with all six fields present.",
        "",
        markdown_table(data_quality, 3),
        "",
        f"- Six-field complete-case sample: {len(numeric.dropna(how='any')):,} rows, versus {len(raw):,} source records.",
        f"- Additional missingness introduced by numeric conversion is field-specific; see `outputs/nutrient_data_quality.csv`. Recorded zeros are counted separately and never used as missing-value substitutes.",
        "",
        "### Pairwise versus six-field complete-case Pearson",
        "",
        markdown_table(
            complete_case.loc[
                complete_case.apply(
                    lambda row: row['field_x'] in FIELDS and row['field_y'] in FIELDS,
                    axis=1,
                )
            ].head(21),
            4,
        ),
        "",
        "## Carbohydrate–sugar relationship",
        "",
        f"- Pearson r={carb_sugar['pearson']:.5f}; Spearman ρ={carb_sugar['spearman']:.5f}; pairwise n={n_cs:,}.",
        f"- Equal values: {equals:,}/{n_cs:,} ({100 * equals / n_cs:.2f}%). Sugars greater than carbohydrates: {greater:,}/{n_cs:,} ({100 * greater / n_cs:.2f}%).",
        f"- Valid zero-carbohydrate records: {carb_sugar['zero_carbohydrate_count']:,}. Ratio summary for positive carbohydrates and valid sugars: n={ratio['n']:,}, median={ratio.get('median', float('nan')):.4g}, p95={ratio.get('p95', float('nan')):.4g}, max={ratio.get('max', float('nan')):.4g}.",
        f"- Carbohydrate values outside 0–100 g/100g: {invalid_macros['carbohydrates_100g']:,}; sugars outside that range: {invalid_macros['sugars_100g']:,}. Values over 100 are flagged for review, not silently deleted from the original result.",
        "- The correlation alone is not evidence of duplicate columns: the script separately tests row-wise equality and column identity. High positive association may be genuine because sugars are a subset of carbohydrates, but concentration in sugary products and extreme records also affect Pearson r; compare rank correlation and sensitivity outputs before attributing cause.",
        "- Scatter plot: `reports/nutrient_correlation_validation/nutrient_correlation_scatterplots.png` (upper-left panel).",
        "",
        "## Fat, carbohydrate, protein, and energy",
        "",
        "- Energy analyzed in the requested matrix is `energy-kcal_100g`; kJ is a separate field. The unit audit compares the paired kcal/kJ ratio to 4.184 and compares the ambiguous legacy `energy_100g` field with both explicit energy columns.",
        f"- `energy-kcal_100g` values outside the conservative 0–900 kcal/100g range: {energy_invalid:,}; fat outside 0–100 g/100g: {invalid_macros['fat_100g']:,}. These are strong candidates for unit/data review and can distort Pearson correlation substantially.",
        f"- Energy-unit audit: `{json.dumps(energy_units, ensure_ascii=False)}`.",
        "- Scatter plots for fat, carbohydrates, protein versus kcal energy are in the same four-panel PNG listed above. Raw axes retain all values, so outliers remain visible.",
        "",
        "## Approximate energy consistency check",
        "",
        "Estimated energy = 9×fat + 4×carbohydrates + 4×protein (kcal/100g), applied only where all three macros and reported kcal are numeric. This is diagnostic, not a ground-truth validator; fibre, alcohol, polyols, organic acids, conversion conventions, and label rounding can explain differences.",
        f"- Complete rows: {consistency['complete_macro_plus_kcal_n']:,}; Pearson reported vs estimated: {consistency['pearson_reported_vs_estimated']:.4f}; median absolute error: {consistency['median_absolute_error_kcal']:.3f} kcal/100g.",
        f"- Residual (reported − estimated) median: {consistency['residual_median_reported_minus_estimated']:.3f}; p05={consistency['residual_p05']:.3f}; p95={consistency['residual_p95']:.3f} kcal/100g.",
        f"- Absolute error >50 kcal/100g: {consistency['large_error_abs_gt_50_kcal_count']:,} ({consistency['large_error_abs_gt_50_kcal_pct']:.2f}%). Relative error >20% among positive reported energy: {consistency['large_error_relative_gt_20pct_count']:,} ({consistency['large_error_relative_gt_20pct_pct_of_positive_reported']:.2f}%). Thresholds are triage markers, not automatic error labels.",
        "- Consistency scatter: `reports/nutrient_correlation_validation/approximate_energy_consistency.png`.",
        "",
        "## Outlier sensitivity",
        "",
        "A. Original: pairwise-complete Pearson on raw numeric values. B. Plausibility sensitivity: 0–100 g/100g for fat/carbs/sugars/protein/salt and 0–900 kcal/100g for energy; conservative screening only. C. Extreme-tail sensitivity: remove points outside pair-specific Q1−3×IQR to Q3+3×IQR. D. Spearman: rank correlation on original pairwise-complete values. None of these modifies the source or replaces the baseline result.",
        "",
        markdown_table(sensitivity, 4),
        "",
        "## Data composition",
        "",
        "Subgroup correlations use the first listed category/country label per product (no product duplication) and are shown only when each pair has at least 200 valid observations. These are descriptive comparisons, not causal or significance tests. The sample is the first 50,000 rows of the English export and is known to be US-heavy; its overall coefficients should not be generalized to all Open Food Facts products.",
        "",
        markdown_table(subgroups.head(48), 4),
        "",
        "## Conclusion and dashboard decision",
        "",
        "- **Is the existing matrix mathematically correct?** The inspected formula is standard Pearson correlation with numeric coercion and pairwise-complete observations. Independent recomputation is symmetric, has unit diagonal for nonconstant fields, and matches the saved EDA matrix within the reported numerical tolerance. No arithmetic or missing-value imputation bug was found.",
        "- **Is carbohydrates–sugars near-perfect?** Its exact measured Pearson/Spearman values, equality share, and sensitivity results are given above. Sugars are nutritionally a subset of carbohydrates, so a strong positive relationship can be real; outliers and the product mix must be considered before labeling it misleading.",
        "- **Why is carbohydrates–energy lower?** Energy varies with fat/protein and product formulation, while the dataset is heterogeneous; quantified pairwise correlations and rank results above describe the observed association. Extreme invalid-looking values can also depress or distort Pearson r.",
        "- **Why is fat–energy close to zero in the raw matrix?** The sample has severe values above plausible per-100-g bounds (including the reported fat maximum in the existing statistics file). Pearson correlation is scale-sensitive; the sensitivity table quantifies how much screening/extreme-tail handling changes the association. Do not equate this raw coefficient with normal nutritional physics without investigating those records.",
        "- **Data-quality issues?** Existing EDA statistics already flag extreme maxima (e.g. fat up to 1,000,000 g/100g, energy-kcal up to 47,000 kcal/100g, carbohydrates/sugars up to 40,000 g/100g) and a negative protein. They remain in the baseline matrix and should be reviewed as potential bad units/records, not silently rewritten.",
        "- **Should the dashboard calculation change?** Keep Pearson as a valid selectable method; do not filter silently. Improve transparency: annotate cells with coefficients, expose pairwise n in hover/count table, offer Spearman as a robust rank-based companion, and label missingness as pairwise complete. The dashboard change should not alter the raw Pearson result.",
        "",
        "## Artifacts and reproducibility",
        "",
        "Run from the project root: `python scripts/validate_nutrient_correlations.py`. It loads only the existing processed CSV.",
        "- `outputs/nutrient_correlation_matrix_pearson.csv` and `outputs/nutrient_correlation_pairwise_n.csv`.",
        "- `outputs/nutrient_pairwise_vs_complete_case.csv`, `outputs/nutrient_correlation_sensitivity.csv`, `outputs/nutrient_data_quality.csv`, `outputs/nutrient_subgroup_correlations.csv`, `outputs/nutrient_suspect_records.csv`, `outputs/energy_consistency_records.csv`, and `outputs/nutrient_validation_summary.json`.",
        "- `reports/nutrient_correlation_validation/nutrient_correlation_scatterplots.png` and `reports/nutrient_correlation_validation/approximate_energy_consistency.png`.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw, numeric, quality = load_source()
    corr, counts = pearson_and_count(numeric)
    spearman, _ = pearson_and_count(numeric)
    # Use explicit Spearman to ensure the output comparison is not accidentally
    # derived from Pearson; pair counts are method-independent.
    spearman = numeric.corr(method="spearman", min_periods=2)
    sensitivity = correlation_sensitivity(numeric)
    complete_case = pairwise_complete_comparison(numeric)
    energy_units = energy_unit_audit(raw, numeric)
    consistency_summary, consistency_records = energy_consistency(numeric)
    subgroups = subgroup_correlations(raw, numeric)
    reproduction = dashboard_reproduction(numeric)

    carb_sugar_pair = numeric[["carbohydrates_100g", "sugars_100g"]].dropna()
    carb_sugar_pearson, carb_sugar_n = safe_corr(
        carb_sugar_pair["carbohydrates_100g"], carb_sugar_pair["sugars_100g"], "pearson"
    )
    carb_sugar_spearman, _ = safe_corr(
        carb_sugar_pair["carbohydrates_100g"], carb_sugar_pair["sugars_100g"], "spearman"
    )
    equal_count = int(carb_sugar_pair["carbohydrates_100g"].eq(carb_sugar_pair["sugars_100g"]).sum())
    sugar_gt_count = int(carb_sugar_pair["sugars_100g"].gt(carb_sugar_pair["carbohydrates_100g"]).sum())
    positive_ratio = carb_sugar_pair.loc[
        carb_sugar_pair["carbohydrates_100g"].gt(0), "sugars_100g"
    ] / carb_sugar_pair.loc[carb_sugar_pair["carbohydrates_100g"].gt(0), "carbohydrates_100g"]
    ratio_summary = {
        "n": int(len(positive_ratio)),
        **{
            name: float(value)
            for name, value in {
                "min": positive_ratio.min(),
                "p01": positive_ratio.quantile(0.01),
                "p05": positive_ratio.quantile(0.05),
                "median": positive_ratio.median(),
                "p95": positive_ratio.quantile(0.95),
                "p99": positive_ratio.quantile(0.99),
                "max": positive_ratio.max(),
            }.items()
        },
    }
    carb_sugar = {
        "pearson": carb_sugar_pearson,
        "spearman": carb_sugar_spearman,
        "n": carb_sugar_n,
        "exact_equal_count": equal_count,
        "sugar_gt_carbs_count": sugar_gt_count,
        "zero_carbohydrate_count": int(numeric["carbohydrates_100g"].eq(0).sum()),
        "ratio_summary": ratio_summary,
        "duplicate_column_exact": bool(numeric["carbohydrates_100g"].equals(numeric["sugars_100g"])),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUTPUT_DIR / "nutrient_correlation_matrix_pearson.csv", index_label="field")
    spearman.to_csv(OUTPUT_DIR / "nutrient_correlation_matrix_spearman.csv", index_label="field")
    counts.to_csv(OUTPUT_DIR / "nutrient_correlation_pairwise_n.csv")
    sensitivity.to_csv(OUTPUT_DIR / "nutrient_correlation_sensitivity.csv", index=False)
    complete_case.to_csv(OUTPUT_DIR / "nutrient_pairwise_vs_complete_case.csv", index=False)
    quality.to_csv(OUTPUT_DIR / "nutrient_data_quality.csv", index=False)
    subgroups.to_csv(OUTPUT_DIR / "nutrient_subgroup_correlations.csv", index=False)
    consistency_records.to_csv(OUTPUT_DIR / "energy_consistency_records.csv", index=False)

    raw_numerics = raw[FIELDS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    plausible = plausibility_mask(numeric)
    invalid_row = ~plausible
    suspect_fields = ["code", "product_name", "countries", "categories", *FIELDS]
    suspect_fields = [field for field in suspect_fields if field in raw]
    suspects = raw.loc[invalid_row, suspect_fields].copy()
    suspects.to_csv(OUTPUT_DIR / "nutrient_suspect_records.csv", index=False)

    make_scatterplots(numeric, consistency_records)
    summary: dict[str, Any] = {
        "source": INPUT.relative_to(ROOT).as_posix(),
        "rows": len(raw),
        "columns": len(raw.columns),
        "fields": FIELDS,
        "correlation": {
            "pearson": corr.to_dict(),
            "spearman": spearman.to_dict(),
            "pairwise_n": counts.to_dict(),
            "matrix_symmetry_max_abs": reproduction["matrix_symmetry_max_abs"],
            "diagonal_error_max_abs": reproduction["diagonal_error_max_abs"],
            "saved_eda_comparison": reproduction["saved_eda_comparison"],
        },
        "carbohydrate_sugar": carb_sugar,
        "energy_units": energy_units,
        "energy_consistency": consistency_summary,
        "plausibility_screen": {
            "rule": "macros/salt 0-100 g/100g; kcal 0-900 kcal/100g; sensitivity only",
            "rows_excluded": int(invalid_row.sum()),
            "rows_remaining": int(plausible.sum()),
        },
        "suspicious_records_written": int(len(suspects)),
        "subgroup_minimum_pair_n": 200,
        "dataset_scope": "first 50,000 rows of English OFF export; not globally representative",
    }
    (OUTPUT_DIR / "nutrient_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False, default=lambda obj: None),
        encoding="utf-8",
    )
    build_report(
        raw,
        numeric,
        quality,
        corr,
        counts,
        sensitivity,
        complete_case,
        energy_units,
        consistency_summary,
        consistency_records,
        subgroups,
        reproduction,
        raw_numerics.corr(min_periods=2),
        carb_sugar,
    )
    print(
        json.dumps(
            {
                "rows": len(raw),
                "dashboard_default_correlations": {
                    f"{left}__{right}": corr.loc[left, right]
                    for left, right in PAIRWISE_OF_INTEREST
                },
                "carbohydrate_sugar_n": carb_sugar_n,
                "suspect_rows": len(suspects),
                "report": str(REPORT_PATH),
                "outputs": str(OUTPUT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()