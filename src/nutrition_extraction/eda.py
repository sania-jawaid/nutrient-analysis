"""Profile an Open Food Facts export without claiming unobserved results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

NUTRITION_PATTERN = re.compile(
    r"(energy|fat|saturated-fat|trans-fat|carbohydrates|sugars|fiber|fibre|proteins|salt|sodium|cholesterol|vitamin|calcium|iron).*(_100g|_100ml)$",
    re.IGNORECASE,
)


def load_export(
    path: str | Path,
    row_limit: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load OFF CSV/TSV/JSON exports, optionally selecting available columns."""
    source = str(path)
    suffix = Path(source.split("?", maxsplit=1)[0]).suffix.lower()
    if suffix in {".csv", ".tsv", ".gz"} or source.startswith(("https://", "http://")):
        separator = "\t" if suffix in {".tsv", ".gz"} or "openfoodfacts" in source else None
        read_options: dict[str, Any] = {"sep": separator, "nrows": row_limit}
        if suffix == ".gz" or source.endswith(".gz"):
            read_options["compression"] = "gzip"
        if separator is None:
            read_options["engine"] = "python"
        if columns:
            available = set(
                pd.read_csv(
                    source,
                    sep=separator,
                    compression=read_options.get("compression", "infer"),
                    nrows=0,
                ).columns
            )
            selected = [column for column in columns if column in available]
            missing = sorted(set(columns) - available)
            if missing:
                print(f"Requested columns absent from export (ignored): {missing}")
            if not selected:
                raise ValueError("None of the requested columns exist in the export.")
            read_options["usecols"] = selected
        return pd.read_csv(source, **read_options)
    if suffix == ".jsonl":
        frame = pd.read_json(source, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(source)
    else:
        raise ValueError(f"Unsupported export format: {suffix}. Use CSV, TSV, GZIP TSV, JSON, or JSONL.")
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]
    return frame.head(row_limit) if row_limit is not None else frame


def _nonempty(series: pd.Series) -> pd.Series:
    """Treat nulls, whitespace, and common export placeholders as missing."""
    normalized = series.astype("string").str.strip().str.casefold()
    placeholders = {"", "undefined", "nan", "none", "null", "n/a", "na"}
    return series.notna() & ~normalized.isin(placeholders)


def split_off_labels(value: Any) -> set[str]:
    """Normalize comma-separated OFF country/category labels for membership tests."""
    if value is None or pd.isna(value):
        return set()
    return {part.strip().casefold() for part in str(value).split(",") if part.strip()}


def filter_off_export(
    frame: pd.DataFrame,
    selected_countries: list[str] | tuple[str, ...] = (),
    selected_categories: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    """Filter product records by any selected country/category label.

    Country and category groups combine with AND; multiple values within each
    group combine with OR. Matching is case-insensitive and comma-spacing agnostic.
    """
    filtered = frame
    countries = {value.strip().casefold() for value in selected_countries if value.strip()}
    categories = {value.strip().casefold() for value in selected_categories if value.strip()}
    if countries and "countries" in filtered:
        filtered = filtered.loc[filtered["countries"].map(lambda value: bool(split_off_labels(value) & countries))]
    if categories and "categories" in filtered:
        filtered = filtered.loc[filtered["categories"].map(lambda value: bool(split_off_labels(value) & categories))]
    return filtered.copy()


def analyze_export(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return descriptive summaries for missingness and nutrition-label coverage."""
    if frame.empty:
        raise ValueError("The loaded export contains no rows.")

    columns_lower = {str(column).lower(): column for column in frame.columns}
    code_column = next((columns_lower[name] for name in ("code", "id", "product_id") if name in columns_lower), None)
    nutrition_columns = [column for column in frame.columns if NUTRITION_PATTERN.search(str(column))]
    image_columns = [
        column for column in frame.columns
        if "image" in str(column).lower() and str(column).lower().endswith("_url")
    ]
    nutrition_image_columns = [
        column for column in image_columns
        if "nutrition" in str(column).lower()
    ]
    missingness = pd.DataFrame(
        {
            "column": [str(column) for column in frame.columns],
            "missing_count": [int((~_nonempty(frame[column])).sum()) for column in frame.columns],
            "missing_fraction": [float((~_nonempty(frame[column])).mean()) for column in frame.columns],
        }
    ).sort_values("missing_fraction", ascending=False)

    nutrition_coverage = pd.DataFrame(
        [
            {
                "field": str(column),
                "non_missing_count": int(_nonempty(frame[column]).sum()),
                "coverage_fraction": float(_nonempty(frame[column]).mean()),
            }
            for column in nutrition_columns
        ]
    ).sort_values("coverage_fraction", ascending=False)

    summary: dict[str, Any] = {
        "rows_examined": int(len(frame)),
        "columns_examined": int(frame.shape[1]),
        "nutrition_columns_detected": [str(column) for column in nutrition_columns],
        "image_columns_detected": [str(column) for column in image_columns],
        "identifier_column": str(code_column) if code_column is not None else None,
        "duplicate_identifier_rows": int(frame[code_column].duplicated(keep=False).sum()) if code_column is not None else None,
        "columns": [str(column) for column in frame.columns],
    }

    for candidate in ("brands", "categories", "countries", "languages", "lc"):
        column = columns_lower.get(candidate)
        if column is not None:
            values = frame[column].dropna().astype("string").str.split(",").explode().str.strip()
            values = values[~values.str.casefold().isin({"", "undefined", "nan", "none", "null", "n/a", "na"})]
            summary[f"top_{candidate}"] = values.value_counts().head(15).to_dict()

    if image_columns:
        image_presence = pd.concat([_nonempty(frame[column]) for column in image_columns], axis=1).any(axis=1)
        summary["rows_with_any_image_field"] = int(image_presence.sum())
        summary["image_field_row_fraction"] = float(image_presence.mean())
    else:
        summary["rows_with_any_image_field"] = None
        summary["image_field_row_fraction"] = None
    if nutrition_image_columns:
        nutrition_image_presence = pd.concat(
            [_nonempty(frame[column]) for column in nutrition_image_columns], axis=1
        ).any(axis=1)
        summary["nutrition_image_columns"] = [str(column) for column in nutrition_image_columns]
        summary["rows_with_nutrition_image"] = int(nutrition_image_presence.sum())
        summary["nutrition_image_row_fraction"] = float(nutrition_image_presence.mean())
    else:
        summary["nutrition_image_columns"] = []
        summary["rows_with_nutrition_image"] = 0
        summary["nutrition_image_row_fraction"] = 0.0
    if nutrition_columns:
        nutrition_presence = pd.concat([_nonempty(frame[column]) for column in nutrition_columns], axis=1).any(axis=1)
        summary["rows_with_any_structured_nutrition"] = int(nutrition_presence.sum())
        summary["structured_nutrition_row_fraction"] = float(nutrition_presence.mean())
    else:
        summary["rows_with_any_structured_nutrition"] = 0
        summary["structured_nutrition_row_fraction"] = 0.0
    return summary, missingness, nutrition_coverage


def nutrition_statistics(frame: pd.DataFrame, nutrition_columns: list[str]) -> pd.DataFrame:
    """Summarize numeric nutrient fields, including negative values and IQR outliers."""
    rows: list[dict[str, Any]] = []
    for column in nutrition_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            rows.append({"field": str(column), "valid_numeric_count": 0})
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        frequencies = valid.value_counts()
        mode_frequency = int(frequencies.max())
        modes = sorted(float(value) for value in frequencies.loc[frequencies.eq(mode_frequency)].index)
        rows.append({
            "field": str(column),
            "valid_numeric_count": int(valid.size),
            "mean": float(valid.mean()),
            "std": float(valid.std()) if valid.size > 1 else 0.0,
            "min": float(valid.min()),
            "p01": float(valid.quantile(0.01)),
            "median": float(valid.median()),
            "mode": "; ".join(f"{value:g}" for value in modes) if mode_frequency > 1 else "No unique mode (all values unique)",
            "mode_frequency": mode_frequency,
            "mode_count": len(modes),
            "p99": float(valid.quantile(0.99)),
            "max": float(valid.max()),
            "negative_count": int((valid < 0).sum()),
            "iqr_outlier_count": int(((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)).sum()),
        })
    return pd.DataFrame(rows)


def select_off_columns(header: list[str]) -> list[str]:
    """Keep core product/image/nutrition fields plus country and serving context."""
    key_fields = {
        "code", "product_name", "generic_name", "quantity", "serving_size", "serving_quantity",
        "nutrition_data_per", "ingredients_text", "categories", "brands", "countries",
        "image_url", "image_ingredients_url", "image_nutrition_url",
    }
    return [
        column for column in header
        if column in key_fields or NUTRITION_PATTERN.search(column)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="OFF CSV/TSV/JSON/JSONL path or URL")
    parser.add_argument("--output", required=True, type=Path, help="Directory for EDA artifacts")
    parser.add_argument("--row-limit", type=int, default=50_000, help="Rows to profile (default: first 50,000)")
    parser.add_argument("--columns", nargs="*", default=None, help="Explicit columns to load; default selects OFF text/image/nutrition fields")
    args = parser.parse_args()

    if args.columns:
        columns = args.columns
    elif "openfoodfacts" in args.input.lower():
        # Read only the compressed TSV header, then select fields present in this export version.
        header = pd.read_csv(args.input, sep="\t", compression="gzip", nrows=0).columns.tolist()
        columns = select_off_columns(header)
    else:
        columns = None

    frame = load_export(args.input, args.row_limit, columns)
    summary, missingness, nutrition = analyze_export(frame)
    nutrition_columns = list(nutrition["field"]) if not nutrition.empty else []
    statistics = nutrition_statistics(frame, nutrition_columns)
    numeric_frame = frame[nutrition_columns].apply(pd.to_numeric, errors="coerce") if nutrition_columns else pd.DataFrame()
    correlations = numeric_frame.corr(min_periods=2) if not numeric_frame.empty else pd.DataFrame()
    summary["source_file"] = args.input
    summary["row_limit"] = args.row_limit
    summary["loaded_columns"] = [str(column) for column in frame.columns]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    missingness.to_csv(args.output / "missingness.csv", index=False)
    nutrition.to_csv(args.output / "nutrition_coverage.csv", index=False)
    statistics.to_csv(args.output / "nutrition_statistics.csv", index=False)
    correlations.to_csv(args.output / "nutrition_correlations.csv")
    print(f"EDA artifacts written to {args.output.resolve()}")
    print(f"Rows examined: {summary['rows_examined']}; detected nutrition fields: {len(summary['nutrition_columns_detected'])}")


if __name__ == "__main__":
    main()
