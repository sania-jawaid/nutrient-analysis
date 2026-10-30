import pandas as pd

from nutrition_extraction.eda import analyze_export, filter_off_export


def test_eda_reports_missingness_and_coverage() -> None:
    frame = pd.DataFrame(
        {
            "code": ["a", "a", "c"],
            "image_nutrition_url": ["url-1", "", None],
            "energy-kcal_100g": [100, None, 0],
            "proteins_100g": [None, 4.0, None],
        }
    )
    summary, missingness, coverage = analyze_export(frame)
    assert summary["rows_examined"] == 3
    assert summary["duplicate_identifier_rows"] == 2
    assert summary["rows_with_any_image_field"] == 1
    assert summary["rows_with_any_structured_nutrition"] == 3
    assert set(coverage["field"]) == {"energy-kcal_100g", "proteins_100g"}
    assert "missing_fraction" in missingness.columns


def test_off_filters_match_comma_separated_values_with_or_and_and_semantics() -> None:
    frame = pd.DataFrame({
        "code": ["1", "2", "3", "4"],
        "countries": ["United States,Canada", "France", "CANADA, France", None],
        "categories": ["Beverages,Plant-based foods", "Beverages", "Snacks", "Beverages"],
    })

    country_only = filter_off_export(frame, selected_countries=[" Canada "])
    assert country_only["code"].tolist() == ["1", "3"]

    category_only = filter_off_export(frame, selected_categories=["beverages"])
    assert category_only["code"].tolist() == ["1", "2", "4"]

    combined = filter_off_export(
        frame,
        selected_countries=["canada", "FRANCE"],
        selected_categories=["Beverages", "snacks"],
    )
    assert combined["code"].tolist() == ["1", "2", "3"]
