"""Nutrient explanations and saved-sample evidence; no server or inference."""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_nutrient_explanations", ROOT / "app_final.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def test_saved_sample_facts_are_verified_not_supplied_stale_median():
    path = ROOT / "data/openfoodfacts_eda_enriched.csv"
    frame = pd.read_csv(path, dtype={"code": "string"}, low_memory=False)
    facts = clean.sample_nutrient_examples(clean.fingerprint([path]))
    assert facts == clean.nutrient_example_facts(frame)
    assert facts["rows"] == 50_000
    assert facts["nutrient_fields"] == 27
    assert facts["energy_count"] == 47_759
    assert facts["energy_median"] == 1350  # Not the user-supplied 1088.
    assert facts["energy_mean"] == pytest.approx(2434.831661249001)
    assert facts["energy_max"] == 680_000
    assert facts["carbs_count"] == 47_687
    assert facts["carbs_median"] == 25.5
    assert facts["carbs_above_100"] == 7857
    assert facts["energy_correlation"] == pytest.approx(.8307193001789009)
    assert facts["energy_pairs"] == 47_748
    assert facts["exact_duplicates"] == facts["duplicate_codes"] == 0
    assert facts["missing_names"] == 261
    assert facts["any_nutrition"] == 47_775
    assert 100 * facts["any_nutrition"] / facts["rows"] == 95.55
    assert facts["micronutrient_missing"]["vitamin-d_100g"] == 42_242


def test_helper_counts_only_finite_nutrients_and_blank_names_without_mutation():
    frame = pd.DataFrame({
        "code": ["001", "001", " ", None, "002", "003"],
        "product_name": ["Rice", "Rice", "  ", None, "Name", ""],
        "energy_100g": [10, 10, np.inf, "bad", 0, None],
        "energy-kcal_100g": [2, 2, -np.inf, "bad", 0, None],
        "carbohydrates_100g": [101, 101, None, None, "0", None],
        "vitamin-d_100g": [None, None, np.inf, "bad", "0.01", None],
        "not_a_nutrient": [1, 1, 1, 1, 1, 1],
    })
    before = frame.copy(deep=True)
    facts = clean.nutrient_example_facts(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert facts["any_nutrition"] == 3
    assert facts["energy_count"] == facts["energy_pairs"] == 3
    assert facts["duplicate_codes"] == facts["exact_duplicates"] == 1
    assert facts["missing_names"] == 3
    assert facts["carbs_above_100"] == 2
    assert facts["micronutrient_missing"]["vitamin-d_100g"] == 5


def test_cached_examples_use_saved_sample_and_invalidate_with_stamp(monkeypatch):
    clean.sample_nutrient_examples.clear()
    loader = MagicMock(return_value=pd.DataFrame({
        "energy_100g": [1, 2, 3], "energy-kcal_100g": [1, 2, 3],
        "carbohydrates_100g": [1, 2, 3],
    }))
    monkeypatch.setattr(clean, "load_sample", loader)
    first = clean.sample_nutrient_examples((("sample", 1, 100),))
    assert clean.sample_nutrient_examples((("sample", 1, 100),))["energy_median"] == first["energy_median"]
    loader.assert_called_once_with((("sample", 1, 100),))
    clean.sample_nutrient_examples((("sample", 2, 101),))
    assert loader.call_count == 2
    clean.sample_nutrient_examples.clear()


@pytest.mark.parametrize("kind, trace, color", [
    ("Histogram", "histogram", "#64B5F6"),
    ("Box plot", "box", "#62D6BD"),
    ("Violin plot", "violin", "#B699FF"),
])
def test_explanations_preserve_charts_summary_and_correlation(monkeypatch, kind, trace, color):
    ui = MagicMock()
    ui.selectbox.side_effect = ["fat_100g", kind]
    ui.multiselect.return_value = ["fat_100g", "sugars_100g"]
    monkeypatch.setattr(clean, "st", ui)
    figures = MagicMock()
    monkeypatch.setattr(clean, "chart", figures)
    frame = pd.DataFrame({"fat_100g": [1, 1, 2, 3], "sugars_100g": [2, 3, 4, 5]})
    clean.render_nutrients(frame)
    statistics = ui.dataframe.call_args.args[0]
    expected = clean.numeric_nutrients(frame).describe().T.reset_index(names="source_field")
    pd.testing.assert_frame_equal(statistics.drop(columns="Mode"), expected)
    assert statistics.iloc[0]["Mode"] == "1"
    distribution, correlation = [call.args[0] for call in figures.call_args_list]
    assert distribution.data[0].type == trace
    assert distribution.data[0].marker.color == color
    np.testing.assert_allclose(correlation.data[0].z, clean.numeric_nutrients(frame).corr(min_periods=3))
    text = "\n".join(str(call.args[0]) for call in ui.mock_calls if call.args)
    for phrase in ["nutritional characteristics", "active dataset after filters", "skew", "median (50%)",
                   "standard deviation", "min/max", "Mode", "not causation", "Unexpected relationships",
                   "20 g carbs / 5 g sugars", "60 g carbs / 35 g sugars", "rice", "starch",
                   "nor a perfect correlation is guaranteed", "duplicate product codes", "missing product names", "micronutrient",
                   "does **not directly improve BIO tagging**", "Presentation-ready summary",
                   "50K sample examples · not changed by filters", "1,350", "42,242"]:
        assert phrase in text
    assert all(not call.kwargs["expanded"] for call in ui.expander.call_args_list)


@pytest.mark.parametrize("dataset", clean.DATASETS)
def test_fixed_examples_stay_separate_from_filtered_active_statistics(dataset):
    app = AppTest.from_file(str(ROOT / "app_final.py"), default_timeout=90).run()
    next(w for w in app.selectbox if w.label == "Dataset").set_value(dataset).run()
    app.radio[0].set_value("Nutrients").run()
    assert not app.exception
    baseline = next(e for e in app.expander if e.label == "50K sample examples · not changed by filters")
    example_text = [m.value for m in baseline.markdown]
    before = app.dataframe[0].value.copy()
    countries = next(w for w in app.multiselect if w.label == "Country labels")
    countries.set_value([countries.options[0]]).run()
    assert not app.exception and not app.error
    fixed = next(e for e in app.expander if e.label == baseline.label)
    assert [m.value for m in fixed.markdown] == example_text
    assert not app.dataframe[0].value.equals(before)
    assert "Mode" in app.dataframe[0].value
    assert len(app.get("plotly_chart")) == 2