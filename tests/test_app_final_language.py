"""Plain-language display contracts; no server, OCR or model execution."""
import importlib.util
import json
from pathlib import Path
import re
from unittest.mock import MagicMock

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_language_dashboard", ROOT / "app_final.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


@pytest.fixture
def ui(monkeypatch):
    display = MagicMock()
    monkeypatch.setattr(clean, "st", display)
    monkeypatch.setattr(clean, "cards", MagicMock())
    return display


def copy_text(ui):
    return "\n".join(str(call.args[0]) for call in ui.mock_calls if call.args)


def summary(status="paused_low_memory"):
    return dict(completed_images=1, pending_images=2, failed_images=1,
                target_images=4, status=status, updated_at_utc="saved time",
                target_splits={"test": 2, "train": 2}, metrics_by_split={
                    "test": dict(images=1, precision=.5, recall=.25, f1=1/3,
                                 token_accuracy=.8, token_support=10,
                                 true_positive_spans=1, predicted_spans=2, gold_spans=4)})


def test_friendly_table_headings_do_not_rename_backend_columns(ui):
    frame = pd.DataFrame({"source_field": ["fat_100g"], "iqr_outliers": [2]})
    before = frame.copy(deep=True)
    clean.table(frame, {"iqr_outliers": "Unusually low/high (IQR)"})
    pd.testing.assert_frame_equal(frame, before)
    assert ui.dataframe.call_args.args[0] is frame
    assert ui.dataframe.call_args.kwargs["column_config"]["iqr_outliers"] == "Unusually low/high (IQR)"


def test_results_explain_scores_and_preserve_metric_rows(ui):
    saved = summary()
    expected = clean.metric_rows(saved)
    clean.render_results(saved)
    pd.testing.assert_frame_equal(ui.dataframe.call_args.args[0], expected)
    text = copy_text(ui)
    for phrase in ["How many model answers were right", "How many reference answers were found",
                   "balanced score", "including O", "Exact entity match", "not independently verified",
                   "False positive", "False negative", "training history", "fixed target",
                   "not OCR reading quality", "not averaged image by image"]:
        assert phrase in text
    headings = ui.dataframe.call_args.kwargs["column_config"]
    assert headings["token_accuracy"] == "Word-label accuracy"
    assert "processed_images" in expected and pd.isna(expected.iloc[1]["precision"])


@pytest.mark.parametrize("status, explanation", [
    ("paused_limit", "image limit"), ("paused_low_memory", "free memory"),
    ("paused_model_load_error", "model could not be loaded"),
    ("interrupted", "interrupted"), ("running", "does not prove a worker"),
    ("incomplete_with_failures", "images that failed"),
    ("cached_subset", "small test group"), ("unknown", "no live worker check"),
])
def test_status_explains_saved_reason_and_pending_vs_failed(ui, status, explanation):
    clean.render_status(summary(status))
    text = copy_text(ui)
    assert explanation in text
    assert "Pending = not yet completed" in text
    assert "Failed = an attempt ended with an error" in text
    assert "Partial results" in text and "not scored as zero" in text
    assert "cannot start or resume" in text
    assert ui.progress.call_args.args[0] == .25


def test_bio_has_concrete_word_example_even_without_predictions(ui):
    clean.render_bio(pd.DataFrame(), None, pd.DataFrame(), None)
    text = copy_text(ui)
    for phrase in ["Fat 12 g", "B-FAT_100G", "I-FAT_100G", "outside any marked answer",
                   "not independently checked"]:
        assert phrase in text


def test_approaches_explain_fair_comparison_and_overlapping_windows(ui, monkeypatch, tmp_path):
    monkeypatch.setattr(clean, "BUNDLE", tmp_path / "absent.json")
    clean.render_approaches()
    text = copy_text(ui)
    for phrase in ["hand-written Python", "already-trained", "Basic Nutrient Detection", "Detailed Nutrient Identification",
                   "How the Methods Were Compared", "No machine learning", "not retrained",
                   "serving-size group becomes BASIS", "30 g", "not automatically a reference BASIS",
                   "12 g → Fat per100g", "not a separately tagged reference phrase", "Nutrient-name words",
                   "same entity type and first/last word boundaries", "correct span does not prove a correct number",
                   "Needs review", "no guessing", "partial results are not final",
                   "Overlapping windows", "share some words", "not another model", "no second model",
                   "no matching reference BIO labels", "Nothing is run"]:
        assert phrase in text
    assert not re.search(r"\b(coarse|fine(?:[- ]grained)?|granularity|like-for-like)\b|canvascanvas", text, re.I)
    assert ui.expander.call_args.kwargs["expanded"] is False


def test_comparison_separates_tasks_and_preserves_saved_scores(ui):
    before = clean.BUNDLE.read_bytes()
    saved = json.loads(before)
    clean.render_approaches()
    tables = [call.args[0] for call in ui.dataframe.call_args_list]
    assert [len(frame) for frame in tables] == [2, 1]
    assert tables[0]["Approach / task"].tolist() == [
        "Rules · Basic Nutrient Detection", "LayoutLMv3 · Basic Nutrient Detection"]
    assert tables[1]["Approach / task"].tolist() == ["LayoutLMv3 · Detailed Nutrient Identification"]
    for row, key, expected in zip(
        [tables[0].iloc[0], tables[0].iloc[1], tables[1].iloc[0]],
        ["coarse_rule_metrics", "coarse_model_metrics", "fine_grained_metrics"],
        [(46.8, 79.6, 58.9), (97.7, 99.5, 98.6), (92.6, 95.3, 93.9)],
    ):
        for column, metric, percent in zip(["Precision", "Recall", "F1"], ["precision", "recall", "f1"], expected):
            assert row[column] == f"{saved[key][metric]:.1%}" == f"{percent:.1f}%"
    text = copy_text(ui)
    assert "20 selected official-test images" in text
    assert "results-source selector and product filters do not change" in text
    assert "not independently verified" in text
    assert clean.BUNDLE.read_bytes() == before
    assert not re.search(r"\b(coarse|fine(?:[- ]grained)?|granularity|like-for-like)\b|canvascanvas", text, re.I)


def test_results_use_requested_heading_without_changing_scores(ui):
    clean.render_results(summary())
    assert ui.subheader.call_args.args[0] == "Understanding the Results"
    text = copy_text(ui)
    assert "Detailed Nutrient Identification scores" in text
    assert not re.search(r"\b(coarse|fine(?:[- ]grained)?|granularity|like-for-like)\b|canvascanvas", text, re.I)


def test_units_filters_and_unknown_values_are_explained(ui):
    clean.render_understanding(pd.DataFrame({"code": ["001"]}), pd.DataFrame(), clean.DATASETS[0])
    text = copy_text(ui)
    for phrase in ["per100g = per 100 g", "perServing = per portion", "100 ml for liquids",
                   "unknown, not zero", "first 50,000", "not a random sample", "Leading zeros",
                   "No units are converted", "complete saved Part 1 export"]:
        assert phrase in text


def test_recent_nutrient_charts_mode_and_large_correlation_are_preserved(ui, monkeypatch):
    frame = pd.DataFrame({"fat_100g": [1, 1, 2, 3], "sugars_100g": [2, 3, 4, 5]})
    ui.selectbox.side_effect = ["fat_100g", "Violin plot"]
    ui.multiselect.return_value = ["fat_100g", "sugars_100g"]
    figures = MagicMock()
    monkeypatch.setattr(clean, "chart", figures)
    clean.render_nutrients(frame)
    statistics = ui.dataframe.call_args.args[0]
    assert "Mode" in statistics and statistics.iloc[0]["Mode"] == "1"
    assert figures.call_args_list[0].args[0].data[0].type == "violin"
    assert figures.call_args_list[1].kwargs["height"] >= 650
    assert figures.call_args_list[1].args[0].data[0].type == "heatmap"


def test_quality_keeps_iqr_column_and_plain_language_caution(ui, monkeypatch):
    monkeypatch.setattr(clean, "chart", MagicMock())
    clean.render_quality(pd.DataFrame({"code": ["001", "002"], "fat_100g": [1, 101]}))
    assert "iqr_outliers" in ui.dataframe.call_args.args[0]
    text = copy_text(ui)
    assert "middle half" in text and "not proven wrong" in text
    assert "Nothing is repaired or removed" in text


def test_raw_product_and_annotation_names_are_unchanged(ui, monkeypatch):
    frame = pd.DataFrame({"code": ["001"], "fat_100g": [12]})
    render_original = clean.render_original
    monkeypatch.setattr(clean, "render_original", MagicMock())
    clean.render_raw(frame, pd.DataFrame())
    pd.testing.assert_frame_equal(ui.dataframe.call_args.args[0], frame)
    assert ui.dataframe.call_args.kwargs["column_config"] == {}
    assert "code = barcode" in copy_text(ui)

    annotations = pd.DataFrame({"ocr_token": ["12"], "bounding_box": [[0, 0, 1, 1]],
                                "gold_bio_id": [1], "gold_bio_label": ["B-FAT_100G"]})
    monkeypatch.setattr(clean, "choose_images", lambda *args: {"image_key": "test:0", "barcode": "001"})
    monkeypatch.setattr(clean, "fingerprint", lambda *args: ())
    monkeypatch.setattr(clean, "original_annotations", lambda *args: annotations)
    render_original(pd.DataFrame(), "raw")
    pd.testing.assert_frame_equal(ui.dataframe.call_args.args[0], annotations)
    assert ui.dataframe.call_args.kwargs["column_config"] == {}
    assert "gold_bio_label = reference word label" in copy_text(ui)