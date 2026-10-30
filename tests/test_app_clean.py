"""Cache-only dashboard regression checks; never start a server or worker."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clean_dashboard", ROOT / "app_clean.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def healthy(app):
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]


@pytest.fixture(scope="module", autouse=True)
def preserve_original_files():
    """Assert existing protected code/notebooks/config/models are not changed by AppTest."""
    paths = {ROOT / "app.py"}
    for folder in ["src", "part2", "configs", "models"]:
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update(ROOT.glob("*.ipynb"))
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    original_hash = hashlib.sha256((ROOT / "app.py").read_bytes()).hexdigest()
    yield
    assert all(p.exists() and (p.stat().st_size, p.stat().st_mtime_ns) == state for p, state in before.items())
    assert hashlib.sha256((ROOT / "app.py").read_bytes()).hexdigest() == original_hash


def test_imports_are_read_only_and_never_import_original_or_model_modules():
    tree = ast.parse((ROOT / "app_clean.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(name.split(".")[0] in {"app", "part2", "torch", "transformers", "onnxruntime", "easyocr", "requests"}
                   for name in modules)
    assert "app" not in sys.modules


@pytest.mark.parametrize("dataset", clean.DATASETS)
def test_all_sections_and_original_annotation_schema(dataset):
    app = AppTest.from_file(str(ROOT / "app_clean.py"), default_timeout=90).run()
    healthy(app)
    assert app.radio[0].options == clean.SECTIONS
    assert app.radio[0].value == "View Raw Data"
    assert widget(app, "selectbox", "Dataset").options == clean.DATASETS
    widget(app, "selectbox", "Dataset").set_value(dataset).run()
    healthy(app)
    expected = 50_000 if dataset == clean.DATASETS[0] else 3075
    assert any(f"{expected:,} / {expected:,} product rows" in c.value for c in app.caption)
    for section in clean.SECTIONS:
        app.radio[0].set_value(section).run()
        healthy(app)
        if section in ["View Raw Data", "Part 2 Dataset Info"]:
            original = next(t.value for t in app.dataframe if "gold_bio_id" in t.value)
            assert {"ocr_token", "bounding_box", "gold_bio_id", "gold_bio_label"}.issubset(original)
            key = widget(app, "selectbox", "Image").value
            pd.testing.assert_frame_equal(original, clean.annotation_preview(clean.LINKED, key, limit=1_000_000).sort_values("token_index"))
        if section == "Results":
            metrics = next(t.value for t in app.dataframe if "processed_images" in t.value)
            assert metrics["target_images"].sum() == 3075


@pytest.mark.parametrize("dataset", clean.DATASETS)
def test_filters_apply_every_exploratory_page_and_exact_part2_scope(dataset):
    app = AppTest.from_file(str(ROOT / "app_clean.py"), default_timeout=90).run()
    widget(app, "selectbox", "Dataset").set_value(dataset).run()
    products, inventory, _ = clean.load_linked(clean.LINKED)
    frame = (pd.read_csv(ROOT / "data/openfoodfacts_eda_enriched.csv", dtype={"code": "string"}, low_memory=False)
             if dataset == clean.DATASETS[0] else clean.products_for_eda(products))
    both = frame.loc[frame["countries"].notna() & frame["categories"].notna()]
    example = both.iloc[0]
    country = sorted(clean.label_set(example["countries"]))[0]
    category = sorted(clean.label_set(example["categories"]))[0]
    widget(app, "multiselect", "Country labels").set_value([country]).run()
    widget(app, "multiselect", "Category labels").set_value([category]).run()
    filtered = clean.filter_products(frame, [country], [category])
    expected_images = inventory.loc[inventory["barcode"].isin(filtered["code"])]
    assert 0 < len(filtered) < len(frame)
    for section in clean.SECTIONS[:6]:
        app.radio[0].set_value(section).run()
        healthy(app)
        assert any(f"{len(filtered):,} / {len(frame):,} product rows" in c.value for c in app.caption)
        if section == "View Raw Data":
            # Streamlit's Arrow transport normalizes object-column nulls to None.
            expected = filtered.head(200).copy()
            for name in expected.select_dtypes(include="object"):
                expected[name] = expected[name].where(expected[name].notna(), None)
            pd.testing.assert_frame_equal(app.dataframe[0].value, expected)
        if section == "Part 2 Dataset Info":
            assert next(m.value for m in app.metric if m.label == "Images in scope") == f"{len(expected_images):,}"
            if not expected_images.empty:
                assert set(widget(app, "selectbox", "Image").options) == set(expected_images["image_key"])


def test_filter_semantics_leading_zero_and_empty_intersection():
    products = pd.DataFrame(dict(code=["001", "1", "002", "003"],
                                 countries=["France,Belgium", "France", "Canada", None],
                                 categories=["Snacks", "Drinks", "Snacks", "Snacks"]))
    filtered = clean.filter_products(products, ["FRANCE", "canada"], ["snacks"])
    assert filtered["code"].tolist() == ["001", "002"]
    images = pd.DataFrame(dict(barcode=["001", "1", "999"]))
    assert clean.scoped_images(images, filtered, True)["barcode"].tolist() == ["001"]
    assert len(clean.scoped_images(images, filtered, False)) == 3
    assert clean.scoped_images(images, filtered.iloc[:0], True).empty


def test_bio_sources_split_selectors_and_collapsed_details():
    app = AppTest.from_file(str(ROOT / "app_clean.py"), default_timeout=90).run()
    app.radio[0].set_value("BIO Entity Tagging").run()
    for source in clean.SOURCES:
        widget(app, "selectbox", "Cached predictions source").set_value(source).run()
        healthy(app)
        assert {m.label for m in app.metric} >= {"Completed images", "Pending images", "Failed images", "Run status"}
        assert any(set(["nutrient", "value", "unit", "basis"]).issubset(t.value) for t in app.dataframe)
        advanced = next(e for e in app.expander if e.label.startswith("Advanced"))
        assert not advanced.proto.expanded
        splits = widget(app, "selectbox", "Split").options
        for split in splits[1:]:
            widget(app, "selectbox", "Split").set_value(split).run()
            healthy(app)
            assert all(key.startswith(split + ":") for key in widget(app, "selectbox", "Image").options)
        # Exercise a second selected image, including source switching with stale widget state.
        selector = widget(app, "selectbox", "Image")
        if len(selector.options) > 1:
            selector.set_value(selector.options[-1]).run()
            healthy(app)


def test_results_fixed_under_filters_and_dataset_changes():
    app = AppTest.from_file(str(ROOT / "app_clean.py"), default_timeout=90).run()
    app.radio[0].set_value("Results").run()
    for source in clean.SOURCES:
        widget(app, "selectbox", "Cached predictions source").set_value(source).run()
        healthy(app)
        # The approach comparison precedes results; identify scores by schema.
        baseline = next(t.value for t in app.dataframe if "processed_images" in t.value).copy()
        for dataset in clean.DATASETS:
            widget(app, "selectbox", "Dataset").set_value(dataset).run()
            country = widget(app, "multiselect", "Country labels").options[0]
            widget(app, "multiselect", "Country labels").set_value([country]).run()
            healthy(app)
            # A separate worker may publish between reruns: targets must remain fixed,
            # and stable snapshots must retain exactly the same metrics.
            actual = next(t.value for t in app.dataframe if "processed_images" in t.value)
            pd.testing.assert_series_equal(baseline["target_images"], actual["target_images"])
            if baseline["processed_images"].equals(actual["processed_images"]):
                pd.testing.assert_frame_equal(baseline, actual)
        assert any("True positive" in m.value and "False negative" in m.value for m in app.markdown)


def test_empty_filters_are_safe_on_all_pages():
    app = AppTest.from_file(str(ROOT / "app_clean.py"), default_timeout=90).run()
    countries = widget(app, "multiselect", "Country labels").options
    categories = widget(app, "multiselect", "Category labels").options
    frame = pd.read_csv(ROOT / "data/openfoodfacts_eda_enriched.csv", dtype={"code": "string"}, low_memory=False)
    country = countries[0]
    country_frame = clean.filter_products(frame, [country], [])
    available_categories = set(clean.label_choices(country_frame, "categories"))
    category = next(c for c in categories if c not in available_categories)
    widget(app, "multiselect", "Country labels").set_value([country]).run()
    widget(app, "multiselect", "Category labels").set_value([category]).run()
    for section in clean.SECTIONS:
        app.radio[0].set_value(section).run()
        healthy(app)
        assert any("No product rows match" in i.value for i in app.info)


def test_unprocessed_splits_are_blank_not_zero():
    summary = dict(target_splits={"test": 199, "train": 2876}, metrics_by_split={
        "train": dict(images=0, precision=0, recall=0, f1=0, token_accuracy=0),
        "test": dict(images=1, true_positive_spans=2, predicted_spans=3, gold_spans=4,
                     precision=2/3, recall=.5, f1=4/7, token_accuracy=.9, token_support=10)})
    rows = clean.metric_rows(summary).set_index("split")
    assert rows.loc["train", ["precision", "recall", "F1", "token_accuracy", "token_support"]].isna().all()
    assert rows.loc["test", "false_positives"] == 1
    assert rows.loc["test", "false_negatives"] == 2


def test_annotation_mismatch_or_partial_coverage_is_rejected_without_rewriting():
    annotations = pd.DataFrame(dict(ocr_token=["12", "g"], bounding_box=[[0,0,1,1], [1,1,2,2]],
                                    gold_bio_label=["B-FAT_100G", "I-FAT_100G"]))
    payload = dict(barcode="001", tokens=["12", "g"], bboxes=[[0,0,1,1], [1,1,2,2]],
                   gold_labels=["B-FAT_100G", "I-FAT_100G"], predicted_labels=["O", "O"],
                   coverage_mask=[True, True], covered_token_count=2)
    clean.checked_prediction(payload, {"barcode": "001"}, annotations)
    payload["gold_labels"] = ["O", "O"]
    before = json.dumps(payload)
    with pytest.raises(ValueError, match="gold differ"):
        clean.checked_prediction(payload, {"barcode": "001"}, annotations)
    assert json.dumps(payload) == before
    payload["gold_labels"] = annotations["gold_bio_label"].tolist()
    payload["coverage_mask"] = [True, False]
    with pytest.raises(ValueError, match="incomplete"):
        clean.checked_prediction(payload, {"barcode": "001"}, annotations)


def test_missing_and_incompatible_cache_states(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "FULL", tmp_path)
    monkeypatch.setattr(clean, "BUNDLE", tmp_path / "bundle.json")
    assert clean.read_results(clean.SOURCES[0])[0] is None
    assert clean.read_results(clean.SOURCES[1])[0] is None
    (tmp_path / "summary.json").write_text(json.dumps({"inference_schema_version": "old"}))
    with pytest.raises(ValueError, match="incompatible"):
        clean.read_results(clean.SOURCES[0])
    (tmp_path / "bundle.json").write_text(json.dumps({"inference_schema_version": "old"}))
    with pytest.raises(ValueError, match="incompatible"):
        clean.read_results(clean.SOURCES[1])


def test_quality_rules_do_not_apply_grams_limit_to_energy():
    frame = pd.DataFrame(dict(code=["001", "001", None], fat_100g=[101, -1, "bad"],
                              **{"energy-kcal_100g": [500, 300, None]}, countries=["", None, "France"]))
    missing, anomalies = clean.quality_tables(frame)
    flags = anomalies.set_index("source_field")
    assert flags.loc["fat_100g", "above_100g"] == 1
    assert flags.loc["fat_100g", "negative"] == 1
    assert flags.loc["fat_100g", "invalid_numeric"] == 1
    assert pd.isna(flags.loc["energy-kcal_100g", "above_100g"])
    assert missing.set_index("source_field").loc["countries", "missing_rows"] == 2


def test_full_cohort_published_metrics_are_actual_micro_counts():
    summary, rows, _ = clean.read_results(clean.SOURCES[0])
    assert summary is not None
    assert not rows["image_key"].duplicated().any()
    assert len(rows) == summary["completed_images"]
    assert summary["completed_images"] + summary["pending_images"] + summary["failed_images"] == summary["target_images"]
    for split, m in summary["metrics_by_split"].items():
        selected = rows.loc[rows["split"].eq(split)]
        assert len(selected) == m["images"]
        if selected.empty:
            continue
        tp = selected["true_positive_spans"].sum()
        predicted = selected["predicted_spans"].sum()
        gold = selected["gold_spans"].sum()
        tokens = selected["token_support"].sum()
        assert tp == m["true_positive_spans"]
        assert predicted == m["predicted_spans"]
        assert gold == m["gold_spans"]
        assert tokens == m["token_support"]
        assert m["precision"] == pytest.approx(tp / predicted)
        assert m["recall"] == pytest.approx(tp / gold)
        assert m["f1"] == pytest.approx(2 * tp / (predicted + gold))
        assert m["token_accuracy"] == pytest.approx(selected["correct_tokens"].sum() / tokens)


def test_seeded_published_metrics_match_cached_exact_spans():
    """Independent count check on saved tags only; no model or inference imports."""
    summary, _, bundle = clean.read_results(clean.SOURCES[1])
    assert summary is not None

    def spans(labels):
        result = set()
        active = None
        start = 0
        for i, label in enumerate([*labels, "O"]):
            prefix, sep, kind = label.partition("-")
            entity = bool(sep) and prefix in {"B", "I"}
            if not entity or prefix == "B" or active != kind:
                if active is not None:
                    result.add((active, start, i))
                active = kind if entity else None
                start = i
        return result

    tp = predicted = gold = correct = tokens = 0
    for sample in bundle["samples"]:
        mask = sample["coverage_mask"]
        assert len(mask) == len(sample["tokens"]) and all(mask)
        reference = spans(sample["gold_labels"])
        estimates = spans(sample["predicted_labels"])
        tp += len(reference & estimates)
        predicted += len(estimates)
        gold += len(reference)
        correct += sum(a == b for a, b in zip(sample["gold_labels"], sample["predicted_labels"]))
        tokens += len(sample["tokens"])
    m = bundle["fine_grained_metrics"]
    assert m["true_positive_spans"] == tp
    assert m["predicted_spans"] == predicted
    assert m["gold_spans"] == gold
    assert m["precision"] == pytest.approx(tp / predicted)
    assert m["recall"] == pytest.approx(tp / gold)
    assert m["f1"] == pytest.approx(2 * tp / (predicted + gold))
    assert m["token_accuracy"] == pytest.approx(correct / tokens)