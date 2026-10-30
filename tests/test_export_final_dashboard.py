"""Offline artifact and capture-adapter checks; no servers or inference.

Generate the artifact before running this focused test file. Assertions read the
frozen snapshot, never assume that the full-cohort worker has finished.
"""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NUTRITION_DATA_DIR", str(ROOT / "data")))
SPEC = importlib.util.spec_from_file_location("export_dashboard2", ROOT / "scripts/export_final_dashboard.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
SECTIONS = ["View Raw Data", "Data Overview", "Data Understanding", "Nutrients",
            "Data Quality", "Part 2 Dataset Info", "BIO Entity Tagging", "Results"]


@pytest.fixture(scope="module")
def artifact():
    path = ROOT / "main_output/index.html"
    assert path.is_file(), "Generate Dashboard2 with scripts/export_final_dashboard.py first"
    html = path.read_text(encoding="utf-8")
    start = '<script id="dashboard2-snapshot" type="application/json">'
    text = html.split(start, 1)[1].split("</script>", 1)[0]
    return html, json.loads(text), path


def events_text(events):
    return "\n".join(str(e.get("html", e.get("text", ""))) for e in events)


@pytest.mark.parametrize("value", [
    "</script><script>alert('OCR')</script>",
    '<img src="https://example.invalid/image" onerror="alert(1)">',
    "A&B > C < D \u2028 \u2029 café 栄養", "` ${template} \\ slash / quote \"",
])
def test_json_script_escaping_roundtrips(value):
    encoded = exporter.script_json({"ocr": value})
    assert "<" not in encoded and ">" not in encoded and "&" not in encoded
    assert json.loads(encoded) == {"ocr": value}


def test_json_rejects_nonfinite():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            exporter.script_json(value)


def test_recorder_columns_context_and_table_escaping(tmp_path):
    recorder = exporter.Recorder(tmp_path)
    recorder.columns(2)[0].metric("Rows", "2")
    with recorder.expander("Details"):
        recorder.dataframe(pd.DataFrame({"ocr_token": ["</script><img src=x>"]}),
                           column_config={"ocr_token": "Supplied word"})
    assert [e["kind"] for e in recorder.events] == ["metric", "details_start", "table", "details_end"]
    table = recorder.events[2]["html"]
    assert "Supplied word" in table and "&lt;img" in table and "<img" not in table


def test_download_deduplication_is_lossless(tmp_path):
    recorder = exporter.Recorder(tmp_path)
    content = 'barcode,word\n001234,"é <nutrition>"\n'.encode()
    for _ in range(2):
        recorder.download_button("Download", content, "example.csv", "text/csv")
    assert len(recorder.downloads) == 1
    saved = next(iter(recorder.downloads.values()))
    assert gzip.decompress(base64.b64decode(saved["data"])) == content


def test_capture_restores_state_on_error(tmp_path):
    recorder = exporter.Recorder(tmp_path)
    recorder.choices = {"old": "value"}
    original_events = recorder.events

    def broken():
        recorder.caption("Not committed")
        raise ValueError("Bad source cache")

    with pytest.raises(ValueError, match="Bad source cache"):
        recorder.capture(broken, choices={"new": "value"})
    assert recorder.events is original_events and recorder.events == []
    assert recorder.choices == {"old": "value"}


def test_exact_navigation_dataset_counts_and_source_fingerprint(artifact):
    _, data, path = artifact
    assert data["sections"] == SECTIONS
    assert data["datasets"] == ["50K Dataset", "Part1 + Part2 Overlapped Dataset"]
    assert data["views"]["0"]["row_count"] == 50_000
    assert data["views"]["1"]["row_count"] == 3075
    assert sum(data["inventory_splits"].values()) == 3083
    # Frozen snapshots retain their historical source filename across a rename.
    assert len(data["source_sha256"]) == 1
    assert next(iter(data["source_sha256"].values())) == exporter.file_sha256(ROOT / "app_final.py")
    assert path.stat().st_size < 50 * 1024**2
    for views in data["views"].values():
        assert all(section in views for section in SECTIONS[:6])
    assert data["chart_spec_count"] >= 170
    static_plots = sum(e["kind"] == "plot" for views in data["views"].values()
                       for events in views.values() if isinstance(events, list) for e in events)
    distribution_plots = sum(len(plots) for variants in data["distributions"].values() for plots in variants.values())
    assert data["chart_spec_count"] == static_plots + distribution_plots
    print(f"\nDashboard2: {path.stat().st_size:,} bytes; {data['chart_spec_count']} chart specs; "
          f"{len(data['images'])} embedded thumbnails; {len(data['originals'])} original annotation selectors; "
          f"nutrient fields per dataset: {[len(v) for v in data['distributions'].values()]}")


def test_every_nutrient_has_exact_three_graph_types_and_black_palette(artifact):
    _, data, _ = artifact
    count = 0
    for dataset, variants in data["distributions"].items():
        assert len(variants) >= 20
        for field, plots in variants.items():
            assert list(plots) == ["Histogram", "Box plot", "Violin plot"]
            for kind, trace_type, color, axis in [
                ("Histogram", "histogram", "#64B5F6", "x"),
                ("Box plot", "box", "#62D6BD", "y"),
                ("Violin plot", "violin", "#B699FF", "y"),
            ]:
                spec = plots[kind]
                count += 1
                assert spec["data"][0]["type"] == trace_type
                assert spec["data"][0]["marker"]["color"] == color
                assert spec["layout"]["paper_bgcolor"] == "#000000"
                assert spec["layout"]["plot_bgcolor"] == "#000000"
                assert spec["layout"]["font"]["color"] == "#C5D0DC"
                assert spec["data"][0][axis] == {"$values": f"{dataset}:{field}"}
                assert len(data["values"][f"{dataset}:{field}"]) > 0
                if kind == "Histogram":
                    assert spec["data"][0]["nbinsx"] == 35
                if kind == "Violin plot":
                    assert spec["data"][0]["box"]["visible"] is True
    assert count >= 162


def test_distributions_preserve_all_finite_values_zeros_and_extremes(artifact):
    _, data, _ = artifact
    sample = pd.read_csv(DATA_DIR / "openfoodfacts_eda_enriched.csv", low_memory=False)
    for field in data["distributions"]["0"]:
        actual = pd.to_numeric(sample[field], errors="coerce").replace([float("inf"), -float("inf")], float("nan")).dropna().tolist()
        assert data["values"][f"0:{field}"] == actual
    assert max(data["values"]["0:energy_100g"]) == 680000
    assert 0 in data["values"]["0:fat_100g"]


def test_overview_missingness_numeric_flags_and_full_large_correlation(artifact):
    _, data, _ = artifact
    for dataset, views in data["views"].items():
        overview = [e["spec"] for e in views["Data Overview"] if e["kind"] == "plot"]
        assert len(overview) == 3
        assert [p["layout"]["title"]["text"] for p in overview] == [
            "Country labels", "Category labels", "How often nutrient values are available"]
        nutrient_events = views["Nutrients"]
        heatmaps = [e["spec"] for e in nutrient_events if e["kind"] == "plot"]
        assert len(heatmaps) == 1 and heatmaps[0]["data"][0]["type"] == "heatmap"
        assert len(heatmaps[0]["data"][0]["x"]) == len(data["distributions"][dataset])
        assert heatmaps[0]["layout"]["height"] >= 650
        assert heatmaps[0]["layout"]["coloraxis"]["cmin"] == -1
        assert heatmaps[0]["layout"]["coloraxis"]["cmax"] == 1
        quality = views["Data Quality"]
        assert len([e for e in quality if e["kind"] == "plot"]) == 1
        assert any({"negative", "iqr_outliers", "above_100g", "invalid_numeric"}.issubset(e.get("columns", [])) for e in quality)
        assert "What this chart tells you" in events_text(nutrient_events)
        assert "Each pair uses rows with both values present" in events_text(nutrient_events)
        assert "do not directly improve BIO tagging" in events_text(nutrient_events)
        assert "Nothing is repaired or removed" in events_text(quality)


def test_original_annotations_and_prediction_reference_tables_are_retained(artifact):
    _, data, _ = artifact
    required = {"image_key", "token_index", "ocr_token", "bounding_box", "gold_bio_id", "gold_bio_label"}
    assert {key.split(":")[0] for key in data["originals"]} == {"test", "train"}
    for events in data["originals"].values():
        assert any(required.issubset(e.get("columns", [])) and e["rows"] > 0 for e in events)
        assert "not independently verified" in events_text(events)
    assert len(data["bio"]["1"]["examples"]) == data["summaries"]["1"]["completed_images"] == 20
    for saved in data["bio"].values():
        for events in saved["examples"].values():
            assert any(required.union({"predicted_bio_label"}).issubset(e.get("columns", [])) for e in events)
            assert any({"nutrient", "value", "unit", "basis", "status"}.issubset(e.get("columns", [])) for e in events)


def test_snapshot_progress_and_split_micro_metrics_are_consistent(artifact):
    _, data, _ = artifact
    for key, summary in data["summaries"].items():
        rows = data["result_rows"][key]
        assert summary["completed_images"] == len(rows)
        assert summary["target_images"] == sum(summary["target_splits"].values())
        assert summary["target_images"] == sum(summary[k] for k in ("completed_images", "pending_images", "failed_images"))
        for split, m in summary["metrics_by_split"].items():
            subset = [r for r in rows if r["split"] == split]
            assert m["images"] == len(subset)
            if key == "0" and subset:
                tp = sum(r["true_positive_spans"] for r in subset)
                predicted = sum(r["predicted_spans"] for r in subset)
                gold = sum(r["gold_spans"] for r in subset)
                assert m["precision"] == pytest.approx(tp / predicted)
                assert m["recall"] == pytest.approx(tp / gold)
        text = events_text(data["results"][key])
        assert "Train vs test" in text and "not OCR reading quality" in text
        if summary["completed_images"] < summary["target_images"]:
            assert "Partial results" in text
    print("\nSaved snapshot:", {k: (s["completed_images"], s["target_images"], s["status"]) for k, s in data["summaries"].items()})


def test_comparison_copy_and_scores_match_live_rendering_in_both_sources(artifact):
    """Check actual regenerated events, including table/graph/tooltip text, not raw cache keys."""
    _, data, _ = artifact
    app = exporter.load_dashboard(ROOT)
    recorder = exporter.Recorder(ROOT)
    app.st = recorder
    expected = recorder.capture(app.render_approaches)
    expected_tables = [event for event in expected if event["kind"] == "table"]
    assert [event["rows"] for event in expected_tables] == [2, 1]
    for percent in ("46.8%", "79.6%", "58.9%", "97.7%", "99.5%", "98.6%"):
        assert percent in expected_tables[0]["html"]
    for percent in ("92.6%", "95.3%", "93.9%"):
        assert percent in expected_tables[1]["html"]
    for key in data["summaries"]:
        for events in [data["results"][key], data["bio"][key]["intro"]]:
            # Identical comparison events must survive the exporter, not a separately maintained explanation.
            assert any(events[start:start + len(expected)] == expected for start in range(len(events)))
            text = json.dumps([e for e in events if e["kind"] != "json"], ensure_ascii=False)
            for phrase in ("How the Methods Were Compared", "Basic Nutrient Detection",
                           "Detailed Nutrient Identification", "30 g", "serving-size group becomes BASIS",
                           "not automatically a reference BASIS", "12 g", "Fat per100g",
                           "not a separately tagged reference phrase", "20 selected official-test images",
                           "correct span does not prove a correct number"):
                assert phrase in text
            assert not re.search(r"\b(coarse|fine(?:[- ]grained)?|granularity|like-for-like)\b|canvascanvas", text, re.I)
        assert "Understanding the Results" in json.dumps(data["results"][key])
        details = next(e for e in expected if e.get("label") == "Details · long images and new-image reading")
        assert details["expanded"] is False
    bundle = json.loads((DATA_DIR / "part2_cache/layoutlm_bio/latest_test_bundle.json").read_text())
    assert data["summaries"]["1"]["metrics_by_split"]["test"] == dict(images=20, **bundle["fine_grained_metrics"])


def test_images_are_inline_and_hashes_match_thumbnails(artifact):
    html, data, _ = artifact
    assert len(data["images"]) >= 6
    for original_hash, image in data["images"].items():
        assert image["uri"].startswith("data:image/jpeg;base64,")
        assert image["original_sha256"] == original_hash
        assert re.fullmatch("[0-9a-f]{64}", original_hash)
        thumbnail = base64.b64decode(image["uri"].split(",", 1)[1])
        assert hashlib.sha256(thumbnail).hexdigest() == image["thumbnail_sha256"]
    assert not re.search(r'<(?:script|img|link|iframe)\b[^>]*(?:src|href)\s*=\s*[\"\x27](?:https?:|file:|/)', html, re.I)
    assert "connect-src 'none'" in html
    assert "/*__PLOTLY__*/" not in html and "__DASHBOARD2_JSON__" not in html
    assert "Bounding boxes · dataset gold vs cached predictions" in html
    assert '"kind":"bio_overlay"' in html
    assert "Both panels use the same supplied OCR word boxes" not in html
    assert "Dataset group = official train or test split" in html
    assert "Plotly.newPlot" in html and "plotly.js v" in html


def test_embedded_product_downloads_contain_all_unfiltered_rows(artifact):
    _, data, _ = artifact
    for dataset, expected in (("0", 50_000), ("1", 3075)):
        button = next(e for e in data["views"][dataset]["View Raw Data"] if e["kind"] == "download")
        saved = data["downloads"][button["digest"]]
        from io import BytesIO
        raw = gzip.decompress(base64.b64decode(saved["data"]))
        assert hashlib.sha256(raw).hexdigest() == button["digest"]
        rows = pd.read_csv(BytesIO(raw), dtype={"code": "string"}, low_memory=False)
        assert len(rows) == expected and "code" in rows


def test_html_javascript_parses_without_server(artifact):
    executable = shutil.which("node")
    if not executable:
        pytest.skip("Node is optional; install it for JavaScript syntax validation")
    html, _, _ = artifact
    inline = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(inline) == 2  # Embedded Plotly and self-contained UI controller.
    for script in inline:
        result = subprocess.run([executable, "--check", "-"], input=script, text=True,
                                capture_output=True, encoding="utf-8", check=False)
        assert result.returncode == 0, result.stderr


def test_controller_supports_required_offline_interactions():
    text = (ROOT / "src/nutrition_extraction/clean_dashboard_offline.html").read_text(encoding="utf-8")
    for feature in ("Cached predictions source", "Nutrient field (source unit and basis)",
                    "Chart type", "Saved extraction image", "Saved extraction split",
                    "Representative original image", "Original annotation split",
                    "DecompressionStream", "Plotly.purge", "Plotly.Plots.resize"):
        assert feature in text
    assert "fetch(" not in text and "XMLHttpRequest" not in text


def test_exporter_import_never_loads_models_or_original_app():
    assert not any(name.startswith(("torch", "transformers", "onnxruntime", "easyocr", "part2.layoutlm_bio")) for name in sys.modules)
    assert "app" not in sys.modules
