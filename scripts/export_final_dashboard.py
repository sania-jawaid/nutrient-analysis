"""Export Dashboard2's actual read-only render functions to one offline HTML file.

No AppTest server, model imports, inference, image fetching, or worker writes.
Run with the dashboard environment (pandas/pyarrow/plotly/Pillow/Markdown).
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys

import markdown
import pandas as pd
from PIL import Image
import plotly.express as px
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]


def script_json(value):
    """Strict JSON, safe even for adversarial OCR containing </script>."""
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dashboard(root):
    """Import only the cache-only dashboard, never app.py or Part 2 models."""
    spec = importlib.util.spec_from_file_location("dashboard2_offline_source", root / "app_final.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the read-only dashboard")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    # Do not involve Streamlit's cache/runtime in a command-line export.
    for name in ("load_inventory", "load_sample", "original_annotations", "sample_nutrient_examples"):
        function = getattr(app, name)
        setattr(app, name, getattr(function, "__wrapped__", function))
    return app


class Recorder:
    """Small explicit Streamlit output adapter; unsupported calls fail loudly.

    This calls the original render functions directly, preserving every caption,
    table heading and Plotly spec. It does not imitate Streamlit's internal tree.
    """

    def __init__(self, root):
        self.root = root
        self.events = []
        self.choices = {}
        self.downloads = {}
        self.images = {}
        self.chart_count = 0

    def emit(self, kind, **values):
        self.events.append(dict(kind=kind, **values))

    def markdown(self, text, **kwargs):
        self.emit("html", html=str(text) if kwargs.get("unsafe_allow_html") else
                  markdown.markdown(str(text), extensions=["tables", "fenced_code"]))

    def caption(self, text):
        self.emit("caption", html=markdown.markdown(str(text), extensions=["tables"]))

    def subheader(self, text):
        self.emit("heading", text=str(text))

    def metric(self, label, value):
        self.emit("metric", label=str(label), value=str(value))

    def columns(self, count):
        # cards() uses col.metric directly, not a with-column context.
        return [self for _ in range(count)]

    @contextmanager
    def expander(self, label, expanded=False):
        self.emit("details_start", label=label, expanded=expanded)
        yield self
        self.emit("details_end")

    def selectbox(self, label, options, key=None):
        options = list(options)
        chosen = self.choices.get(key, self.choices.get(label, options[0]))
        if chosen not in options:
            raise ValueError(f"Invalid snapshot selection {label}: {chosen}")
        self.emit("static_control", label=label, value=str(chosen))
        return chosen

    def multiselect(self, label, options, default=None):
        # Deliberately export the FULL correlation matrix, not just six fields.
        return list(options)

    def dataframe(self, frame, column_config=None, **kwargs):
        headings = column_config or {}
        display = frame.rename(columns={key: value for key, value in headings.items() if isinstance(value, str)})
        self.emit("table", rows=len(frame), columns=[str(c) for c in frame.columns],
                  html=display.to_html(index=False, border=0, escape=True, na_rep=""))

    def plotly_chart(self, fig, **kwargs):
        self.chart_count += 1
        self.emit("plot", spec=json.loads(fig.to_json()))

    def info(self, text):
        self.emit("notice", level="info", text=str(text))

    def warning(self, text):
        self.emit("notice", level="warning", text=str(text))

    def progress(self, value, text):
        self.emit("progress", value=value, text=text)

    def json(self, value):
        self.emit("json", text=json.dumps(value, ensure_ascii=False, indent=2))

    def link_button(self, label, url):
        # Keep the original label and provenance visible, but never link/fetch.
        self.caption(f"{label}: unavailable in the offline snapshot. Source reference: {url}")

    def download_button(self, label, data, file_name, mime):
        raw = data.encode("utf-8") if isinstance(data, str) else data
        digest = hashlib.sha256(raw).hexdigest()
        self.downloads[digest] = dict(data=base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
                                      file_name=file_name, mime=mime)
        self.emit("download", label=label, digest=digest)

    def image(self, path, **kwargs):
        if isinstance(path, Image.Image):
            preview = path.copy()
            preview.thumbnail((800, 800))
            # Bounded illustrative PNGs, not source images or model inputs.
            preview = preview.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
            stream = BytesIO()
            preview.save(stream, format="PNG")
            raw = stream.getvalue()
            self.emit("bio_overlay", caption=kwargs.get("caption", ""),
                      uri="data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
                      sha256=hashlib.sha256(raw).hexdigest(), source_size=list(path.size),
                      original_sha256=path.info.get("source_sha256"), original_file=path.info.get("source_file"))
            return
        source = Path(path).resolve()
        import os
        data_root = Path(os.environ.get("NUTRITION_DATA_DIR", str(self.root / "data"))).resolve()
        if not any(source.is_relative_to(base) for base in (self.root.resolve(), data_root)):
            raise ValueError("Refusing an image outside the project")
        original_digest = file_sha256(source)
        if original_digest not in self.images:
            with Image.open(source) as image:
                preview = image.convert("RGB")
                preview.thumbnail((700, 700))
                stream = BytesIO()
                preview.save(stream, format="JPEG", quality=82)
            self.images[original_digest] = dict(
                uri="data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii"),
                original_sha256=original_digest, original_file=source.name,
                thumbnail_sha256=hashlib.sha256(stream.getvalue()).hexdigest())
        self.emit("image", digest=original_digest)

    def capture(self, function, *args, choices=None):
        prior_events, prior_choices = self.events, self.choices
        self.events, self.choices = [], choices or {}
        try:
            function(*args)
            return self.events
        finally:
            self.events, self.choices = prior_events, prior_choices


def nutrient_variants(app, recorder, frame, dataset_index):
    """Exact px arguments from render_nutrients; shared finite arrays save space.

    Null/NaN observations never contribute to any of these distributions. Keep
    every finite observation, including zeros and extremes; no sampled violins.
    """
    numeric = app.numeric_nutrients(frame)
    available = numeric.columns[numeric.notna().any()].tolist()
    arrays, variants = {}, {}
    for name in available:
        ref = f"{dataset_index}:{name}"
        arrays[ref] = numeric[name].dropna().tolist()
        variants[name] = {}
        for kind in ("Histogram", "Box plot", "Violin plot"):
            if kind == "Histogram":
                fig = px.histogram(numeric, x=name, nbins=35, color_discrete_sequence=["#64B5F6"])
            elif kind == "Box plot":
                fig = px.box(numeric, y=name, color_discrete_sequence=["#62D6BD"])
            else:
                fig = px.violin(numeric, y=name, box=True, color_discrete_sequence=["#B699FF"])
            fig.update_layout(title="Spread of available nutrient values")
            events = recorder.capture(app.chart, fig)
            spec = events[0]["spec"]
            for trace in spec["data"]:
                trace["x" if kind == "Histogram" else "y"] = {"$values": ref}
            variants[name][kind] = spec
    return arrays, variants


def representative_keys(images, limit=3):
    """Deterministic bounded examples, separately selected within each split."""
    keys = []
    for split in ("test", "train"):
        keys.extend(images.loc[images["dataset_split"].eq(split), "image_key"].head(limit).tolist())
    return keys


def build_payload(root=ROOT):
    root = Path(root).resolve()
    app = load_dashboard(root)
    recorder = Recorder(root)
    app.st = recorder
    products, inventory, provenance = app.load_linked(app.LINKED)
    sample = app.load_sample(())
    frames = [sample, app.products_for_eda(products)]
    payload = dict(exported_at=datetime.now(timezone.utc).isoformat(), sections=app.SECTIONS,
                   datasets=app.DATASETS, sources=app.SOURCES, views={}, values={}, distributions={},
                   originals={}, bio={}, results={}, summaries={}, result_rows={},
                   inventory_splits=inventory.groupby("dataset_split").size().to_dict(),
                   source_sha256={"app_final.py": file_sha256(root / "app_final.py")},
                   snapshot_notes=[
                       "Frozen, unfiltered Dashboard2 snapshot. No product country/category filtering or uploads.",
                       "All eight sections retain the source dashboard's explanations, headings and tables.",
                       "Every available nutrient has histogram, box and violin plots with all finite observations, not sampled values.",
                       "Correlation initially shows all available fields (the live dashboard initially selects six).",
                       "Original annotation browsing and full-cohort extraction examples are representative saved images only; seeded results include every saved sample.",
                       "Thumbnails are resized copies of cached originals. Original and thumbnail SHA-256 hashes are included; source files are unchanged.",
                       "No live updates, model loading, training, OCR, image fetching, or external image links. Pending/failed images are not scored as zero.",
                       "Product downloads include every unfiltered row, compressed inside this file; downloads need a browser with DecompressionStream (current Edge/Chrome/Firefox).",
                       "Sources: Open Food Facts (ODbL/Database Contents License); nutrient-detection-layout and product images (CC-BY-SA; original contributor attribution applies). Model: openfoodfacts/nutrition-extractor (CC-BY-NC-SA-4.0). Research snapshot, not verified nutrition advice."])

    # Replace only the UI adapter at runtime, never the source file. Each section
    # uses the exact original render function; annotation selectors are embedded
    # independently so switching images doesn't duplicate large product tables.
    original_renderer = app.render_original

    def original_placeholder(images, prefix):
        recorder.emit("original_selector")

    app.render_original = original_placeholder
    all_original_keys = set(representative_keys(inventory))
    for source_index, source in enumerate(app.SOURCES):
        summary, rows, bundle = app.read_results(source)
        source_key = str(source_index)
        if summary is None:
            payload["results"][source_key] = [dict(kind="notice", level="info", text="No saved results for this source. Nothing was executed.")]
            payload["bio"][source_key] = dict(intro=payload["results"][source_key], examples={})
            continue
        payload["summaries"][source_key] = summary
        payload["result_rows"][source_key] = json.loads(rows.to_json(orient="records"))

        def result_view():
            app.render_status(summary)
            if source_index == 0 and summary["failed_images"]:
                with recorder.expander("Failure details · not scored as zero"):
                    recorder.json(app.read_json(app.FULL / "failures.json"))
            app.render_approaches()
            app.render_results(summary)
            with recorder.expander("Saved per-image results at export time"):
                app.table(rows)

        payload["results"][source_key] = recorder.capture(result_view)
        available = inventory.loc[inventory["image_key"].isin(rows["image_key"])]
        keys = (available["image_key"].tolist() if bundle is not None else representative_keys(available))
        all_original_keys.update(keys)
        intro = recorder.capture(app.render_status, summary)
        intro += recorder.capture(app.render_approaches)
        payload["bio"][source_key] = dict(intro=intro, examples={}, split_keys={})
        for key in keys:
            # Validate against original token/box/gold and coverage via the real
            # render_bio -> checked_prediction, no inferred/reference rewrites.
            events = recorder.capture(app.render_bio, inventory, summary, rows, bundle,
                                      choices={"bio_image": key})
            # Keep the gold/prediction images and concise panel captions, but omit
            # the long overlay explanation requested for removal from the HTML.
            events = [event for event in events if not (
                event.get("kind") == "caption"
                and "Both panels use the same supplied OCR word boxes" in event.get("html", ""))]
            payload["bio"][source_key]["examples"][key] = events
            split = str(available.loc[available["image_key"].eq(key), "dataset_split"].iloc[0])
            payload["bio"][source_key]["split_keys"].setdefault(split, []).append(key)
        payload["bio"][source_key]["scores"] = recorder.capture(app.render_results, summary)
        if not keys:
            payload["bio"][source_key]["intro"] += recorder.capture(app.render_bio, inventory, summary, rows, bundle)

    for key in sorted(all_original_keys):
        payload["originals"][key] = recorder.capture(original_renderer, inventory, "offline_original",
                                                     choices={"offline_original_image": key})

    for index, (dataset, frame) in enumerate(zip(app.DATASETS, frames)):
        dataset_key = str(index)
        views = payload["views"][dataset_key] = {}
        views[app.SECTIONS[0]] = recorder.capture(app.render_raw, frame, inventory)
        views[app.SECTIONS[1]] = recorder.capture(app.render_overview, frame, inventory)
        views[app.SECTIONS[2]] = recorder.capture(app.render_understanding, frame, inventory, dataset)
        nutrient_events = recorder.capture(app.render_nutrients, frame)
        # The first chart becomes the interactive nutrient/type selector. The
        # second remains the original full correlation Plotly specification.
        for event in nutrient_events:
            if event["kind"] == "plot":
                event.clear()
                event.update(kind="distribution_selector")
                break
        nutrient_events = [e for e in nutrient_events if not (e["kind"] == "static_control" and
                           e["label"] in ("Nutrient field (source unit and basis)", "Chart type"))]
        views[app.SECTIONS[3]] = nutrient_events
        arrays, variants = nutrient_variants(app, recorder, frame, index)
        payload["values"].update(arrays)
        payload["distributions"][dataset_key] = variants
        views[app.SECTIONS[4]] = recorder.capture(app.render_quality, frame)
        views[app.SECTIONS[5]] = recorder.capture(app.render_part2, inventory, inventory)
        views["row_count"] = len(frame)

    payload["downloads"] = recorder.downloads
    payload["images"] = recorder.images
    payload["provenance"] = provenance
    # Count retained specs, not temporary default distributions replaced by the
    # selector. Exclude numeric arrays/download strings from this traversal.
    def count_specs(value):
        if isinstance(value, dict):
            if "data" in value and "layout" in value:
                return 1
            return sum(count_specs(child) for child in value.values())
        if isinstance(value, list):
            return sum(count_specs(child) for child in value)
        return 0
    payload["chart_spec_count"] = sum(count_specs(payload[key]) for key in
                                     ("views", "distributions", "originals", "bio", "results"))
    payload["source_rendering"] = "Direct app_final.py render functions with a read-only Streamlit output adapter"
    forbidden = [name for name in sys.modules if name.startswith(("part2.layoutlm_bio", "part2.inference", "torch", "transformers", "onnxruntime", "easyocr"))]
    if forbidden:
        raise RuntimeError(f"Unexpected inference/model imports: {forbidden}")
    # Validate strict JSON before committing the HTML.
    script_json(payload)
    return payload


def export(root=ROOT):
    root = Path(root).resolve()
    payload = build_payload(root)
    template = (root / "src/nutrition_extraction/clean_dashboard_offline.html").read_text(encoding="utf-8")
    # Single pass: do not accidentally replace a marker inside an OCR string.
    before, after = template.split("__DASHBOARD2_JSON__")
    before = before.replace("/*__PLOTLY__*/", get_plotlyjs())
    after = after.replace("/*__BIO_VISUALIZATION__*/", visualization_controller(root))
    destination = root / "main_output/index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(before + script_json(payload) + after, encoding="utf-8")
    print(f"Created {destination} ({destination.stat().st_size:,} bytes / {destination.stat().st_size / 1024**2:.2f} MiB)")
    print(f"Sections: {len(payload['sections'])}; datasets: {len(payload['datasets'])}; chart specs: {payload['chart_spec_count']}")
    for key, summary in payload["summaries"].items():
        print(f"{payload['sources'][int(key)]}: {summary['completed_images']}/{summary['target_images']} saved; status={summary['status']}; {summary['updated_at_utc']}")
    print(f"Embedded thumbnails: {len(payload['images'])}; original annotation examples: {len(payload['originals'])}. No inference executed.")
    return destination


def visualization_controller(root):
    return (Path(root) / "src/nutrition_extraction/bio_comparison_offline.js").read_text(encoding="utf-8")


def refresh_bio(root=ROOT):
    """Refresh only saved BIO example events; keep the custom frozen shell/data."""
    root = Path(root).resolve()
    destination = root / "main_output/index.html"
    html = destination.read_text(encoding="utf-8")
    marker = '<script id="dashboard2-snapshot" type="application/json">'
    before, rest = html.split(marker, 1)
    raw, after = rest.split("</script>", 1)
    payload = json.loads(raw)
    app = load_dashboard(root)
    recorder = Recorder(root)
    app.st = recorder
    _, inventory, _ = app.load_linked(app.LINKED)
    for source, saved in payload["bio"].items():
        bundle = app.read_json(app.BUNDLE) if source == "1" else None
        if bundle is not None and bundle.get("inference_schema_version") != app.CACHE_SCHEMA:
            raise ValueError("Incompatible seeded cache schema")
        rows = pd.DataFrame(payload["result_rows"][source])
        for key in saved["examples"]:
            saved["examples"][key] = recorder.capture(
                app.render_bio, inventory, payload["summaries"][source], rows, bundle,
                choices={"bio_image": key})
    payload["downloads"].update(recorder.downloads)
    payload["images"].update(recorder.images)
    payload["source_sha256"] = {"app_final.py": file_sha256(root / "app_final.py")}
    if not any(e["kind"] == "bio_overlay" for s in payload["bio"].values()
               for events in s["examples"].values() for e in events):
        raise ValueError("No local official-dataset comparison available; artifact not written")
    start, end = "/* BIO_COMPARISON_START */", "/* BIO_COMPARISON_END */"
    controller = visualization_controller(root)
    if start in after:
        first, tail = after.split(start, 1)
        _, tail = tail.split(end, 1)
        after = first + controller + tail
    else:
        anchor = "D.sections.forEach(name=>"
        if after.count(anchor) != 1:
            raise ValueError("Cannot identify frozen controller insertion point")
        after = after.replace(anchor, controller + "\n" + anchor, 1)
    destination.write_text(before + marker + script_json(payload) + "</script>" + after, encoding="utf-8")
    print(f"Refreshed BIO examples only: {destination}; charts, metrics, custom shell retained.")
    return destination


if __name__ == "__main__":
    if "--refresh-bio" in sys.argv:
        refresh_bio()
    else:
        export()