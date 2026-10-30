"""Local-only overlays and final dashboard contract; no models or image fetching."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nutrition_extraction.bio_visualization import comparison_overlays, pixel_box, resolve_cached_image


def test_normalized_coordinates():
    assert pixel_box([0, 0, 1000, 1000], (201, 101)) == (0, 0, 200, 100)
    assert pixel_box([250, 250, 750, 750], (201, 101)) == (50, 25, 150, 75)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [4, 1, 2, 3], [-1, 0, 2, 3], [0, 0, 1001, 3], [0, 0, float('nan'), 3]])
def test_invalid_boxes_not_fabricated(box):
    with pytest.raises(ValueError):
        pixel_box(box, (100, 100))


def test_overlays_counts_no_mutation_orientation(tmp_path):
    path = tmp_path / "sideways.png"
    Image.new("RGB", (200, 80), "white").save(path)
    before = path.read_bytes()
    boxes = [[0, 0, 300, 300], [400, 400, 700, 700], [750, 750, 1000, 1000]]
    gold, predicted = ["B-FAT", "O", "O"], ["B-FAT", "B-SALT", "O"]
    original = deepcopy((boxes, gold, predicted))
    panels, counts = comparison_overlays(path, boxes, gold, predicted)
    assert counts == dict(tokens=3, matches=2, mismatches=1, gold_non_o=1, predicted_non_o=2, active_tokens=2)
    assert all(p.size == (200, 80) for p in panels)
    assert panels[0].getpixel((0, 0)) == (36, 199, 138)
    assert panels[1].getpixel(pixel_box(boxes[1], (200, 80))[:2]) == (255, 173, 66)
    assert panels[0].getpixel((160, 70)) == (255, 255, 255)  # O stays unboxed.
    assert path.read_bytes() == before and (boxes, gold, predicted) == original


def test_no_gold_no_comparison(tmp_path):
    with pytest.raises(ValueError, match="aligned"):
        comparison_overlays(tmp_path / "missing.png", [[0, 0, 10, 10]], [], ["O"])


def test_portable_resolution_hash_and_allowed_roots(tmp_path):
    data = tmp_path / "data"
    cache = data / "part2_cache/images"
    cache.mkdir(parents=True)
    image = cache / "saved.png"
    Image.new("RGB", (20, 10)).save(image)
    payload = dict(image_path=r"C:\old-project\data\part2_cache\images\saved.png",
                   image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    assert resolve_cached_image(payload, data, tmp_path / "project") == image
    payload["image_sha256"] = "0" * 64
    assert resolve_cached_image(payload, data, tmp_path / "project") is None
    outside = tmp_path / "outside.png"
    Image.new("RGB", (20, 10)).save(outside)
    assert resolve_cached_image(dict(image_path=str(outside)), data, tmp_path / "project") is None
    assert resolve_cached_image(dict(image_path=str(image)), data, tmp_path / "project") == image


def test_artifact_has_embedded_gold_prediction_pair_and_caveats():
    html = (ROOT / "main_output/index.html").read_text(encoding="utf-8")
    data = json.loads(html.split('<script id="dashboard2-snapshot" type="application/json">', 1)[1].split('</script>', 1)[0])
    events = next(iter(data["bio"]["1"]["examples"].values()))
    overlays = [e for e in events if e["kind"] == "bio_overlay"]
    assert len(overlays) == 2
    assert "gold" in overlays[0]["caption"] and "predictions" in overlays[1]["caption"]
    for overlay in overlays:
        raw = base64.b64decode(overlay["uri"].split(',', 1)[1])
        assert hashlib.sha256(raw).hexdigest() == overlay["sha256"]
        with Image.open(BytesIO(raw)) as image:
            assert image.format == "PNG"
    text = json.dumps(events)
    for phrase in ("not model-predicted locations", "not independently verified truth", "Uploads have no reference gold", "Exact token labels", "not a replacement for span"):
        assert phrase in text
    assert "BIO_COMPARISON_START" in html and "How was the overlap calculated?" in html
    assert "FINAL" in html and "$('timestamp').textContent" not in html


def test_live_cached_only_bio_ui():
    from streamlit.testing.v1 import AppTest
    if not os.environ.get("NUTRITION_DATA_DIR"):
        pytest.skip("Set NUTRITION_DATA_DIR to the existing local data cache")
    at = AppTest.from_file(str(ROOT / "app_final.py"), default_timeout=30).run()
    at.sidebar.radio[0].set_value("BIO Entity Tagging").run()
    assert not at.exception
    assert any("dataset gold vs cached predictions" in h.value for h in at.subheader)
    assert any("Exact token labels" in c.value for c in at.caption)
    assert len(at.get("image")) == 2
    assert [image.proto.imgs[0].caption for image in at.get("image")] == [
        "Dataset gold · original annotations", "Cached model predictions"]
    assert not any(n.startswith(("torch", "transformers", "onnxruntime", "easyocr", "part2.inference")) for n in sys.modules)