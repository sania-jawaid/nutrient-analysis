"""Read-only visualization of aligned, supplied OCR boxes; never runs inference."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageDraw

MATCH = "#24C78A"
MISMATCH = "#FFAD42"
CAVEAT = ("Both panels use the same supplied OCR word boxes, not model-predicted locations. "
          "Only non-O labels are boxed in each panel; green = exact BIO-label match, amber = mismatch. "
          "Small numbers are zero-based token indices (see Advanced). Dataset gold is the original "
          "annotation, not independently verified truth. Original orientation is preserved. "
          "Uploads have no reference gold labels; this comparison is for saved official-dataset samples only.")


def resolve_cached_image(payload, data_dir, root):
    """Allow only explicit local roots; portable cache basename first, hash if saved."""
    data_dir, root = Path(data_dir).resolve(), Path(root).resolve()
    saved = str(payload.get("image_path", ""))
    name = PureWindowsPath(saved).name
    candidates = [data_dir / "part2_cache/images" / name, Path(saved)] if name else []
    expected = payload.get("image_sha256")
    for candidate in candidates:
        candidate = candidate.resolve()
        if not any(candidate.is_relative_to(base) for base in (data_dir, root)) or not candidate.is_file():
            continue
        if expected and hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            continue
        if payload.get("image_size"):
            try:
                with Image.open(candidate) as image:
                    if list(image.size) != list(payload["image_size"]):
                        continue
            except OSError:
                continue
        return candidate
    return None


def pixel_box(box, size):
    """Map the saved 0–1000 frame to source pixels, including image boundaries."""
    if len(box) != 4 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1000 for v in box):
        raise ValueError("Expected four finite normalized OCR coordinates in 0–1000.")
    x0, y0, x1, y1 = map(float, box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("OCR box must have positive area.")
    w, h = size
    return tuple(round(v * (extent - 1) / 1000) for v, extent in zip((x0, y0, x1, y1), (w, h, w, h)))


def comparison_overlays(path, boxes, gold, predicted):
    """Return detached RGB copies and exact-label counts, without changing inputs."""
    if not (len(boxes) == len(gold) == len(predicted)) or not boxes:
        raise ValueError("Gold, predictions and OCR boxes must be nonempty and aligned.")
    if any(not isinstance(label, str) or not label for label in [*gold, *predicted]):
        raise ValueError("Missing BIO labels; no gold may be fabricated.")
    with Image.open(path) as source:
        original = source.convert("RGB")  # Deliberately no EXIF transpose/rotation.
    coordinates = [pixel_box(box, original.size) for box in boxes]
    panels = []
    width = max(2, round(max(original.size) / 650))
    for labels in (gold, predicted):
        panel = original.copy()
        panel.info["source_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        panel.info["source_file"] = Path(path).name
        draw = ImageDraw.Draw(panel)
        for i, (box, label) in enumerate(zip(coordinates, labels)):
            if label == "O":
                continue
            color = MATCH if gold[i] == predicted[i] else MISMATCH
            draw.rectangle(box, outline=color, width=width)
            draw.text((box[0], max(0, box[1] - 12)), str(i), fill=color, stroke_width=1, stroke_fill="#000000")
        panels.append(panel)
    matches = sum(g == p for g, p in zip(gold, predicted))
    active = sum(g != "O" or p != "O" for g, p in zip(gold, predicted))
    return panels, dict(tokens=len(gold), matches=matches, mismatches=len(gold) - matches,
                        gold_non_o=sum(g != "O" for g in gold),
                        predicted_non_o=sum(p != "O" for p in predicted), active_tokens=active)