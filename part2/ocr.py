"""CPU EasyOCR adapter with a reusable reader and content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from part2.config import OCRConfig
from part2.preprocessing import preprocess_path
from part2.schemas import OCRDetection, OCRResult

_READERS: dict[tuple[tuple[str, ...], str, bool, bool], Any] = {}
_READER_LOCK = threading.Lock()


def _reader(config: OCRConfig) -> Any:
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError("EasyOCR is optional. Install with: pip install -e '.[easyocr]'") from error
    key = (tuple(config.languages), str(config.model_storage_directory.resolve()), config.allow_model_downloads, False)
    with _READER_LOCK:
        if key not in _READERS:
            config.model_storage_directory.mkdir(parents=True, exist_ok=True)
            _READERS[key] = easyocr.Reader(
                list(config.languages),
                gpu=False,
                quantize=False,
                download_enabled=config.allow_model_downloads,
                model_storage_directory=str(config.model_storage_directory),
                verbose=False,
            )
        return _READERS[key]


def _cache_key(image_path: Path, config: OCRConfig) -> str:
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    options = json.dumps(
        {
            "sha256": digest,
            "quantize": False,
            "languages": config.languages,
            "preprocessing": config.preprocessing.__dict__,
        },
        sort_keys=True,
    )
    return hashlib.sha256(options.encode("utf-8")).hexdigest()


def _reading_order(detections: list[OCRDetection]) -> list[OCRDetection]:
    if not detections:
        return detections
    heights = sorted(max(1, max(p[1] for p in d.bbox) - min(p[1] for p in d.bbox)) for d in detections)
    tolerance = max(8, heights[len(heights) // 2] // 2)
    ordered = sorted(detections, key=lambda d: (sum(p[1] for p in d.bbox) / 4, min(p[0] for p in d.bbox)))
    rows: list[list[OCRDetection]] = []
    row_y: list[float] = []
    for detection in ordered:
        center_y = sum(point[1] for point in detection.bbox) / 4
        row_index = next((i for i in range(len(rows) - 1, -1, -1) if abs(row_y[i] - center_y) <= tolerance), None)
        if row_index is None:
            rows.append([detection])
            row_y.append(center_y)
        else:
            rows[row_index].append(detection)
            row_y[row_index] = sum(sum(p[1] for p in item.bbox) / 4 for item in rows[row_index]) / len(rows[row_index])
    result: list[OCRDetection] = []
    for row in rows:
        result.extend(sorted(row, key=lambda d: min(point[0] for point in d.bbox)))
    return [OCRDetection(d.text, d.bbox, d.confidence, i) for i, d in enumerate(result)]


def _from_cache(path: Path, image_path: Path) -> OCRResult | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        detections = [OCRDetection(**item) for item in payload["detections"]]
        return OCRResult(str(image_path), detections, payload.get("error"), payload.get("elapsed_seconds"), True)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def run_easyocr(image_path: str | Path, config: OCRConfig | None = None) -> OCRResult:
    """Run EasyOCR with gpu=False; failures are returned in OCRResult.error."""
    options = config or OCRConfig()
    path = Path(image_path)
    cache_path: Path | None = None
    try:
        key = _cache_key(path, options)
        cache_path = options.cache_directory / f"{key}.json"
        cached = _from_cache(cache_path, path)
        if cached is not None:
            return cached
        image = preprocess_path(path, options.preprocessing)
        reader = _reader(options)
        start = time.perf_counter()
        raw_results = reader.readtext(np.asarray(image), detail=1, paragraph=False)
        elapsed = time.perf_counter() - start
        detections: list[OCRDetection] = []
        for polygon, text, confidence in raw_results:
            cleaned = str(text).strip()
            if not cleaned:
                continue
            points = [[int(round(float(point[0]))), int(round(float(point[1])))] for point in polygon]
            if len(points) < 4:
                continue
            detections.append(OCRDetection(cleaned, points, float(confidence), len(detections)))
        ordered = _reading_order(detections)
        result = OCRResult(str(path), ordered, elapsed_seconds=elapsed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as error:  # noqa: BLE001 - optional model/runtime errors are surfaced in JSON.
        return OCRResult(str(path), [], error=f"{type(error).__name__}: {error}")