"""Replaceable OCR backend interface with lazy-loaded open-source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from nutrition_extraction.preprocessing import normalize_ocr_text, preprocess_image


@dataclass(frozen=True)
class OCRToken:
    """OCR word with pixel-space bounding box and calibrated-or-raw confidence."""

    text: str
    box: tuple[int, int, int, int]
    confidence: float


class OCRBackend(Protocol):
    """Minimal interface consumed by the extraction pipeline."""

    def extract(self, image: Image.Image) -> list[OCRToken]:
        """Return OCR tokens in pixel coordinates for the supplied RGB image."""


class TesseractBackend:
    """CPU OCR adapter; requires the Tesseract executable installed separately."""

    def __init__(self, executable: str | None = None, language: str = "eng") -> None:
        self.executable = executable
        self.language = language

    def extract(self, image: Image.Image) -> list[OCRToken]:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as error:
            raise RuntimeError("Install the OCR optional dependency: pip install -e '.[ocr]'") from error
        if self.executable:
            pytesseract.pytesseract.tesseract_cmd = self.executable
        data = pytesseract.image_to_data(image, lang=self.language, output_type=Output.DICT)
        tokens: list[OCRToken] = []
        for index, raw_text in enumerate(data["text"]):
            text = normalize_ocr_text(str(raw_text))
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0
            if not text or confidence < 0:
                continue
            x, y = int(data["left"][index]), int(data["top"][index])
            width, height = int(data["width"][index]), int(data["height"][index])
            tokens.append(OCRToken(text, (x, y, x + width, y + height), confidence / 100.0))
        return tokens


class EasyOCRBackend:
    """EasyOCR adapter running on CPU; reader weights download only when explicitly enabled.

    Args:
        languages: EasyOCR language codes, e.g. ["fr"] or ["fr", "en"].
        download_enabled: Allow EasyOCR to download detector/recognizer weights if absent.
            Keep false until the user approves the model download.
    """

    def __init__(
        self,
        languages: list[str] | tuple[str, ...] = ("en",),
        *,
        gpu: bool = False,
        download_enabled: bool = False,
        model_storage_directory: str | None = None,
    ) -> None:
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("Install the optional EasyOCR dependency: pip install -e '.[easyocr]'") from error
        if not languages:
            raise ValueError("EasyOCR requires at least one language code.")
        self._reader = easyocr.Reader(
            list(languages),
            gpu=gpu,
            download_enabled=download_enabled,
            model_storage_directory=model_storage_directory,
            verbose=False,
        )

    def extract(self, image: Image.Image) -> list[OCRToken]:
        """Return EasyOCR line detections as OCR tokens in original pixel coordinates."""
        import numpy as np

        results = self._reader.readtext(np.asarray(image), detail=1, paragraph=False)
        tokens: list[OCRToken] = []
        for polygon, raw_text, confidence in results:
            text = normalize_ocr_text(str(raw_text))
            if not text:
                continue
            xs = [int(point[0]) for point in polygon]
            ys = [int(point[1]) for point in polygon]
            tokens.append(OCRToken(text, (min(xs), min(ys), max(xs), max(ys)), float(confidence)))
        return tokens


class PaddleOCRBackend:
    """Optional PaddleOCR adapter; imports only when instantiated/used."""

    def __init__(self, language: str = "en") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR is optional and is not installed. Follow PaddleOCR's current Windows CPU installation guide."
            ) from error
        self._engine = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)

    def extract(self, image: Image.Image) -> list[OCRToken]:
        import numpy as np

        result = self._engine.ocr(np.asarray(image), cls=True)
        tokens: list[OCRToken] = []
        for line in (result[0] or []) if result else []:
            polygon, (raw_text, confidence) = line
            xs = [int(point[0]) for point in polygon]
            ys = [int(point[1]) for point in polygon]
            text = normalize_ocr_text(str(raw_text))
            if text:
                tokens.append(OCRToken(text, (min(xs), min(ys), max(xs), max(ys)), float(confidence)))
        return tokens


def load_preprocessed_image(image_path: Path, max_side: int = 2_000) -> Image.Image:
    """Single image-loading entry point shared by OCR backends."""
    return preprocess_image(image_path, max_side=max_side)
