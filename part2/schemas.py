"""JSON-compatible schemas shared by OCR, extraction, and evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OCRDetection:
    text: str
    bbox: list[list[int]]
    confidence: float
    reading_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OCRResult:
    image_path: str
    detections: list[OCRDetection]
    error: str | None = None
    elapsed_seconds: float | None = None
    from_cache: bool = False

    @property
    def text(self) -> str:
        return " ".join(detection.text for detection in self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "text": self.text,
            "detections": [detection.to_dict() for detection in self.detections],
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "from_cache": self.from_cache,
        }


@dataclass(frozen=True)
class NutrientValue:
    value: float
    unit: str | None
    basis: str
    raw_text: str
    ocr_confidence: float | None = None
    normalized_value: float | None = None
    normalized_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    barcode: str | None
    image_path: str
    nutrition_basis: str | None = None
    nutrients: dict[str, dict[str, list[NutrientValue]]] = field(default_factory=dict)
    unresolved_fields: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ocr_error: str | None = None
    processing_time_seconds: float | None = None
    method: str = "easyocr+spatial-rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "barcode": self.barcode,
            "image_path": self.image_path,
            "nutrition_basis": self.nutrition_basis,
            "nutrients": {
                basis: {
                    nutrient: [item.to_dict() for item in values]
                    for nutrient, values in nutrient_values.items()
                }
                for basis, nutrient_values in self.nutrients.items()
            },
            "unresolved_fields": self.unresolved_fields,
            "warnings": self.warnings,
            "ocr_error": self.ocr_error,
            "processing_time_seconds": self.processing_time_seconds,
            "method": self.method,
        }


def validate_extraction_payload(payload: dict[str, Any]) -> None:
    """Raise ValueError for structural or non-finite numeric output errors."""
    required = {"barcode", "image_path", "nutrition_basis", "nutrients", "unresolved_fields", "warnings"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Extraction JSON is missing required keys: {sorted(missing)}")
    if not isinstance(payload["nutrients"], dict):
        raise ValueError("nutrients must be an object keyed by nutrition basis.")
    for basis, nutrients in payload["nutrients"].items():
        if basis not in {"per_100g", "per_100ml", "per_serving", "unspecified"}:
            raise ValueError(f"Unsupported nutrition basis: {basis}")
        if not isinstance(nutrients, dict):
            raise ValueError(f"Nutrients for {basis} must be an object.")
        for nutrient, values in nutrients.items():
            if not isinstance(values, list):
                raise ValueError(f"{basis}.{nutrient} must be a list to preserve multiple values.")
            for value in values:
                if not isinstance(value, dict) or "value" not in value or "unit" not in value:
                    raise ValueError(f"Malformed value for {basis}.{nutrient}.")
                number = value["value"]
                if not isinstance(number, (int, float)):
                    raise ValueError(f"Non-numeric value for {basis}.{nutrient}.")
                if not math.isfinite(float(number)):
                    raise ValueError(f"Non-finite value for {basis}.{nutrient}.")