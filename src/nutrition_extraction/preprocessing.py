"""Shared, deterministic preprocessing for OCR and document-model inputs."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageOps


def preprocess_image(image_path: Path, max_side: int = 2_000) -> Image.Image:
    """Apply EXIF orientation, RGB conversion, and bounded resize without changing aspect ratio."""
    if max_side < 64:
        raise ValueError("max_side must be at least 64 pixels")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return image.copy()


def normalize_ocr_text(text: str) -> str:
    """Normalize Unicode and whitespace while retaining useful decimal punctuation."""
    normalized = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    normalized = re.sub(r"[\t\r\n]+", " ", normalized)
    return re.sub(r"\s{2,}", " ", normalized).strip()


def normalize_nutrient_name(name: str) -> str:
    """Map common label variants to stable field names; unknown terms stay explicit."""
    key = normalize_ocr_text(name).casefold().strip(" :;,.\t")
    aliases = {
        "energy": "energy", "calories": "energy", "kcal": "energy", "kj": "energy",
        "fat": "fat", "total fat": "fat", "saturates": "saturated_fat",
        "saturated fat": "saturated_fat", "of which saturates": "saturated_fat",
        "carbohydrate": "carbohydrates", "carbohydrates": "carbohydrates",
        "of which sugars": "sugars", "sugar": "sugars", "sugars": "sugars",
        "fibre": "fiber", "fiber": "fiber", "protein": "proteins", "proteins": "proteins",
        "salt": "salt", "sodium": "sodium",
    }
    return aliases.get(key, re.sub(r"[^a-z0-9]+", "_", key).strip("_"))
