"""Reusable, non-destructive preprocessing for nutrition-panel images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from part2.config import PreprocessingConfig


def load_image(path: str | Path) -> Image.Image:
    """Load an image and return an EXIF-oriented RGB copy without modifying it."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            oriented = ImageOps.exif_transpose(source)
            return oriented.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to load image: {image_path}") from error


def preprocess_image(
    image: Image.Image,
    config: PreprocessingConfig | None = None,
) -> Image.Image:
    """Return a processed copy; source images are never overwritten."""
    options = config or PreprocessingConfig()
    if options.max_side < 1:
        raise ValueError("max_side must be positive.")
    if options.contrast <= 0:
        raise ValueError("contrast must be positive.")

    processed = ImageOps.exif_transpose(image).convert("RGB")
    if options.resize and max(processed.size) > options.max_side:
        scale = options.max_side / max(processed.size)
        size = (max(1, round(processed.width * scale)), max(1, round(processed.height * scale)))
        processed = processed.resize(size, Image.Resampling.LANCZOS)
    if options.contrast != 1.0:
        processed = ImageEnhance.Contrast(processed).enhance(options.contrast)
    if options.grayscale:
        processed = ImageOps.grayscale(processed).convert("RGB")
    return processed


def preprocess_path(path: str | Path, config: PreprocessingConfig | None = None) -> Image.Image:
    return preprocess_image(load_image(path), config)