"""Typed configuration for bounded Part 2 dataset and inference runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    max_side: int = 2000
    contrast: float = 1.15
    grayscale: bool = False
    resize: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str = "openfoodfacts/nutrient-detection-layout"
    config_name: str = "default"
    split: str = "train"
    subset_size: int = 30
    batch_size: int = 2
    num_workers: int = 0
    image_cache: Path = Path("data/part2_cache/images")

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'; preserve the official split.")
        if not 1 <= self.subset_size <= 50:
            raise ValueError("subset_size must be between 1 and 50; full-dataset reads are disabled.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")


@dataclass(frozen=True)
class OCRConfig:
    languages: tuple[str, ...] = ("en",)
    gpu: bool = False
    allow_model_downloads: bool = False
    model_storage_directory: Path = Path("models/easyocr")
    cache_directory: Path = Path("data/part2_cache/ocr")
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def __post_init__(self) -> None:
        if self.gpu:
            raise ValueError("This prototype is CPU-only; set gpu=False.")
        if not self.languages:
            raise ValueError("At least one EasyOCR language is required.")


DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_OCR_CONFIG = OCRConfig()