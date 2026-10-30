"""Image + OCR-word dataset for LayoutLMv3 token classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NutritionDocumentDataset(Dataset[dict[str, torch.Tensor]]):
    """Load product images and aligned OCR tokens, boxes, and BIO labels."""

    def __init__(
        self,
        manifest_path: Path,
        processor: Any,
        label_to_id: dict[str, int],
        image_root: Path = Path("."),
        max_length: int = 512,
    ) -> None:
        self.processor = processor
        self.label_to_id = label_to_id
        self.image_root = image_root
        self.max_length = max_length
        with manifest_path.open(encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        if not self.records:
            raise ValueError(f"No annotation records found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        words = record["words"]
        boxes = record["boxes"]
        tags = record["tags"]
        if not (len(words) == len(boxes) == len(tags)):
            raise ValueError(f"Words, boxes, and tags must align for record {record.get('product_id', index)}")
        unknown_labels = sorted(set(tags) - self.label_to_id.keys())
        if unknown_labels:
            raise ValueError(f"Unknown labels in record {record.get('product_id', index)}: {unknown_labels}")
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        encoded = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            word_labels=[self.label_to_id[tag] for tag in tags],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items() if isinstance(value, torch.Tensor)}


class ProductImageMetadataDataset(Dataset[dict[str, Any]]):
    """Return a resized image tensor, text metadata, and structured product targets.

    This dataset is useful for image/text ingestion demonstrations and image-level
    baselines. Its product-level metadata targets are weak labels only; use
    NutritionDocumentDataset with manually verified OCR-token annotations for
    supervised image-text extraction.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        image_root: Path = Path("."),
        image_key: str = "image_path",
        text_keys: tuple[str, ...] = ("product_name", "ingredients_text", "categories", "brands"),
        target_keys: tuple[str, ...] = (),
        image_size: int = 224,
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        self.records = list(records)
        self.image_root = image_root
        self.image_key = image_key
        self.text_keys = text_keys
        self.target_keys = target_keys
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = Path(str(record[self.image_key]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                image_array = np.asarray(image, dtype="float32") / 255.0
        except (OSError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot load image for record {record.get('code', index)}: {image_path}") from error
        text = " ".join(
            str(record[key]).strip()
            for key in self.text_keys
            if key in record and record[key] is not None and str(record[key]).strip()
        )
        targets = {key: record.get(key) for key in self.target_keys if key in record}
        return {
            "image": torch.from_numpy(image_array).permute(2, 0, 1),
            "product_id": str(record.get("code", record.get("product_id", index))),
            "text_metadata": text,
            "nutrition_targets_json": json.dumps(targets, ensure_ascii=False, default=str),
        }
