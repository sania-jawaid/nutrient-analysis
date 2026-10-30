"""Bounded Hugging Face dataset-server loader and PyTorch collation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from part2.config import DatasetConfig


# Verified from the dataset repository's live ClassLabel metadata (revision
# 7d3245ca596f0c8f41dc740ca0fa5987b1e0fd4d), class ids 0–70.
BIO_LABELS: tuple[str, ...] = (
    "O",
    "B-ENERGY_KJ_100G", "I-ENERGY_KJ_100G",
    "B-VITAMIN_D_SERVING", "I-VITAMIN_D_SERVING",
    "B-SODIUM_SERVING", "I-SODIUM_SERVING",
    "B-PROTEINS_SERVING", "I-PROTEINS_SERVING",
    "B-ADDED_SUGARS_SERVING", "I-ADDED_SUGARS_SERVING",
    "B-CALCIUM_SERVING", "I-CALCIUM_SERVING",
    "B-FAT_SERVING", "I-FAT_SERVING",
    "B-ENERGY_KJ_SERVING", "I-ENERGY_KJ_SERVING",
    "B-SUGARS_100G", "I-SUGARS_100G",
    "B-SATURATED_FAT_SERVING", "I-SATURATED_FAT_SERVING",
    "B-SERVING_SIZE", "I-SERVING_SIZE",
    "B-SALT_SERVING", "I-SALT_SERVING",
    "B-ENERGY_KCAL_SERVING", "I-ENERGY_KCAL_SERVING",
    "B-FAT_100G", "I-FAT_100G",
    "B-SUGARS_SERVING", "I-SUGARS_SERVING",
    "B-FIBER_SERVING", "I-FIBER_SERVING",
    "B-TRANS_FAT_SERVING", "I-TRANS_FAT_SERVING",
    "B-POTASSIUM_SERVING", "I-POTASSIUM_SERVING",
    "B-CARBOHYDRATES_100G", "I-CARBOHYDRATES_100G",
    "B-POTASSIUM_100G", "I-POTASSIUM_100G",
    "B-IRON_SERVING", "I-IRON_SERVING",
    "B-CHOLESTEROL_100G", "I-CHOLESTEROL_100G",
    "B-TRANS_FAT_100G", "I-TRANS_FAT_100G",
    "B-ADDED_SUGARS_100G", "I-ADDED_SUGARS_100G",
    "B-FIBER_100G", "I-FIBER_100G",
    "B-CALCIUM_100G", "I-CALCIUM_100G",
    "B-SODIUM_100G", "I-SODIUM_100G",
    "B-ENERGY_KCAL_100G", "I-ENERGY_KCAL_100G",
    "B-CHOLESTEROL_SERVING", "I-CHOLESTEROL_SERVING",
    "B-CARBOHYDRATES_SERVING", "I-CARBOHYDRATES_SERVING",
    "B-SALT_100G", "I-SALT_100G",
    "B-VITAMIN_D_100G", "I-VITAMIN_D_100G",
    "B-SATURATED_FAT_100G", "I-SATURATED_FAT_100G",
    "B-PROTEINS_100G", "I-PROTEINS_100G",
    "B-IRON_100G", "I-IRON_100G",
)


class HuggingFaceNutritionDataset(Dataset[dict[str, Any]]):
    """Fetch only a small prefix of one official split; images are lazy and cached.

    The Hugging Face datasets-server `rows` endpoint returns per-row metadata,
    OCR token arrays, boxes, tags and a signed image URL. It avoids downloading
    the 635 MB repository parquet shards or the complete image dataset.
    """

    def __init__(
        self,
        config: DatasetConfig | None = None,
        *,
        timeout: int = 45,
        row_indices: list[int] | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self.timeout = timeout
        self.config.image_cache.mkdir(parents=True, exist_ok=True)
        if row_indices is not None:
            self._load_selected_rows(row_indices)
            return
        query = urlencode(
            {
                "dataset": self.config.dataset_id,
                "config": self.config.config_name,
                "split": self.config.split,
                "offset": 0,
                "length": self.config.subset_size,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        payload = self._request_json(url)
        self.total_split_rows = int(payload.get("num_rows_total", 0))
        self.records = [item.get("row", {}) for item in payload.get("rows", [])]
        if not self.records:
            raise RuntimeError(f"No rows returned for official split {self.config.split!r}.")
        for index, record in enumerate(self.records):
            self._validate_record(record, index)

    def _load_selected_rows(self, row_indices: list[int], page_size: int = 100) -> None:
        """Fetch only metadata pages needed for official split row indices."""
        if not row_indices:
            raise ValueError("row_indices must contain at least one official split index.")
        if len(row_indices) > 50:
            raise ValueError("At most 50 selected records may be loaded per bounded request.")
        if len(set(row_indices)) != len(row_indices) or any(index < 0 for index in row_indices):
            raise ValueError("Selected row indices must be unique non-negative integers.")

        rows_by_index: dict[int, dict[str, Any]] = {}
        self.total_split_rows = 0
        for offset in sorted({(index // page_size) * page_size for index in row_indices}):
            length = min(page_size, max(row_indices) + 1 - offset)
            query = urlencode({
                "dataset": self.config.dataset_id,
                "config": self.config.config_name,
                "split": self.config.split,
                "offset": offset,
                "length": length,
            })
            payload = self._request_json(f"https://datasets-server.huggingface.co/rows?{query}")
            self.total_split_rows = int(payload.get("num_rows_total", 0))
            for local_index, item in enumerate(payload.get("rows", [])):
                rows_by_index[offset + local_index] = item.get("row", {})

        missing = [index for index in row_indices if index not in rows_by_index or index >= self.total_split_rows]
        if missing:
            raise IndexError(f"Selected indices are outside the official {self.config.split} split: {missing[:10]}")
        self.records = [rows_by_index[index] for index in row_indices]
        self.source_indices = list(row_indices)
        for local_index, record in enumerate(self.records):
            self._validate_record(record, local_index)

    def __len__(self) -> int:
        return len(self.records)

    def _request_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": "nutrition-extraction-part2/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unable to read bounded Hugging Face dataset metadata: {error}") from error

    @staticmethod
    def _validate_record(record: dict[str, Any], index: int) -> None:
        tokens = record.get("tokens")
        boxes = record.get("bboxes")
        tags = record.get("ner_tags")
        if not isinstance(tokens, list) or not isinstance(boxes, list) or not isinstance(tags, list):
            raise ValueError(f"Dataset row {index} lacks tokens, bboxes, or ner_tags sequences.")
        if not (len(tokens) == len(boxes) == len(tags)):
            raise ValueError(
                f"Dataset row {index} has unaligned token/box/tag lengths: "
                f"{len(tokens)}/{len(boxes)}/{len(tags)}."
            )
        if any(not isinstance(box, list) or len(box) != 4 for box in boxes):
            raise ValueError(f"Dataset row {index} has a bounding box outside [x0,y0,x1,y1] form.")
        if any(not isinstance(tag, int) or tag < 0 or tag >= len(BIO_LABELS) for tag in tags):
            raise ValueError(f"Dataset row {index} has a label id outside 0..{len(BIO_LABELS) - 1}.")

    def _image_path(self, record: dict[str, Any]) -> Path:
        metadata = record.get("meta") or {}
        image_id = str(metadata.get("image_id", "image"))
        barcode = str(metadata.get("barcode", "unknown"))
        image_info = record.get("image") or {}
        url = image_info.get("src") if isinstance(image_info, dict) else None
        if not url:
            # Fallback to the source reference if the dataset-server image asset
            # URL is not included in its row response.
            url = metadata.get("image_url")
        if not url:
            raise RuntimeError(f"No image URL available for barcode {barcode}.")
        stable_image_ref = str(metadata.get("image_url") or url)
        cache_key = hashlib.sha256(
            f"{self.config.dataset_id}|{barcode}|{image_id}|{stable_image_ref}".encode("utf-8")
        ).hexdigest()
        destination = self.config.image_cache / f"{cache_key}.jpg"
        if destination.exists() and destination.stat().st_size:
            return destination
        request = Request(str(url), headers={"User-Agent": "nutrition-extraction-part2/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                image_bytes = response.read()
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(image_bytes)
            temporary.replace(destination)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Unable to fetch one dataset image for barcode {barcode}: {error}") from error
        return destination

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        meta = record.get("meta") or {}
        path = self._image_path(record)
        try:
            with Image.open(path) as source:
                image = source.copy().convert("RGB")
        except OSError as error:
            raise RuntimeError(f"Cached dataset image is unreadable: {path}") from error
        return {
            "image": image,
            "image_path": str(path),
            "barcode": str(meta.get("barcode") or ""),
            "tokens": [str(token) for token in record["tokens"]],
            "bboxes": [[int(value) for value in box] for box in record["bboxes"]],
            "ner_tags": [int(tag) for tag in record["ner_tags"]],
            "bio_labels": [BIO_LABELS[int(tag)] for tag in record["ner_tags"]],
            "meta": dict(meta),
            "split": self.config.split,
            "source_index": getattr(self, "source_indices", list(range(len(self.records))))[index],
        }


def collate_documents(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep images/tokens as lists and pad boxes/tags for variable sequence lengths."""
    if not batch:
        return {"images": [], "tokens": [], "barcodes": [], "metadata": []}
    lengths = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    padded_boxes = torch.zeros((len(batch), max_length, 4), dtype=torch.long)
    padded_tags = torch.full((len(batch), max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for row_index, item in enumerate(batch):
        length = len(item["tokens"])
        if length:
            padded_boxes[row_index, :length] = torch.tensor(item["bboxes"], dtype=torch.long)
            padded_tags[row_index, :length] = torch.tensor(item["ner_tags"], dtype=torch.long)
            attention_mask[row_index, :length] = True
    return {
        "images": [item["image"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "barcodes": [item["barcode"] for item in batch],
        "tokens": [item["tokens"] for item in batch],
        "bio_labels": [item["bio_labels"] for item in batch],
        "boxes": padded_boxes,
        "ner_tags": padded_tags,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "metadata": [item["meta"] for item in batch],
        "split": [item["split"] for item in batch],
    }


def make_dataloader(config: DatasetConfig | None = None) -> tuple[HuggingFaceNutritionDataset, DataLoader[dict[str, Any]]]:
    options = config or DatasetConfig()
    dataset = HuggingFaceNutritionDataset(options)
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=options.split == "train",
        num_workers=options.num_workers,
        collate_fn=collate_documents,
        pin_memory=False,
    )
    return dataset, loader