"""Evaluate a checkpoint on a labeled manifest and report entity metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from nutrition_extraction.dataset import NutritionDocumentDataset
from nutrition_extraction.metrics import entity_metrics
from nutrition_extraction.model import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Defaults to configured test manifest")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    labels = config["model"]["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    processor = AutoProcessor.from_pretrained(args.checkpoint, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.checkpoint)
    device = choose_device(config["training"].get("device", "auto"))
    model.to(device).eval()
    manifest = args.manifest or Path(config["data"]["test_manifest"])
    dataset = NutritionDocumentDataset(
        manifest, processor, label_to_id, Path(config["data"].get("image_root", ".")), config["data"]["max_length"]
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    truth: list[list[str]] = []
    predictions: list[list[str]] = []
    with torch.no_grad():
        for batch in loader:
            labels_tensor = batch.pop("labels")
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            predicted_ids = model(**model_inputs).logits.argmax(dim=-1).cpu()
            for gold, predicted in zip(labels_tensor, predicted_ids):
                gold_sequence: list[str] = []
                predicted_sequence: list[str] = []
                for gold_id, predicted_id in zip(gold.tolist(), predicted.tolist()):
                    if gold_id == -100:
                        continue
                    gold_sequence.append(id_to_label[gold_id])
                    predicted_sequence.append(id_to_label[predicted_id])
                truth.append(gold_sequence)
                predictions.append(predicted_sequence)
    print(entity_metrics(truth, predictions))


if __name__ == "__main__":
    main()
