"""Training CLI for the annotated nutrition-document dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from nutrition_extraction.dataset import NutritionDocumentDataset
from nutrition_extraction.model import build_model, choose_device, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    labels = config["model"]["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_name"], apply_ocr=False)
    data_cfg = config["data"]
    image_root = Path(data_cfg.get("image_root", "."))
    train_set = NutritionDocumentDataset(Path(data_cfg["train_manifest"]), processor, label_to_id, image_root, data_cfg["max_length"])
    validation_set = NutritionDocumentDataset(Path(data_cfg["validation_manifest"]), processor, label_to_id, image_root, data_cfg["max_length"])
    training_cfg = config["training"]
    train_loader = DataLoader(train_set, batch_size=training_cfg["batch_size"], shuffle=True, num_workers=training_cfg["num_workers"])
    validation_loader = DataLoader(validation_set, batch_size=training_cfg["batch_size"], shuffle=False, num_workers=training_cfg["num_workers"])
    model = build_model(config["model"]["pretrained_name"], labels)
    device = choose_device(training_cfg.get("device", "auto"))
    result = train_model(
        model, train_loader, validation_loader,
        output_dir=Path(config["model"]["output_dir"]),
        epochs=int(training_cfg["epochs"]),
        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        device=device,
    )
    processor.save_pretrained(Path(config["model"]["output_dir"]))
    print(result)


if __name__ == "__main__":
    main()
