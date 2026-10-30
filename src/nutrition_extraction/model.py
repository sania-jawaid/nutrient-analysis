"""Model construction and a compact supervised training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification


def build_model(pretrained_name: str, labels: list[str]) -> LayoutLMv3ForTokenClassification:
    """Load a pretrained layout-aware encoder with the task's BIO label head."""
    return LayoutLMv3ForTokenClassification.from_pretrained(
        pretrained_name,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )


def choose_device(configured_device: str = "auto") -> torch.device:
    """Resolve the configured accelerator, falling back safely to CPU."""
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
    return device


def train_model(
    model: LayoutLMv3ForTokenClassification,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> dict[str, Any]:
    """Train with validation-loss checkpoint selection; keep the test split untouched."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(output.loss.detach().cpu())
            train_batches += 1

        model.eval()
        validation_loss_sum = 0.0
        validation_batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                validation_loss_sum += float(output.loss.detach().cpu())
                validation_batches += 1
        if train_batches == 0 or validation_batches == 0:
            raise ValueError("Training and validation loaders must both contain at least one batch.")
        train_loss = train_loss_sum / train_batches
        validation_loss = validation_loss_sum / validation_batches
        history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_pretrained(output_dir)

    return {"best_validation_loss": best_validation_loss, "history": history, "checkpoint": str(output_dir)}
