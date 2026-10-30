"""OCR and token-level inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


def ocr_image(image_path: Path, tesseract_cmd: str | None = None) -> tuple[Image.Image, list[str], list[list[int]]]:
    """Run Tesseract and map pixel boxes to LayoutLM's 0..1000 coordinate scale."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as error:
        raise RuntimeError("Install the optional OCR dependency with `pip install -e '.[ocr]'`.") from error
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    width, height = image.size
    ocr = pytesseract.image_to_data(image, output_type=Output.DICT)
    words: list[str] = []
    boxes: list[list[int]] = []
    for index, token in enumerate(ocr["text"]):
        token = str(token).strip()
        if not token:
            continue
        x, y = int(ocr["left"][index]), int(ocr["top"][index])
        w, h = int(ocr["width"][index]), int(ocr["height"][index])
        x0 = max(0, min(1000, round(1000 * x / width)))
        y0 = max(0, min(1000, round(1000 * y / height)))
        x1 = max(0, min(1000, round(1000 * (x + w) / width)))
        y1 = max(0, min(1000, round(1000 * (y + h) / height)))
        words.append(token)
        boxes.append([x0, y0, max(x0, x1), max(y0, y1)])
    if not words:
        raise ValueError(f"OCR found no text in {image_path}; check image quality or OCR language settings.")
    return image, words, boxes


def predict_tags(
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    *,
    processor: Any,
    model: Any,
    id_to_label: dict[int, str],
    device: torch.device,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Predict one BIO label per OCR word using the first subtoken representation."""
    encoded = processor(
        images=image, text=words, boxes=boxes, truncation=True,
        padding="max_length", max_length=max_length, return_tensors="pt",
    )
    word_ids = encoded.word_ids(batch_index=0)
    model_inputs = {key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    model.eval()
    with torch.no_grad():
        logits = model(**model_inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        token_ids = probabilities.argmax(dim=-1).tolist()
        confidences = probabilities.max(dim=-1).values.tolist()

    output: list[dict[str, Any]] = []
    seen_word_ids: set[int] = set()
    for token_index, word_index in enumerate(word_ids):
        if word_index is None or word_index in seen_word_ids or word_index >= len(words):
            continue
        seen_word_ids.add(word_index)
        output.append({
            "text": words[word_index],
            "box": boxes[word_index],
            "tag": id_to_label[int(token_ids[token_index])],
            "confidence": float(confidences[token_index]),
        })
    return output


def group_bio_spans(tagged_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent same-type BIO tokens; row/column relation resolution remains a later stage."""
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in tagged_words:
        prefix, _, entity_type = item["tag"].partition("-")
        if prefix == "O" or not entity_type:
            if current:
                spans.append(current)
                current = None
            continue
        if prefix == "B" or current is None or current["type"] != entity_type:
            if current:
                spans.append(current)
            current = {
                "type": entity_type,
                "tokens": [item["text"]],
                "confidence": [item["confidence"]],
                "boxes": [item["box"]],
            }
        else:
            current["tokens"].append(item["text"])
            current["confidence"].append(item["confidence"])
            current["boxes"].append(item["box"])
    if current:
        spans.append(current)
    return [
        {
            "type": span["type"],
            "text": " ".join(span["tokens"]),
            "confidence": sum(span["confidence"]) / len(span["confidence"]),
            "box": [
                min(box[0] for box in span["boxes"]),
                min(box[1] for box in span["boxes"]),
                max(box[2] for box in span["boxes"]),
                max(box[3] for box in span["boxes"]),
            ],
        }
        for span in spans
    ]


def assemble_candidate_rows(spans: list[dict[str, Any]], vertical_tolerance: int = 30) -> list[dict[str, Any]]:
    """Pair spans on nearby visual rows; ambiguous or multi-column rows need review."""
    ordered = sorted(spans, key=lambda span: ((span["box"][1] + span["box"][3]) / 2, span["box"][0]))
    visual_rows: list[dict[str, Any]] = []
    for span in ordered:
        center_y = (span["box"][1] + span["box"][3]) / 2
        matching_row = next(
            (row for row in reversed(visual_rows) if abs(center_y - row["center_y"]) <= vertical_tolerance),
            None,
        )
        if matching_row is None:
            matching_row = {"center_y": center_y, "spans": []}
            visual_rows.append(matching_row)
        matching_row["spans"].append(span)
        matching_row["center_y"] = sum((item["box"][1] + item["box"][3]) / 2 for item in matching_row["spans"]) / len(matching_row["spans"])

    candidates: list[dict[str, Any]] = []
    for row in visual_rows:
        row_spans = sorted(row["spans"], key=lambda span: span["box"][0])
        nutrients = [span for span in row_spans if span["type"] == "NUTRIENT"]
        if not nutrients:
            continue
        values = [span["text"] for span in row_spans if span["type"] == "VALUE"]
        units = [span["text"] for span in row_spans if span["type"] == "UNIT"]
        servings = [span["text"] for span in row_spans if span["type"] == "SERVING"]
        candidates.append({
            "nutrient": nutrients[0]["text"],
            "values": values,
            "units": units,
            "serving_context": servings,
            "confidence": min(span["confidence"] for span in row_spans),
            "needs_review": len(nutrients) != 1 or not values or len(values) > 1,
        })
    return candidates
