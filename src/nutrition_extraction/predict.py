"""Run OCR and entity tagging for one product image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from nutrition_extraction.inference import assemble_candidate_rows, group_bio_spans, ocr_image, predict_tags
from nutrition_extraction.model import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or Path(config["model"]["output_dir"])
    labels = config["model"]["labels"]
    processor = AutoProcessor.from_pretrained(checkpoint, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint)
    device = choose_device(config["training"].get("device", "auto"))
    model.to(device)
    image, words, boxes = ocr_image(args.image, config.get("inference", {}).get("tesseract_cmd"))
    tagged = predict_tags(
        image, words, boxes, processor=processor, model=model,
        id_to_label={index: label for index, label in enumerate(labels)},
        device=device, max_length=int(config["data"]["max_length"]),
    )
    spans = group_bio_spans(tagged)
    rows = assemble_candidate_rows(spans)
    threshold = float(config.get("inference", {}).get("confidence_threshold", 0.7))
    result = {
        "image": str(args.image),
        "entities": spans,
        "candidate_rows": rows,
        "needs_review": any(entity["confidence"] < threshold for entity in spans)
        or any(row["needs_review"] or row["confidence"] < threshold for row in rows),
        "notice": "Row association is a spatial heuristic; verify multi-column tables and serving context.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prediction written to {args.output}")


if __name__ == "__main__":
    main()
