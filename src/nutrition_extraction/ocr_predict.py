"""CPU OCR-baseline prediction CLI with configurable OCR engines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from nutrition_extraction.extraction import extract_record
from nutrition_extraction.ocr import EasyOCRBackend, PaddleOCRBackend, TesseractBackend, load_preprocessed_image

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--product-id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ocr-backend", choices=("easyocr", "tesseract", "paddleocr"), default="easyocr")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ocr_cfg = config.get("ocr", {})
    if args.ocr_backend == "easyocr":
        languages = ocr_cfg.get("easyocr_languages", ["en"])
        if isinstance(languages, str):
            languages = [languages]
        backend = EasyOCRBackend(
            languages=languages,
            gpu=False,
            download_enabled=bool(ocr_cfg.get("allow_model_downloads", False)),
            model_storage_directory=str(ocr_cfg.get("model_storage_directory", "models/easyocr")),
        )
    elif args.ocr_backend == "paddleocr":
        backend = PaddleOCRBackend(language=ocr_cfg.get("paddle_language", "en"))
    else:
        backend = TesseractBackend(
            executable=ocr_cfg.get("tesseract_cmd"),
            language=ocr_cfg.get("tesseract_language", "eng"),
        )
    image = load_preprocessed_image(args.image, int(ocr_cfg.get("max_image_side", 2_000)))
    tokens = backend.extract(image)
    if not tokens:
        raise RuntimeError(f"OCR returned no readable text for {args.image}")
    record = extract_record(
        product_id=args.product_id or args.image.stem,
        tokens=tokens,
        source_image=str(args.image),
        confidence_threshold=float(ocr_cfg.get("confidence_threshold", 0.70)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %d extracted fields to %s", len(record.fields), args.output)


if __name__ == "__main__":
    main()
