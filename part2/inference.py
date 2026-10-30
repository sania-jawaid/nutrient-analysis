"""Single-image EasyOCR inference and bounded dataset demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from part2.config import DatasetConfig, OCRConfig, PreprocessingConfig
from part2.dataset import make_dataloader
from part2.entity_extraction import extract_nutrients
from part2.ocr import run_easyocr
from part2.preprocessing import preprocess_path
from part2.schemas import ExtractionResult, validate_extraction_payload


def run_image_inference(
    image_path: str | Path,
    *,
    barcode: str | None = None,
    ocr_config: OCRConfig | None = None,
) -> ExtractionResult:
    path = Path(image_path)
    start = time.perf_counter()
    options = ocr_config or OCRConfig()
    ocr = run_easyocr(path, options)
    try:
        width, _ = preprocess_path(path, options.preprocessing).size
    except OSError:
        width = 0
    result = extract_nutrients(
        ocr.detections,
        barcode=barcode,
        image_path=str(path),
        image_width=width,
    )
    result.ocr_error = ocr.error
    result.processing_time_seconds = time.perf_counter() - start
    if ocr.error:
        result.warnings.append("OCR failed; no nutrient values were inferred.")
    payload = result.to_dict()
    validate_extraction_payload(payload)
    return result


def _write_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _run_loader_demo(args: argparse.Namespace) -> int:
    config = DatasetConfig(
        split=args.split or "train",
        subset_size=args.limit,
        batch_size=args.batch_size,
        num_workers=0,
    )
    dataset, loader = make_dataloader(config)
    batch = next(iter(loader))
    print(
        json.dumps(
            {
                "demonstration": "bounded PyTorch Dataset/DataLoader sample; no OCR or label-driven tuning",
                "split": config.split,
                "fetched_subset": len(dataset),
                "batch_images": len(batch["images"]),
                "barcodes": batch["barcodes"],
                "token_lengths": batch["lengths"].tolist(),
                "box_tensor_shape": list(batch["boxes"].shape),
                "tag_tensor_shape": list(batch["ner_tags"].shape),
                "first_tokens": batch["tokens"][0][:12],
                "first_bio_labels": batch["bio_labels"][0][:12],
                "first_image_size": list(batch["images"][0].size),
                "image_cache": str(config.image_cache),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _run_dataset_inference(args: argparse.Namespace) -> int:
    config = DatasetConfig(
        split=args.dataset_split,
        subset_size=args.limit,
        batch_size=1,
        num_workers=0,
    )
    dataset, loader = make_dataloader(config)
    ocr_config = _ocr_config(args)
    output_path = Path(args.output or f"outputs/part2_{args.dataset_split}_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    successes = failures = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            image_path = batch["image_paths"][0]
            barcode = batch["barcodes"][0]
            result = run_image_inference(image_path, barcode=barcode, ocr_config=ocr_config)
            payload = result.to_dict()
            payload["dataset_split"] = args.dataset_split
            # Gold tokens/tags are intentionally not fed into OCR inference.
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            if result.ocr_error:
                failures += 1
            else:
                successes += 1
    print(json.dumps({"split": args.dataset_split, "subset": len(dataset), "ocr_success": successes, "ocr_failures": failures, "output": str(output_path)}, indent=2))
    return 0


def _ocr_config(args: argparse.Namespace) -> OCRConfig:
    languages = tuple(lang.strip() for lang in args.languages.split(",") if lang.strip())
    return OCRConfig(
        languages=languages,
        gpu=False,
        allow_model_downloads=args.allow_model_downloads,
        model_storage_directory=Path(args.model_directory),
        cache_directory=Path(args.ocr_cache),
        preprocessing=PreprocessingConfig(max_side=args.max_side, contrast=args.contrast, grayscale=args.grayscale),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--image", type=Path, help="Run single-image inference.")
    action.add_argument("--dataset-split", choices=["train", "test"], help="Run on a bounded official split subset.")
    action.add_argument("--demo-loader", action="store_true", help="Load and print a small Dataset/DataLoader batch only.")
    parser.add_argument("--split", choices=["train", "test"], help="Official split to use with --demo-loader; defaults to train.")
    parser.add_argument("--limit", type=int, default=2, help="Subset size, limited to 1–50 images.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--barcode")
    parser.add_argument("--output", help="JSON output for --image; JSONL for dataset inference.")
    parser.add_argument("--languages", default="en", help="Comma-separated EasyOCR language codes.")
    parser.add_argument("--allow-model-downloads", action="store_true", help="Explicitly allow EasyOCR weight download if absent.")
    parser.add_argument("--model-directory", default="models/easyocr")
    parser.add_argument("--ocr-cache", default="data/part2_cache/ocr")
    parser.add_argument("--max-side", type=int, default=2000)
    parser.add_argument("--contrast", type=float, default=1.15)
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error("--limit must be 1–50; full-dataset reads are disabled.")
    if args.demo_loader:
        return _run_loader_demo(args)
    if args.dataset_split:
        return _run_dataset_inference(args)
    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    result = run_image_inference(args.image, barcode=args.barcode, ocr_config=_ocr_config(args))
    output = Path(args.output) if args.output else args.image.with_suffix(".nutrition.json")
    _write_json(result.to_dict(), output)
    print(json.dumps({"image": str(args.image), "output": str(output), "ocr_error": result.ocr_error, "nutrient_fields": sum(len(v) for m in result.nutrients.values() for v in m.values()), "unresolved": len(result.unresolved_fields), "seconds": result.processing_time_seconds}, indent=2))
    return 0 if not result.ocr_error else 2


if __name__ == "__main__":
    raise SystemExit(main())