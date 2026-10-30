"""CPU-only inference with Open Food Facts' pretrained LayoutLMv3 BIO model."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

MODEL_ID = "openfoodfacts/nutrition-extractor"
MODEL_REVISION = "4c28d479f332339ffc367d9aae66036a4583127c"
MODEL_LICENSE = "cc-by-nc-sa-4.0"
MODEL_WEIGHT_BYTES = 1_429_090_779
MIN_FREE_RAM_GB = 2.5
MIN_FREE_DISK_GB = 2.5
MODEL_CACHE_DIR = Path("part2/weights")
INFERENCE_CACHE_DIR = Path("data/part2_cache/layoutlm_bio")
MAX_MODEL_TOKENS = 512
WINDOW_WORDS = 180
WINDOW_OVERLAP_WORDS = 32
INFERENCE_SCHEMA_VERSION = "5-adaptive-sliding-structured"
REQUIRED_LOCAL_ASSETS = (
    "model.onnx", "config.json", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
)


class InsufficientResourcesError(RuntimeError):
    """Raised before a model download/load that could exhaust machine resources."""


def available_ram_bytes() -> int:
    """Return currently available physical RAM without adding a required dependency."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Could not read Windows physical-memory status.")
        return int(status.ullAvailPhys)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, OSError, ValueError):
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except ImportError as error:
            raise RuntimeError("Cannot determine available RAM on this platform.") from error


def resource_snapshot(model_dir: Path = MODEL_CACHE_DIR) -> dict[str, float | bool]:
    """Summarize memory/disk headroom and whether the complete checkpoint is cached."""
    model_dir = Path(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    free_ram_gb = available_ram_bytes() / (1024**3)
    free_disk_gb = shutil.disk_usage(model_dir.parent.resolve()).free / (1024**3)
    onnx_cached = (model_dir / "model.onnx").is_file()
    safetensors_cached = (model_dir / "model.safetensors").is_file()
    weights_cached = onnx_cached or safetensors_cached
    return {
        "available_ram_gb": round(free_ram_gb, 2),
        "free_disk_gb": round(free_disk_gb, 2),
        "weights_cached": weights_cached,
        "onnx_weights_cached": onnx_cached,
        "safetensors_cached": safetensors_cached,
        "checkpoint_size_gb": round(MODEL_WEIGHT_BYTES / (1024**3), 2),
        "safe_to_load": free_ram_gb >= MIN_FREE_RAM_GB,
        "safe_to_download": free_ram_gb >= MIN_FREE_RAM_GB and (weights_cached or free_disk_gb >= MIN_FREE_DISK_GB),
    }


def _entity_spans(labels: list[str]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    active_label: str | None = None
    start = 0
    for index, label in enumerate([*labels, "O"]):
        prefix, separator, entity_type = label.partition("-")
        is_entity = bool(separator) and prefix in {"B", "I"} and label != "O"
        if not is_entity or prefix == "B" or active_label != entity_type:
            if active_label is not None:
                entities.append({"label": active_label, "start": start, "end": index})
            active_label = entity_type if is_entity else None
            start = index
    return entities


def _label_semantics(entity_label: str) -> tuple[str, str, str | None]:
    """Return normalized nutrient, serving basis, and label-implied energy unit."""
    if entity_label == "SERVING_SIZE":
        return "serving_size", "per_serving", None
    basis = "per_serving" if entity_label.endswith("_SERVING") else "per_100g" if entity_label.endswith("_100G") else "unspecified"
    body = entity_label.removesuffix("_SERVING").removesuffix("_100G")
    energy_unit = None
    if body.startswith("ENERGY_KJ"):
        nutrient, energy_unit = "energy", "kj"
    elif body.startswith("ENERGY_KCAL"):
        nutrient, energy_unit = "energy", "kcal"
    else:
        nutrient = body.casefold()
    return nutrient, basis, energy_unit


def decode_bio_entities(
    tokens: list[str],
    labels: list[str],
    *,
    barcode: str | None = None,
    image_path: str | None = None,
    token_confidences: list[float] | None = None,
) -> dict[str, Any]:
    """Decode predicted BIO spans into conservative normalized nutrition JSON.

    Original tags/tokens and each span's raw text are retained. Fields with
    multiple numbers, unparseable text, or missing non-energy units are kept in
    `uncertain_fields` rather than silently normalized or discarded.
    """
    import re

    from part2.normalization import NUMBER_UNIT_RE, canonicalize, normalize_unit, parse_number

    if len(tokens) != len(labels):
        raise ValueError("BIO labels must align one-to-one with source OCR tokens.")
    if token_confidences is not None and len(token_confidences) != len(tokens):
        raise ValueError("Token confidence values must align with OCR tokens.")
    entities: list[dict[str, Any]] = []
    parsed_fields: list[dict[str, Any]] = []
    uncertain_fields: list[dict[str, Any]] = []
    nutrient_values: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for span in _entity_spans(labels):
        raw_tokens = tokens[span["start"]:span["end"]]
        raw_text = " ".join(raw_tokens).strip()
        nutrient, basis, label_unit = _label_semantics(span["label"])
        entity = {**span, "nutrient": nutrient, "basis": basis, "text": raw_text, "raw_text": raw_text, "raw_tokens": raw_tokens}
        if token_confidences is not None:
            entity["mean_token_confidence"] = sum(token_confidences[span["start"]:span["end"]]) / max(1, span["end"] - span["start"])
        entities.append(entity)

        matches = list(NUMBER_UNIT_RE.finditer(raw_text))
        candidates = []
        for match in matches:
            value = parse_number(match.group("number"))
            if value is not None:
                candidates.append({
                    "value": value,
                    "unit": normalize_unit(match.group("unit")) or label_unit,
                    "qualifier": match.group("qualifier"),
                    "text": match.group(0).strip(),
                    "start": match.start(),
                    "end": match.end(),
                })

        reasons: list[str] = []
        if len(candidates) != 1:
            reasons.append("no_numeric_value" if not candidates else "multiple_numeric_values_in_one_BIO_span")
        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate is not None and candidate["unit"] is None:
            reasons.append("unit_missing_or_not_read")
        if candidate is not None:
            residue = raw_text[:candidate["start"]] + " " + raw_text[candidate["end"]:]
            # Unexplained alphabetic residue often indicates OCR concatenation (e.g. 10gl).
            residue_letters = re.sub(r"(?i)kcal|kj|mg|mcg|μg|µg|ug|ml|g|l|%", "", residue)
            if re.search(r"[A-Za-zÀ-ÿ]", residue_letters):
                reasons.append("unparsed_letters_near_numeric_value")

        record: dict[str, Any] = {
            **entity,
            "parse_status": "uncertain" if reasons else "parsed",
            "uncertainty_reasons": reasons,
        }
        if candidate is not None:
            record.update({
                "value": candidate["value"],
                "unit": candidate["unit"],
                "qualifier": candidate["qualifier"],
                "parsed_text": candidate["text"],
            })
            if candidate["unit"] is not None:
                normalized_value, normalized_unit = canonicalize(candidate["value"], candidate["unit"])
                record["normalized_value"] = normalized_value
                record["normalized_unit"] = normalized_unit

        if reasons:
            uncertain_fields.append(record)
        else:
            parsed_fields.append(record)
            nutrient_values.setdefault(basis, {}).setdefault(nutrient, []).append(record)

    # Preserve duplicates; mark repeated semantic keys and conflicting values.
    for basis, nutrients in nutrient_values.items():
        for nutrient, entries in nutrients.items():
            seen_units: dict[str | None, list[dict[str, Any]]] = {}
            for entry in entries:
                seen_units.setdefault(entry["unit"], []).append(entry)
            for unit_entries in seen_units.values():
                if len(unit_entries) > 1:
                    values = {entry.get("value") for entry in unit_entries}
                    for entry in unit_entries:
                        entry["duplicate_entity"] = True
                        entry["duplicate_values_conflict"] = len(values) > 1
                        entry["uncertainty_reasons"].append(
                            "duplicate_field_conflicting_values" if len(values) > 1 else "duplicate_field"
                        )
                        entry["parse_status"] = "uncertain"
                        uncertain_fields.append(entry)
            nutrients[nutrient] = [entry for entry in entries if entry["parse_status"] == "parsed"]

    parsed_fields = [entry for entry in parsed_fields if entry["parse_status"] == "parsed"]
    nutrient_values = {
        basis: {nutrient: entries for nutrient, entries in nutrients.items() if entries}
        for basis, nutrients in nutrient_values.items()
    }
    nutrient_values = {basis: nutrients for basis, nutrients in nutrient_values.items() if nutrients}

    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "barcode": barcode,
        "image_path": image_path,
        "raw_predictions": {"tokens": tokens, "bio_labels": labels},
        "entities": entities,
        "nutrients": nutrient_values,
        "parsed_fields": parsed_fields,
        "uncertain_fields": uncertain_fields,
    }


def build_structured_output(record: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Preserve raw token predictions and append conservative normalized values."""
    return decode_bio_entities(
        [str(token) for token in record["tokens"]],
        prediction["predicted_labels"],
        barcode=str(record.get("barcode", "")) or None,
        image_path=str(record.get("image_path", "")) or None,
        token_confidences=prediction.get("prediction_confidences"),
    )


def to_coarse_bio(labels: list[str]) -> list[str]:
    """Project the 71-way model/dataset labels to shared value/basis BIO types."""
    projected: list[str] = []
    for label in labels:
        if label == "O":
            projected.append("O")
            continue
        prefix, separator, entity = label.partition("-")
        if not separator or prefix not in {"B", "I"}:
            projected.append("O")
        elif entity == "SERVING_SIZE":
            projected.append(f"{prefix}-BASIS")
        else:
            projected.append(f"{prefix}-VALUE")
    return projected


def rule_baseline_coarse_bio(tokens: list[str], boxes: list[list[int]], image_size: tuple[int, int]) -> list[str]:
    """Map the existing lexical OCR rule tagger into comparable coarse BIO types.

    Nutrient-name tokens map to O because the corpus does not generally annotate
    nutrient names. Numeric values begin VALUE spans; a following unit continues
    the span. Basis phrases map to BASIS spans. This is an explicit coarse task,
    not a fabricated mapping to the model's fine-grained nutrient/basis classes.
    """
    from nutrition_extraction.ocr import OCRToken
    from nutrition_extraction.extraction import tag_ocr_tokens

    width, height = image_size
    detections = []
    for token, box in zip(tokens, boxes):
        x0, y0, x1, y1 = box
        pixel_box = (
            round(x0 * width / 1000), round(y0 * height / 1000),
            round(x1 * width / 1000), round(y1 * height / 1000),
        )
        detections.append(OCRToken(str(token), pixel_box, 1.0))
    tagged = tag_ocr_tokens(detections)
    coarse: list[str] = []
    previous_value = False
    for item in tagged:
        kind = str(item.get("entity_type", "O"))
        if kind == "VALUE":
            coarse.append("B-VALUE")
            previous_value = True
        elif kind == "UNIT" and previous_value:
            coarse.append("I-VALUE")
        elif kind == "BASIS":
            coarse.append("B-BASIS")
            previous_value = False
        else:
            coarse.append("O")
            previous_value = False
    return coarse


def _cache_key(image_path: Path, tokens: list[str], boxes: list[list[int]]) -> str:
    digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    payload = json.dumps(
        {"model": MODEL_ID, "revision": MODEL_REVISION, "inference_schema": INFERENCE_SCHEMA_VERSION,
         "image": digest, "tokens": tokens, "boxes": boxes},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sliding_window_logits(
    image: Image.Image,
    tokens: list[str],
    boxes: list[list[int]],
    *,
    model: Any,
    processor: Any,
    id2label: dict[int, str],
) -> tuple[list[Any], int, int]:
    """Predict first-subword logits for every source word using overlapping windows."""
    import numpy as np

    window_words = min(WINDOW_WORDS, max(1, len(tokens)))
    total_windows = 0
    while True:
        logit_sums: list[Any | None] = [None] * len(tokens)
        logit_counts = [0] * len(tokens)
        overlap_words = min(WINDOW_OVERLAP_WORDS, max(0, window_words // 4))
        step_words = max(1, window_words - overlap_words)
        starts = list(range(0, len(tokens), step_words))
        if starts and starts[-1] + window_words >= len(tokens):
            last_needed = next((i + 1 for i, start in enumerate(starts) if start + window_words >= len(tokens)), len(starts))
            starts = starts[:last_needed]

        for start_index in starts:
            end_index = min(len(tokens), start_index + window_words)
            window_tokens = tokens[start_index:end_index]
            encoding = processor(
                images=image,
                text=window_tokens,
                boxes=boxes[start_index:end_index],
                truncation=True,
                max_length=MAX_MODEL_TOKENS,
                return_tensors="np" if hasattr(model, "get_inputs") else "pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            if hasattr(model, "get_inputs"):
                encoded = dict(encoding)
                inputs = {}
                for item in model.get_inputs():
                    if item.name not in encoded:
                        raise ValueError(f"Processor omitted required ONNX input {item.name!r}.")
                    array = np.asarray(encoded[item.name])
                    if "int64" in item.type:
                        array = array.astype(np.int64, copy=False)
                    elif "float" in item.type:
                        array = array.astype(np.float32, copy=False)
                    inputs[item.name] = array
                outputs = model.run(None, inputs)
                logits = next((value for value in outputs if value.ndim == 3 and value.shape[-1] == len(id2label)), None)
                if logits is None:
                    shapes = [getattr(value, "shape", None) for value in outputs]
                    raise ValueError(f"Could not identify {len(id2label)}-class ONNX token logits: {shapes}.")
                logits = logits[0]
            else:
                import torch

                inputs = {key: value.to("cpu") for key, value in encoding.items() if hasattr(value, "to")}
                with torch.inference_mode():
                    logits = model(**inputs).logits[0].cpu().numpy()

            first_subwords: set[int] = set()
            for subword_index, word_id in enumerate(word_ids):
                if word_id is None or word_id in first_subwords or word_id >= len(window_tokens):
                    continue
                first_subwords.add(word_id)
                source_index = start_index + word_id
                row_logits = np.asarray(logits[subword_index], dtype=np.float64)
                if logit_sums[source_index] is None:
                    logit_sums[source_index] = row_logits.copy()
                else:
                    logit_sums[source_index] += row_logits
                logit_counts[source_index] += 1
            total_windows += 1

        uncovered = [index for index, count in enumerate(logit_counts) if count == 0]
        if not uncovered:
            return [logit_sum / count for logit_sum, count in zip(logit_sums, logit_counts)], total_windows, window_words
        if window_words <= 16:
            raise RuntimeError(f"Sliding-window tokenizer alignment still misses original token positions: {uncovered[:12]}.")
        window_words = max(16, int(window_words * 0.75))


def load_pretrained_model(
    model_dir: Path = MODEL_CACHE_DIR,
    *,
    min_free_ram_gb: float = MIN_FREE_RAM_GB,
    min_torch_ram_gb: float = 6.0,
) -> tuple[Any, Any, dict[int, str], dict[str, Any]]:
    """Load local ONNX (preferred) or safetensors weights after resource checks.

    This function never downloads model weights. The local ONNX Runtime session
    uses sequential CPU execution and one intra-op thread to limit peak memory.
    """
    model_dir = Path(model_dir)
    resources = resource_snapshot(model_dir)
    onnx_path = model_dir / "model.onnx"
    safetensors_path = model_dir / "model.safetensors"
    use_onnx = onnx_path.is_file()
    if not use_onnx and not safetensors_path.is_file():
        raise FileNotFoundError(
            f"No local model.onnx or model.safetensors was found in {model_dir}. "
            "Place the manually downloaded checkpoint there; model weights will not be downloaded automatically."
        )

    required_ram_gb = min_free_ram_gb if use_onnx else min_torch_ram_gb
    if resources["available_ram_gb"] < required_ram_gb:
        raise InsufficientResourcesError(
            f"Only {resources['available_ram_gb']:.2f} GiB RAM is currently available; "
            f"at least {required_ram_gb:.1f} GiB is required before loading the local "
            f"{resources['checkpoint_size_gb']:.2f} GiB {'ONNX' if use_onnx else 'FP32'} checkpoint. "
            "No weights were loaded. Close memory-heavy apps and retry."
        )

    required_assets = list(REQUIRED_LOCAL_ASSETS if use_onnx else (
        "model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
    ))
    missing_assets = [name for name in required_assets if not (model_dir / name).is_file()]
    if missing_assets:
        raise FileNotFoundError(
            f"Local model sidecar files are missing from {model_dir}: {', '.join(missing_assets)}. "
            "Only the small config/tokenizer files are needed; keep the checkpoint local."
        )

    try:
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Transformers processor dependencies are unavailable. Install the existing bio-inference extra."
        ) from error
    from part2.dataset import BIO_LABELS

    processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False, local_files_only=True)
    if use_onnx:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "The local ONNX model is present but ONNX Runtime is not installed. "
                "Install the CPU package with `pip install onnxruntime`; the model file will not be downloaded again."
            ) from error
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        model = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        runtime_info = {
            "runtime": "onnxruntime-cpu",
            "providers": model.get_providers(),
            "input_names": [item.name for item in model.get_inputs()],
            "output_names": [item.name for item in model.get_outputs()],
            "checkpoint_path": str(onnx_path.resolve()),
            "checkpoint_size_bytes": onnx_path.stat().st_size,
        }
    else:
        import torch

        model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(model_dir), local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float32,
        )
        model.to(torch.device("cpu"))
        model.eval()
        runtime_info = {
            "runtime": "transformers-cpu",
            "providers": ["CPU"],
            "input_names": [],
            "output_names": [],
            "checkpoint_path": str(safetensors_path.resolve()),
            "checkpoint_size_bytes": safetensors_path.stat().st_size,
        }
    raw_id2label = json.loads((model_dir / "config.json").read_text(encoding="utf-8-sig"))["id2label"]
    id2label = {int(key): str(value) for key, value in raw_id2label.items()}
    if set(id2label.values()) != set(BIO_LABELS):
        raise ValueError("Pinned model BIO labels do not match the official Part 2 dataset label vocabulary.")
    return model, processor, id2label, runtime_info


def predict_record(
    record: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    id2label: dict[int, str],
    cache_dir: Path = INFERENCE_CACHE_DIR,
) -> dict[str, Any]:
    """Predict BIO labels for one official dataset row using its original inputs."""
    from part2.dataset import BIO_LABELS

    image = record["image"].convert("RGB")
    tokens = [str(token) for token in record["tokens"]]
    boxes = [[int(value) for value in box] for box in record["bboxes"]]
    gold_tags = record.get("ner_tags")
    if len(tokens) != len(boxes) or (gold_tags is not None and len(tokens) != len(gold_tags)):
        raise ValueError("Model inference requires aligned OCR token/box sequences and, when provided, ground-truth tags.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(Path(record['image_path']), tokens, boxes)}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["gold_labels"] = [BIO_LABELS[int(tag)] for tag in gold_tags] if gold_tags is not None else None
        cached["cache_hit"] = True
        return cached

    rss_before_mb: float | None = None
    process = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss_before_mb = process.memory_info().rss / (1024**2)
    except ImportError:
        pass
    available_before = available_ram_bytes()
    started = time.perf_counter()
    average_logits, windows_used, final_window_words = _sliding_window_logits(
        image, tokens, boxes, model=model, processor=processor, id2label=id2label,
    )

    import numpy as np

    predicted_labels: list[str] = []
    prediction_confidences: list[float] = []
    for averaged_logits in average_logits:
        shifted = averaged_logits - float(np.max(averaged_logits))
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        label_id = int(np.argmax(probabilities))
        predicted_labels.append(id2label[label_id])
        prediction_confidences.append(float(probabilities[label_id]))
    covered_count = len(predicted_labels)
    coverage_mask = [True] * covered_count
    if covered_count != len(tokens):
        raise RuntimeError(f"Inference covered {covered_count}/{len(tokens)} source tokens; refusing to score incomplete predictions.")
    elapsed = time.perf_counter() - started
    rss_after_mb = process.memory_info().rss / (1024**2) if process is not None else None
    available_after = available_ram_bytes()
    structured_output = decode_bio_entities(
        tokens, predicted_labels,
        barcode=str(record.get("barcode", "")) or None,
        image_path=str(record.get("image_path", "")) or None,
        token_confidences=prediction_confidences,
    )
    extracted_entities = structured_output["entities"]
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "device": "cpu",
        "runtime": "onnxruntime-cpu" if hasattr(model, "get_inputs") else "transformers-cpu",
        "predicted_labels": predicted_labels,
        "prediction_confidences": prediction_confidences,
        "gold_labels": [BIO_LABELS[int(tag)] for tag in record["ner_tags"]] if record.get("ner_tags") is not None else None,
        "covered_token_count": int(covered_count),
        "coverage_mask": coverage_mask,
        "coverage_fraction": covered_count / len(tokens) if tokens else 1.0,
        "inference_windows": windows_used,
        "window_word_limit": final_window_words,
        "token_count": len(tokens),
        "inference_seconds": elapsed,
        "process_rss_before_mb": rss_before_mb,
        "process_rss_after_mb": rss_after_mb,
        "process_rss_delta_mb": rss_after_mb - rss_before_mb if rss_before_mb is not None and rss_after_mb is not None else None,
        "available_ram_delta_mb": (available_before - available_after) / (1024**2),
        "extracted_entities": extracted_entities,
        "structured_output": structured_output,
        "cache_hit": False,
    }
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(cache_path)
    return result


def predict_uploaded_image(
    image_path: str | Path,
    *,
    model: Any,
    processor: Any,
    id2label: dict[int, str],
    languages: tuple[str, ...] = ("fr", "en"),
    cache_dir: Path = INFERENCE_CACHE_DIR,
    easyocr_model_dir: Path = Path("models/easyocr"),
    ocr_cache_dir: Path = Path("data/part2_cache/ocr"),
) -> dict[str, Any]:
    """Run cached CPU EasyOCR then local BIO inference on an unseen image.

    OCR download permission is explicitly disabled. No gold labels are attached
    to the uploaded record; only predicted tags and normalized candidates return.
    """
    from part2.config import OCRConfig, PreprocessingConfig
    from part2.ocr import run_easyocr
    from part2.preprocessing import load_image, preprocess_image

    path = Path(image_path)
    import torch

    torch.set_num_threads(1)
    ocr_config = OCRConfig(
        languages=languages,
        gpu=False,
        allow_model_downloads=False,
        model_storage_directory=Path(easyocr_model_dir),
        cache_directory=Path(ocr_cache_dir),
        preprocessing=PreprocessingConfig(max_side=2000, contrast=1.15, grayscale=False, resize=True),
    )
    ocr_result = run_easyocr(path, ocr_config)
    if ocr_result.error:
        return {
            "image_path": str(path),
            "ocr_error": ocr_result.error,
            "ocr_cache_hit": ocr_result.from_cache,
            "gold_available": False,
        }
    if not ocr_result.detections:
        return {
            "image_path": str(path),
            "ocr_error": "EasyOCR returned no text detections.",
            "ocr_cache_hit": ocr_result.from_cache,
            "gold_available": False,
        }

    original_image = load_image(path)
    ocr_image = preprocess_image(original_image, ocr_config.preprocessing)
    width, height = ocr_image.size
    tokens: list[str] = []
    boxes: list[list[int]] = []
    confidences: list[float] = []
    for detection in ocr_result.detections:
        xs = [point[0] for point in detection.bbox]
        ys = [point[1] for point in detection.bbox]
        x0, x1 = max(0, min(xs)), min(width, max(xs))
        y0, y1 = max(0, min(ys)), min(height, max(ys))
        if x1 <= x0 or y1 <= y0:
            continue
        normalized_box = [
            max(0, min(1000, round(x0 * 1000 / width))),
            max(0, min(1000, round(y0 * 1000 / height))),
            max(0, min(1000, round(x1 * 1000 / width))),
            max(0, min(1000, round(y1 * 1000 / height))),
        ]
        for word in detection.text.split():
            tokens.append(word)
            boxes.append(normalized_box.copy())
            confidences.append(float(detection.confidence))
    if not tokens:
        return {
            "image_path": str(path),
            "ocr_error": "EasyOCR detections did not contain valid non-empty boxes.",
            "ocr_cache_hit": ocr_result.from_cache,
            "gold_available": False,
        }

    record = {
        "image": original_image,
        "image_path": str(path),
        "barcode": None,
        "tokens": tokens,
        "bboxes": boxes,
        "ocr_confidences": confidences,
        # No ner_tags: uploaded-image inference has no ground truth.
    }
    prediction = predict_record(record, model=model, processor=processor, id2label=id2label, cache_dir=cache_dir)
    structured = build_structured_output(record, prediction)
    structured["ocr_provenance"] = {
        "engine": "EasyOCR CPU, quantize=False",
        "box_granularity": "detection box shared by whitespace-split words; not measured word boxes",
        "gold_available": False,
    }
    return {
        "image_path": str(path),
        "image": original_image,
        "ocr_result": ocr_result,
        "ocr_cache_hit": ocr_result.from_cache,
        "ocr_elapsed_seconds": ocr_result.elapsed_seconds,
        "tokens": tokens,
        "boxes": boxes,
        "ocr_confidences": confidences,
        "prediction": prediction,
        "structured_output": structured,
        "gold_available": False,
        "warning": "EasyOCR words share their parent detection boxes and have no gold alignment; do not report BIO or field accuracy. Rotation, blur and OCR grouping can prevent extraction.",
    }


def analyze_bio_errors(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return actual token disagreements with local OCR context and error type."""
    if len(records) != len(predictions):
        raise ValueError("Records and prediction counts must match for error analysis.")
    errors = []
    for record, prediction in zip(records, predictions):
        tokens = [str(token) for token in record["tokens"]]
        gold = prediction.get("gold_labels")
        predicted = prediction.get("predicted_labels")
        if gold is None:
            continue
        coverage = prediction.get("coverage_mask", [True] * len(tokens))
        for index, (gold_label, predicted_label) in enumerate(zip(gold, predicted)):
            if gold_label == predicted_label or not coverage[index]:
                continue
            gold_prefix, _, gold_entity = gold_label.partition("-")
            pred_prefix, _, pred_entity = predicted_label.partition("-")
            if gold_label == "O" and predicted_label != "O":
                error_type = "false_positive_entity"
            elif predicted_label == "O" and gold_label != "O":
                error_type = "missed_gold_entity"
            elif gold_entity == pred_entity and gold_entity and gold_prefix != pred_prefix:
                error_type = "bio_boundary_mismatch"
            else:
                error_type = "wrong_entity_or_basis_label"
            errors.append({
                "barcode": record.get("barcode"),
                "image_path": str(record.get("image_path", "")),
                "token_index": index,
                "ocr_token": tokens[index],
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "error_type": error_type,
                "context": " ".join(tokens[max(0, index - 4): min(len(tokens), index + 5)]),
                "ocr_text_independently_verified": False,
            })
    return errors


def fetch_openfoodfacts_product(
    barcode: str,
    *,
    cache_dir: Path = Path("data/part2_cache/openfoodfacts_products"),
    timeout: float = 12.0,
) -> dict[str, Any]:
    """Fetch/cache only basic product nutrient metadata for separate reference checks."""
    normalized_barcode = str(barcode).strip()
    if not normalized_barcode.isdigit():
        raise ValueError(f"Invalid product barcode: {barcode!r}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{normalized_barcode}.json"
    if cache_path.is_file():
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        result["from_cache"] = True
        return result

    url = f"https://world.openfoodfacts.org/api/v2/product/{normalized_barcode}.json?fields=code,product_name,nutriments,quantity,serving_size"
    request = Request(url, headers={"User-Agent": "nutrition-extraction-research/0.1 (local evaluation)"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"barcode": normalized_barcode, "error": f"{type(error).__name__}: {error}", "from_cache": False}
    if payload.get("status") != 1 or not isinstance(payload.get("product"), dict):
        result = {"barcode": normalized_barcode, "found": False, "product": None, "from_cache": False}
    else:
        result = {"barcode": normalized_barcode, "found": True, "product": payload["product"], "from_cache": False}
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def compare_fields_to_off_metadata(
    structured_output: dict[str, Any],
    product_response: dict[str, Any],
    *,
    relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Compare only numeric fields for which API reference and units are compatible.

    This product API is weak, mutable product metadata, not visual ground truth;
    results are reported separately and never added to model BIO evaluation.
    """
    from part2.normalization import canonicalize, normalize_unit
    import math

    def comparable_quantity(value: float, unit: str | None) -> tuple[float, str | None]:
        return canonicalize(value, unit) if unit in {"g", "mg", "mcg", "ug"} else (value, unit)

    if product_response.get("error") or not product_response.get("found"):
        return {
            "reference_source": "Open Food Facts product API",
            "reference_is_visual_gold": False,
            "status": product_response.get("error", "product_not_found"),
            "comparisons": [],
        }
    nutriments = product_response["product"].get("nutriments") or {}
    name_map = {
        "fat": "fat", "saturated_fat": "saturated-fat", "carbohydrates": "carbohydrates",
        "sugars": "sugars", "fiber": "fiber", "proteins": "proteins", "salt": "salt",
        "sodium": "sodium", "added_sugars": "added-sugars", "trans_fat": "trans-fat",
        "cholesterol": "cholesterol", "calcium": "calcium", "iron": "iron", "potassium": "potassium",
        "vitamin_d": "vitamin-d",
    }
    basis_suffixes = {"per_100g": "100g", "per_100ml": "100ml", "per_serving": "serving"}
    unit_by_off_name = {"energy-kj": "kj", "energy-kcal": "kcal"}
    reference_values: dict[tuple[str, str, str], float] = {}
    for nutrient, off_name in name_map.items():
        for basis, suffix in basis_suffixes.items():
            reference_key = f"{off_name}_{suffix}"
            raw_value = nutriments.get(reference_key)
            raw_unit = nutriments.get(f"{reference_key}_unit") or unit_by_off_name.get(off_name, "g")
            try:
                value, unit = comparable_quantity(float(raw_value), normalize_unit(raw_unit))
                if math.isfinite(value):
                    reference_values[(nutrient, basis, str(unit))] = value
            except (TypeError, ValueError):
                continue
    for basis, suffix in basis_suffixes.items():
        for off_name in ("energy-kj", "energy-kcal"):
            reference_key = f"{off_name}_{suffix}"
            try:
                value = float(nutriments[reference_key])
                if math.isfinite(value):
                    reference_values[("energy", basis, unit_by_off_name[off_name])] = value
            except (KeyError, TypeError, ValueError):
                continue

    predicted_values: dict[tuple[str, str, str], dict[str, Any]] = {}
    for field in structured_output.get("parsed_fields", []):
        unit = normalize_unit(field.get("unit"))
        value, unit = comparable_quantity(float(field["value"]), unit)
        key = (str(field.get("nutrient")), str(field.get("basis")), str(unit))
        if key[0] == "serving_size" or key not in reference_values:
            continue
        predicted_values.setdefault(key, {**field, "value": value})

    comparisons: list[dict[str, Any]] = []
    true_positive = false_positive = false_negative = exact_count = 0
    for key, reference_value in reference_values.items():
        prediction = predicted_values.get(key)
        if prediction is None:
            false_negative += 1
            comparisons.append({
                "nutrient": key[0], "basis": key[1], "unit": key[2],
                "prediction": None, "off_value": reference_value,
                "exact_value_match": False, "within_tolerance": False,
                "status": "reference_value_without_model_field",
            })
            continue
        predicted_value = float(prediction["value"])
        exact_match = not prediction.get("qualifier") and predicted_value == reference_value
        within_tolerance = not prediction.get("qualifier") and abs(predicted_value - reference_value) <= max(abs(reference_value) * relative_tolerance, 1e-9)
        exact_count += int(exact_match)
        true_positive += int(exact_match)
        false_positive += int(not exact_match)
        false_negative += int(not exact_match)
        comparisons.append({
            "nutrient": key[0], "basis": key[1], "unit": key[2],
            "prediction": predicted_value, "off_value": reference_value,
            "exact_value_match": exact_match, "within_tolerance": within_tolerance,
            "status": "exact_value_match" if exact_match else "value_mismatch",
            "model_span": prediction["raw_text"],
        })
    comparable = len(reference_values)
    scored_prediction_count = sum(key in reference_values for key in predicted_values)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "reference_source": "Open Food Facts product API",
        "reference_is_visual_gold": False,
        "product_name": product_response["product"].get("product_name"),
        "api_reference_fields_available": comparable,
        "model_fields_with_compatible_api_reference": scored_prediction_count,
        "exact_value_count": exact_count,
        "exact_value_accuracy": exact_count / scored_prediction_count if scored_prediction_count else None,
        "field_tuple_precision": precision if comparable else None,
        "field_tuple_recall": recall if comparable else None,
        "field_tuple_f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if comparable else None,
        "field_tuple_metric_note": "Exact nutrient+basis+unit+value tuple against currently available Open Food Facts product nutriments; this API metadata is not image-verified truth.",
        "comparisons": comparisons,
    }


def compare_fields_to_gold(
    tokens: list[str], gold_labels: list[str], predicted_labels: list[str],
) -> dict[str, Any]:
    """Score exact normalized field tuples derived from aligned corpus BIO spans.

    This tests label decoding, not independent OCR transcription or visual truth.
    Unparseable/duplicate gold spans are disclosed and excluded from tuple scoring.
    """
    gold = decode_bio_entities(tokens, gold_labels)
    predicted = decode_bio_entities(tokens, predicted_labels)
    from part2.bio_review import evaluate_fields

    numerical_metrics = evaluate_fields(
        gold["parsed_fields"], predicted["parsed_fields"],
        reference_source="decoded official aligned ner_tags (not independent visual transcription)",
    )

    def as_tuple(row: dict[str, Any], side: str) -> list[Any]:
        return [row["nutrient"], row["basis"], row["unit"], row[f"{side}_value"], row[f"{side}_qualifier"]]

    return {
        **numerical_metrics,
        "reference_source": "decoded official aligned ner_tags (not independent visual transcription)",
        "gold_uncertain_spans": len(gold["uncertain_fields"]),
        "predicted_uncertain_spans": len(predicted["uncertain_fields"]),
        "precision": numerical_metrics["field_precision"], "recall": numerical_metrics["field_recall"],
        "f1": numerical_metrics["field_f1"],
        "exact_tuple_accuracy_over_gold": numerical_metrics["field_recall"],
        "missing_gold_tuples": [as_tuple(row, "reference") for row in numerical_metrics["comparisons"] if row["reference_value"] is not None and row["status"] != "exact_match"],
        "unmatched_prediction_tuples": [as_tuple(row, "predicted") for row in numerical_metrics["comparisons"] if row["predicted_value"] is not None and row["status"] != "exact_match"],
    }


def evaluate_sequences(
    gold_sequences: list[list[str]], predictions: list[list[str]], *, aligned: bool,
    coverage_masks: list[list[bool]] | None = None,
) -> dict[str, Any]:
    """Exact BIO entity-span precision/recall/F1 on explicitly aligned sequences."""
    if not aligned:
        raise ValueError("BIO evaluation requires original dataset token alignment.")
    if len(gold_sequences) != len(predictions):
        raise ValueError("Gold/prediction record counts differ.")
    if coverage_masks is None:
        coverage_masks = [[True] * len(sequence) for sequence in gold_sequences]
    if len(coverage_masks) != len(gold_sequences):
        raise ValueError("Coverage mask count must equal the sequence count.")
    tp = predicted = gold_count = correct_tokens = total_tokens = 0
    all_gold_spans = covered_gold_spans = 0
    for gold, pred, covered_mask in zip(gold_sequences, predictions, coverage_masks):
        if len(gold) != len(pred):
            raise ValueError("Each aligned gold/prediction sequence must have equal length.")
        if len(covered_mask) != len(gold):
            raise ValueError("Each token coverage mask must align to its token sequence.")
        correct_tokens += sum(gold_tag == predicted_tag for gold_tag, predicted_tag, is_covered in zip(gold, pred, covered_mask) if is_covered)
        total_tokens += sum(covered_mask)
        gold_spans = {(item["label"], item["start"], item["end"]) for item in _entity_spans(gold)}
        pred_spans = {(item["label"], item["start"], item["end"]) for item in _entity_spans(pred)}
        covered_gold = {span for span in gold_spans if all(covered_mask[span[1]:span[2]])}
        covered_pred = {span for span in pred_spans if all(covered_mask[span[1]:span[2]])}
        all_gold_spans += len(gold_spans)
        covered_gold_spans += len(covered_gold)
        tp += len(covered_gold & covered_pred)
        predicted += len(covered_pred)
        gold_count += len(covered_gold)
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold_count if gold_count else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive_spans": tp,
        "predicted_spans": predicted,
        "gold_spans": gold_count,
        "token_accuracy": correct_tokens / total_tokens if total_tokens else 0.0,
        "token_support": total_tokens,
        "covered_gold_spans": covered_gold_spans,
        "total_gold_spans": all_gold_spans,
        "gold_entity_coverage": covered_gold_spans / all_gold_spans if all_gold_spans else 1.0,
        "token_coverage": total_tokens / sum(len(sequence) for sequence in gold_sequences) if any(gold_sequences) else 1.0,
    }


def save_prediction_bundle(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    runtime_info: dict[str, Any],
    fine_metrics: dict[str, Any],
    coarse_model_metrics: dict[str, Any],
    coarse_rule_metrics: dict[str, Any],
    coarse_rule_labels: list[list[str]],
    evaluation_metadata: dict[str, Any] | None = None,
    output_path: Path = INFERENCE_CACHE_DIR / "latest_test_bundle.json",
) -> Path:
    """Atomically cache aligned outputs for UI display without reloading weights."""
    if not (len(records) == len(predictions) == len(coarse_rule_labels)):
        raise ValueError("Bundle records, predictions, and rule labels must have equal lengths.")
    samples = []
    for record, prediction, rule_labels in zip(records, predictions, coarse_rule_labels):
        samples.append({
            "barcode": str(record.get("barcode", "")),
            "source_index": record.get("source_index"),
            "image_path": str(record["image_path"]),
            "image_size": list(record["image"].size),
            "tokens": [str(token) for token in record["tokens"]],
            "bboxes": [[int(value) for value in box] for box in record["bboxes"]],
            "gold_labels": prediction["gold_labels"],
            "predicted_labels": prediction["predicted_labels"],
            "structured_output": prediction.get("structured_output"),
            "coarse_rule_labels": rule_labels,
            "covered_token_count": int(prediction["covered_token_count"]),
            "coverage_mask": prediction.get("coverage_mask"),
            "inference_windows": prediction.get("inference_windows"),
            "token_count": int(prediction["token_count"]),
            "inference_seconds": float(prediction["inference_seconds"]),
            "process_rss_after_mb": prediction.get("process_rss_after_mb"),
            "process_rss_delta_mb": prediction.get("process_rss_delta_mb"),
            "available_ram_delta_mb": prediction.get("available_ram_delta_mb"),
            "cache_hit": bool(prediction.get("cache_hit", False)),
            "extracted_entities": prediction["extracted_entities"],
        })
    bundle = {
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "runtime": runtime_info,
        "device": "cpu",
        "split": "test",
        "sample_count": len(samples),
        "fine_grained_metrics": fine_metrics,
        "coarse_model_metrics": coarse_model_metrics,
        "coarse_rule_metrics": coarse_rule_metrics,
        "evaluation_metadata": evaluation_metadata or {},
        "samples": samples,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(destination)
    return destination


def evaluate_official_test_subset(
    *,
    model: Any,
    processor: Any,
    id2label: dict[int, str],
    sample_count: int = 20,
    seed: int = 42,
    image_cache: Path = Path("data/part2_cache/images"),
    inference_cache: Path = INFERENCE_CACHE_DIR,
    min_available_ram_gb: float = 0.75,
) -> dict[str, Any]:
    """Evaluate a deterministic random sample of the full official test split.

    Only the selected examples' images are fetched; split rows are read as bounded
    metadata pages. Per-image inference remains sequential and aborts further
    image retrieval if available RAM drops below the continuation threshold.
    """
    from part2.config import DatasetConfig
    from part2.dataset import HuggingFaceNutritionDataset, BIO_LABELS

    if not 1 <= sample_count <= 30:
        raise ValueError("sample_count must be between 1 and 30.")
    probe = HuggingFaceNutritionDataset(DatasetConfig(
        split="test", subset_size=1, batch_size=1, image_cache=Path(image_cache),
    ))
    total_test_rows = int(probe.total_split_rows)
    actual_sample_count = min(sample_count, total_test_rows)
    selected_indices = sorted(random.Random(seed).sample(range(total_test_rows), actual_sample_count))
    selected_dataset = HuggingFaceNutritionDataset(
        DatasetConfig(split="test", subset_size=actual_sample_count, batch_size=1, image_cache=Path(image_cache)),
        row_indices=selected_indices,
    )

    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for selected_position, source_index in enumerate(selected_indices):
        available_gb = available_ram_bytes() / (1024**3)
        if available_gb < min_available_ram_gb:
            for remaining_index in selected_indices[selected_position:]:
                failures.append({
                    "source_index": remaining_index,
                    "status": "skipped_resource_limit",
                    "error_type": "InsufficientResourcesError",
                    "error": f"Stopped before fetching image: {available_gb:.2f} GiB available; need {min_available_ram_gb:.2f} GiB with the model session resident.",
                })
            break
        try:
            record = selected_dataset[selected_position]
            record["source_index"] = source_index
            prediction = predict_record(
                record, model=model, processor=processor, id2label=id2label,
                cache_dir=Path(inference_cache),
            )
            records.append(record)
            predictions.append(prediction)
        except Exception as error:
            meta = selected_dataset.records[selected_position].get("meta") or {}
            failures.append({
                "source_index": source_index,
                "barcode": str(meta.get("barcode", "")),
                "error_type": type(error).__name__,
                "error": str(error),
            })

    if predictions:
        gold_sequences = [[BIO_LABELS[int(tag)] for tag in record["ner_tags"]] for record in records]
        predicted_sequences = [prediction["predicted_labels"] for prediction in predictions]
        coverage_masks = [prediction["coverage_mask"] for prediction in predictions]
        coarse_gold = [to_coarse_bio(sequence) for sequence in gold_sequences]
        coarse_model = [to_coarse_bio(sequence) for sequence in predicted_sequences]
        coarse_rules = [
            rule_baseline_coarse_bio(record["tokens"], record["bboxes"], record["image"].size)
            for record in records
        ]
        metrics = {
            "fine_grained": evaluate_sequences(gold_sequences, predicted_sequences, aligned=True, coverage_masks=coverage_masks),
            "coarse_model": evaluate_sequences(coarse_gold, coarse_model, aligned=True, coverage_masks=coverage_masks),
            "coarse_rule": evaluate_sequences(coarse_gold, coarse_rules, aligned=True, coverage_masks=coverage_masks),
        }
        error_rows = analyze_bio_errors(records, predictions)
        field_comparisons = [
            {"barcode": record.get("barcode"), **compare_fields_to_gold(record["tokens"], gold, predicted)}
            for record, gold, predicted in zip(records, gold_sequences, predicted_sequences)
        ]
        rule_labels = coarse_rules
    else:
        metrics, error_rows, rule_labels = {}, [], []
        field_comparisons = []

    return {
        "seed": seed,
        "requested_images": sample_count,
        "successful_images": len(records),
        "official_test_rows": total_test_rows,
        "selected_source_indices": selected_indices,
        "failures": failures,
        "records": records,
        "predictions": predictions,
        "rule_labels": rule_labels,
        "metrics": metrics,
        "error_rows": error_rows,
        "field_comparisons": field_comparisons,
    }


def draw_prediction_boxes(image: Image.Image, boxes: list[list[int]], labels: list[str]) -> Image.Image:
    """Return a non-mutating RGB preview with predicted non-O token boxes."""
    from PIL import ImageDraw

    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    for box, label in zip(boxes, labels):
        if label == "O":
            continue
        x0, y0, x1, y1 = box
        rect = (round(x0 * width / 1000), round(y0 * height / 1000), round(x1 * width / 1000), round(y1 * height / 1000))
        draw.rectangle(rect, outline="#00E5A8", width=max(2, round(min(width, height) / 300)))
    return preview
