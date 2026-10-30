"""Support only for the new integration notebook; existing loaders are untouched."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DATASET = "openfoodfacts/nutrient-detection-layout"
SOURCE_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
SAMPLE_NAMES = {"openfoodfacts_sample.csv", "openfoodfacts_eda_enriched.csv"}
PRODUCT_FIELDS = {"product_name": "product_name", "brand": "brands", "category": "categories",
                  "country": "countries", "serving_size": "serving_size", "nutrition_image_url": "image_nutrition_url"}


def _sample_provenance(text):
    return bool(re.search(r"(?m)^\s*rows\s*=\s*50000\s*$", text))


def barcode(value):
    """Trim outer whitespace only. Never pad, cast floats, strip zeros or .0."""
    if value is None or value == "":
        return None, "missing"
    if not isinstance(value, str):
        return None, "invalid_non_string"
    text = value.strip()
    if not text:
        return None, "missing"
    if not re.fullmatch(r"[0-9]+", text):
        return None, "invalid_non_digit"
    return text, "valid"


def inventory(root, extra_roots=()):
    """Inventory only plausible source files, excluding environments/model caches."""
    roots = [Path(root) / "data", Path(root) / "outputs", *map(Path, extra_roots)]
    found = []
    for folder in roots:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or any(piece in {"part2_cache", ".venv", ".venv-1", "site-packages"} for piece in path.parts):
                continue
            if not path.name.endswith((".csv", ".tsv", ".csv.gz", ".tsv.gz", ".parquet")):
                continue
            sidecar = path.with_suffix(path.suffix + ".source.txt")
            provenance = sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""
            found.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                          "known_sample": path.name in SAMPLE_NAMES or _sample_provenance(provenance),
                          "provenance": provenance or "not established; a file name/size does not prove completeness"})
    return found


def validate_full_source(path, *, confirmed=False):
    if path is None:
        return {"available": False, "reason": "Complete local Part 1 export not configured. Only sampled extracts were identified; overlap is NOT calculated."}
    path = Path(path).resolve()
    if not path.is_file():
        return {"available": False, "reason": f"Configured complete source is absent: {path}"}
    sidecar = path.with_suffix(path.suffix + ".source.txt")
    if path.name in SAMPLE_NAMES or (sidecar.exists() and _sample_provenance(sidecar.read_text(encoding="utf-8"))):
        raise ValueError("Known 50,000-row sample refused. A complete source is mandatory.")
    if not confirmed:
        return {"available": False, "reason": "Confirm that this is a complete export using its acquisition provenance, not its name or size."}
    return {"available": True, "path": str(path), "bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns, "completeness": "explicit source-provenance confirmation; full EOF scan still required"}


def _json_request(url):
    last_error = None
    for _ in range(3):
        try:
            request = Request(url, headers={"User-Agent": "OFF-local-dataset-integration/1.0 (metadata-only research)"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429:
                raise RuntimeError(f"Hugging Face rate limit (429); cached pages retained. Retry later, observing Retry-After={error.headers.get('Retry-After', 'not supplied')}. No partial denominator certified.") from error
            last_error = error
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Metadata request failed; no partial inventory will be certified: {url}") from last_error


def _schemas():
    import pyarrow as pa
    images = pa.schema([
        ("image_key", pa.string()), ("barcode", pa.string()), ("barcode_raw", pa.string()),
        ("barcode_status", pa.string()), ("image_id", pa.string()), ("image_url", pa.string()),
        ("dataset_image_url", pa.string()), ("ocr_url", pa.string()), ("dataset_split", pa.string()),
        ("meta_split", pa.string()), ("source_row", pa.int64()), ("image_width", pa.int64()),
        ("image_height", pa.int64()), ("annotation_available", pa.bool_()),
        ("annotation_status", pa.string()), ("token_count", pa.int64()), ("bbox_count", pa.int64()),
        ("tag_count", pa.int64()), ("meta_json", pa.string()),
    ])
    annotations = pa.schema([
        ("image_key", pa.string()), ("barcode", pa.string()), ("image_id", pa.string()),
        ("dataset_split", pa.string()), ("token_index", pa.int64()), ("ocr_token", pa.string()),
        ("bounding_box", pa.list_(pa.int64())), ("gold_bio_id", pa.int64()), ("gold_bio_label", pa.string()),
    ])
    return images, annotations


def flatten_row(row, split, source_index, labels):
    """Retain all rows and annotation values, flag alignment issues without repairs."""
    meta, image = row.get("meta") or {}, row.get("image") or {}
    raw = meta.get("barcode")
    code, code_status = barcode(raw)
    tokens, boxes, tags = (row.get(name) for name in ("tokens", "bboxes", "ner_tags"))
    tokens, boxes, tags = tokens or [], boxes or [], tags or []
    valid_tags = all(type(tag) is int and 0 <= tag < len(labels) for tag in tags)
    valid_boxes = all(isinstance(box, list) and len(box) == 4 and all(type(value) is int for value in box) for box in boxes)
    aligned = len(tokens) == len(boxes) == len(tags)
    status = "available_aligned" if tokens and aligned and valid_tags and valid_boxes else "empty" if not tokens and not boxes and not tags else "inconsistent"
    key = f"{split}:{source_index}"
    image_id = str(meta["image_id"]) if meta.get("image_id") is not None else None
    result = dict(image_key=key, barcode=code, barcode_raw=None if raw is None else str(raw), barcode_status=code_status,
                  image_id=image_id, image_url=meta.get("image_url"), dataset_image_url=image.get("src"),
                  ocr_url=meta.get("ocr_url"), dataset_split=split, meta_split=meta.get("split"),
                  source_row=source_index, image_width=image.get("width"), image_height=image.get("height"),
                  annotation_available=status == "available_aligned", annotation_status=status,
                  token_count=len(tokens), bbox_count=len(boxes), tag_count=len(tags), meta_json=json.dumps(meta, ensure_ascii=False))
    annotations = []
    for index in range(max(len(tokens), len(boxes), len(tags))):
        tag = tags[index] if index < len(tags) else None
        annotations.append(dict(image_key=key, barcode=code, image_id=image_id, dataset_split=split,
                                token_index=index, ocr_token=tokens[index] if index < len(tokens) else None,
                                bounding_box=boxes[index] if index < len(boxes) else None, gold_bio_id=tag,
                                gold_bio_label=labels[tag] if type(tag) is int and 0 <= tag < len(labels) else None))
    return result, annotations


def load_all_part2(cache, *, allow_metadata_requests=True):
    """Page train AND test; no image requests. Cache resumable raw JSON pages."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    def request(url, destination):
        if destination.exists():
            return json.loads(destination.read_text(encoding="utf-8"))
        if not allow_metadata_requests:
            raise RuntimeError(f"Metadata not cached: {destination}. Enable bounded metadata requests; no image or model download is involved.")
        payload = _json_request(url)
        tmp = destination.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(destination)
        return payload
    metadata = request(f"https://huggingface.co/api/datasets/{DATASET}", cache / "repository.json")
    info = metadata["cardData"]["dataset_info"]
    counts = {item["name"]: int(item["num_examples"]) for item in info["splits"]}
    if not {"train", "test"}.issubset(counts):
        raise ValueError("Both official train and test splits must be declared.")
    feature = next(item for item in info["features"] if item["name"] == "ner_tags")
    names = feature["sequence"]["class_label"]["names"]
    labels = [str(names[str(index)]) for index in range(len(names))] if isinstance(names, dict) else list(names)
    revision = metadata["sha"]
    images_schema, annotation_schema = _schemas()
    images_path, annotations_path = cache / "part2_images.parquet", cache / "part2_annotations.parquet"
    manifest_path = cache / "manifest.json"
    if manifest_path.exists() and images_path.exists() and annotations_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["revision"] == revision and manifest["split_rows"] == {split: counts[split] for split in ("train", "test")}:
            files = {"images_sha256": images_path, "annotations_sha256": annotations_path}
            if all(manifest.get(key) == hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()):
                return manifest
    # Manifest is the commit marker: a failure must never certify partial parquet.
    manifest_path.unlink(missing_ok=True)
    all_images, actual_counts, token_rows = [], {}, 0
    with pq.ParquetWriter(annotations_path.with_suffix(".tmp"), annotation_schema, compression="zstd") as writer:
        for split in ("train", "test"):
            actual_counts[split] = 0
            for offset in range(0, counts[split], 100):
                params = urlencode(dict(dataset=DATASET, config="default", split=split, offset=offset, length=min(100, counts[split]-offset)))
                payload = request(f"https://datasets-server.huggingface.co/rows?{params}", cache / f"{revision}_{split}_{offset}.json")
                if payload.get("dataset_git_revision", revision) != revision or int(payload["num_rows_total"]) != counts[split]:
                    raise ValueError("Dataset revision/count changed. Do not mix metadata snapshots; use a fresh cache directory.")
                batch = payload["rows"]
                if len(batch) != min(100, counts[split]-offset):
                    raise ValueError("Incomplete Part 2 page; refusing partial overlap denominator.")
                annotations = []
                for local_index, item in enumerate(batch):
                    source_index = item.get("row_idx", offset+local_index)
                    if source_index != offset+local_index or item.get("truncated_cells"):
                        raise ValueError("Missing/out-of-order/truncated Part 2 row; no annotations may be silently dropped.")
                    inventory_row, token_batch = flatten_row(item["row"], split, source_index, labels)
                    all_images.append(inventory_row)
                    annotations.extend(token_batch)
                writer.write_table(pa.Table.from_pylist(annotations, schema=annotation_schema))
                token_rows += len(annotations)
                actual_counts[split] += len(batch)
    pq.write_table(pa.Table.from_pylist(all_images, schema=images_schema), images_path, compression="zstd")
    annotations_path.with_suffix(".tmp").replace(annotations_path)
    manifest = dict(dataset=DATASET, revision=revision, split_rows=actual_counts, images=len(all_images),
                    annotations=token_rows, labels=labels, fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                    images_path=str(images_path.resolve()), annotations_path=str(annotations_path.resolve()),
                    image_payloads_downloaded=0)
    manifest.update(images_sha256=hashlib.sha256(images_path.read_bytes()).hexdigest(),
                    annotations_sha256=hashlib.sha256(annotations_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def part2_inventory(manifest):
    import pyarrow.parquet as pq
    rows = pq.read_table(manifest["images_path"]).to_pylist()
    counts = Counter(row["barcode"] for row in rows if row["barcode"] is not None)
    return rows, dict(total_images=len(rows), unique_valid_barcodes=len(counts),
                      repeated_barcode_images=sum(count-1 for count in counts.values()),
                      barcodes_with_multiple_images=sum(count>1 for count in counts.values()),
                      missing_or_invalid_barcodes=sum(row["barcode"] is None for row in rows),
                      annotation_gaps=sum(not row["annotation_available"] for row in rows),
                      split_metadata_disagreements=sum(bool(row["meta_split"]) and row["meta_split"] != row["dataset_split"] for row in rows),
                      duplicate_barcode_image_id_groups=sum(count>1 for count in Counter((row["barcode"], row["image_id"]) for row in rows).values()))


def _source_batches(path, chunk_rows):
    import pandas as pd
    import pyarrow.parquet as pq
    path = Path(path)
    if path.suffix == ".parquet":
        source = pq.ParquetFile(path)
        code_field = source.schema_arrow.field("code")
        if str(code_field.type) not in {"string", "large_string"}:
            raise ValueError("Parquet code must already be a string; numeric conversion may have lost leading zeros.")
        for batch in source.iter_batches(batch_size=chunk_rows):
            yield batch.to_pandas()
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline()
        delimiter = "\t" if "\t" in header else ","
        yield from pd.read_csv(path, sep=delimiter, compression="infer", dtype=str, keep_default_na=False,
                               chunksize=chunk_rows, on_bad_lines="error", encoding="utf-8-sig")


def scan_complete_part1(source, target_codes, work_dir, *, confirmed=False, chunk_rows=10000, minimum_rows=50001):
    """One EOF scan, SQLite exact counts. Retain product metadata only for targets."""
    import pandas as pd
    state = validate_full_source(source, confirmed=confirmed)
    if not state["available"]:
        raise FileNotFoundError(state["reason"])
    source = Path(state["path"])
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "full_scan.sqlite"
    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript("PRAGMA cache_size=-32768; PRAGMA temp_store=FILE; CREATE TABLE codes (barcode TEXT PRIMARY KEY, occurrences INTEGER NOT NULL); CREATE TABLE matched (barcode TEXT, source_row INTEGER, payload TEXT);")
    row_count, id_status = 0, Counter()
    target_codes = set(target_codes)
    try:
        for chunk in _source_batches(source, chunk_rows):
            if not {"code", "product_name"}.issubset(chunk.columns):
                raise ValueError("Part 1 schema must contain code and product_name, as required by the existing loader.")
            normalized = [barcode(value) for value in chunk["code"]]
            id_status.update(status for _, status in normalized)
            occurrences = Counter(code for code, _ in normalized if code is not None)
            db.executemany("INSERT INTO codes VALUES (?,?) ON CONFLICT(barcode) DO UPDATE SET occurrences=occurrences+excluded.occurrences", occurrences.items())
            for position, (code, _) in enumerate(normalized):
                if code not in target_codes:
                    continue
                raw = chunk.iloc[position].to_dict()
                raw = {name: None if pd.isna(value) else str(value) for name, value in raw.items()}
                db.execute("INSERT INTO matched VALUES (?,?,?)", (code, row_count+position, json.dumps(raw, ensure_ascii=False, sort_keys=True)))
            row_count += len(chunk)
            db.commit()
            if row_count % 250000 == 0:
                print(f"Full Part 1 scan: {row_count:,} rows processed (not a sampling limit)", flush=True)
        if row_count < minimum_rows:
            raise ValueError(f"Only {row_count} rows found; source is suspiciously small. Full-data scan not certified.")
        if source.stat().st_size != state["bytes"] or source.stat().st_mtime_ns != state["modified_ns"]:
            raise ValueError("Source changed during scan; refusing inconsistent overlap.")
        unique, duplicates, repeated_codes = db.execute("SELECT count(*), coalesce(sum(occurrences-1),0), coalesce(sum(occurrences>1),0) FROM codes").fetchone()
        # All overlap/duplicate counts use the disk-backed full barcode index.
        matches = {row[0] for row in db.execute("SELECT DISTINCT barcode FROM matched")}
        return dict(status="complete_eof_scan", source=state, rows=row_count, unique_barcodes=unique,
                    duplicate_barcode_rows=duplicates, duplicate_barcode_groups=repeated_codes,
                    barcode_status_counts=dict(id_status), matched_barcodes=sorted(matches),
                    unmatched_barcodes=sorted(target_codes-matches), database=str(db_path.resolve()))
    finally:
        db.close()


def overlap_summary(scan, images, p2_summary):
    if scan.get("status") != "complete_eof_scan":
        raise ValueError("Full source EOF scan is required before overlap or exports.")
    matches = set(scan["matched_barcodes"])
    numerator, denominator = len(matches), p2_summary["unique_valid_barcodes"]
    return {"status": "measured_complete_part1_snapshot", "part1_product_rows": scan["rows"],
            "part1_unique_barcodes": scan["unique_barcodes"], "part1_duplicate_rows": scan["duplicate_barcode_rows"],
            "part1_duplicate_barcode_groups": scan["duplicate_barcode_groups"],
            "part2_images": len(images), "part2_unique_barcodes": denominator,
            "part2_repeated_barcode_images": p2_summary["repeated_barcode_images"],
            "overlapping_barcodes": numerator, "overlap_percent_of_part2": 100*numerator/denominator if denominator else None,
            "overlap_percent_of_part1": 100*numerator/scan["unique_barcodes"] if scan["unique_barcodes"] else None,
            "matched_images": sum(row["barcode"] in matches for row in images),
            "unmatched_barcodes": len(scan["unmatched_barcodes"]),
            "invalid_part2_barcode_images": p2_summary["missing_or_invalid_barcodes"]}


def image_reference_status(left_url, right_url, image_id):
    """Metadata equality is not pixel or packaging verification."""
    if not left_url or not right_url:
        return "not_comparable"
    def canonical(url):
        parts = urlsplit(url)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
    if left_url == right_url:
        return "exact_url_metadata_match_not_pixel_verified"
    if canonical(left_url) == canonical(right_url):
        return "url_path_metadata_match_not_pixel_verified"
    if image_id and Path(urlsplit(right_url).path).name.split(".")[0] == image_id:
        return "image_id_filename_match_only_not_pixel_verified"
    return "different_image_references"


def export_linked(scan, images, manifest, output_dir, summary, *, write_outputs=True):
    """Called only AFTER displaying measured overlap; preserve all Part 2 rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if scan.get("status") != "complete_eof_scan" or summary.get("status") != "measured_complete_part1_snapshot":
        raise ValueError("Confirm measured full-source overlap before constructing linked outputs.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(scan["database"])
    targets = sorted({row["barcode"] for row in images if row["barcode"] is not None})
    products = []
    for code in targets:
        candidates = db.execute("SELECT source_row,payload FROM matched WHERE barcode=? ORDER BY source_row", (code,)).fetchall()
        first = json.loads(candidates[0][1]) if candidates else {}
        nutrients = {name: value for name, value in first.items() if re.search(r"_(100g|100ml|serving)$", name) and value not in (None, "")}
        products.append(dict(barcode=code, **{name: first.get(source_name) or None for name, source_name in PRODUCT_FIELDS.items()},
                             structured_nutrients=json.dumps(nutrients, ensure_ascii=False, sort_keys=True),
                             match_status="barcode_matched_image_unverified" if candidates else "barcode_unmatched",
                             source_record_count=len(candidates), source_row=candidates[0][0] if candidates else None,
                             conflicting_product_records=len({payload for _, payload in candidates})>1))
    db.close()
    by_code = {row["barcode"]: row for row in products}
    integrated_images = []
    for image in images:
        product = by_code.get(image["barcode"])
        status = product["match_status"] if product else "invalid_or_missing_barcode"
        integrated_images.append({**image, "match_status": status,
                                  "image_reference_status": image_reference_status(image["image_url"], product["nutrition_image_url"] if product else None, image["image_id"]),
                                  "image_pixels_verified": False})
    product_schema = pa.schema([(name, pa.string()) for name in ["barcode", *PRODUCT_FIELDS, "structured_nutrients", "match_status"]] + [("source_record_count", pa.int64()), ("source_row", pa.int64()), ("conflicting_product_records", pa.bool_())])
    image_schema = _schemas()[0].append(pa.field("match_status", pa.string())).append(pa.field("image_reference_status", pa.string())).append(pa.field("image_pixels_verified", pa.bool_()))
    # Referential validation is mandatory before final outputs are written.
    assert len(products) == len(by_code) == len(targets)
    assert len(integrated_images) == manifest["images"] == len(images)
    assert len({row["image_key"] for row in images}) == len(images)
    image_keys = {row["image_key"] for row in images}
    image_by_key = {row["image_key"]: row for row in images}
    seen_token_keys = set()
    annotation_count = 0
    for batch in pq.ParquetFile(manifest["annotations_path"]).iter_batches(batch_size=10000):
        for row in batch.to_pylist():
            assert row["image_key"] in image_keys
            parent = image_by_key[row["image_key"]]
            assert (row["barcode"], row["image_id"], row["dataset_split"]) == (parent["barcode"], parent["image_id"], parent["dataset_split"])
            token_key = (row["image_key"], row["token_index"])
            assert token_key not in seen_token_keys
            seen_token_keys.add(token_key)
        annotation_count += batch.num_rows
    assert annotation_count == manifest["annotations"]
    matched_products = [row for row in products if row["source_record_count"]]
    coverage = {name: sum(row.get(name) not in (None, "", "{}") for row in matched_products)/len(matched_products) if matched_products else None for name in [*PRODUCT_FIELDS, "structured_nutrients"]}
    nutrient_keys = sorted({key for row in matched_products for key in json.loads(row["structured_nutrients"])})
    nutrient_coverage = {key: sum(key in json.loads(row["structured_nutrients"]) for row in matched_products)/len(matched_products) for key in nutrient_keys}
    quality = dict(matched_product_count=len(matched_products), unmatched_valid_product_count=len(products)-len(matched_products),
                   coverage_denominator="unique barcode-matched products; null coverage if zero matched products",
                   metadata_coverage=coverage, nutrient_field_coverage=nutrient_coverage,
                   image_reference_counts=dict(Counter(row["image_reference_status"] for row in integrated_images)),
                   conflicting_matched_products=sum(row["conflicting_product_records"] for row in products),
                   missing_image_ids=sum(not row["image_id"] for row in images), missing_dimensions=sum(row["image_width"] is None or row["image_height"] is None for row in images),
                   part1_missing_or_invalid_ids=scan["barcode_status_counts"],
                   products_scope="one row per valid Part2 barcode, including unmatched placeholders; not a duplicate full Part1 export",
                   duplicate_resolution="first source row by position; counts and conflicting payloads flagged, never merge nutrient versions",
                   warning="Barcode/URL matches do not verify image pixels, packaging version or nutrient ground truth.")
    if write_outputs:
        for table, filename in [(pa.Table.from_pylist(products, schema=product_schema), "integrated_products.parquet"),
                                (pa.Table.from_pylist(integrated_images, schema=image_schema), "integrated_images.parquet")]:
            pq.write_table(table, output / filename, compression="zstd")
        # Copy the validated token parquet byte-for-byte; gold arrays untouched.
        import shutil
        shutil.copyfile(manifest["annotations_path"], output / "integrated_annotations.parquet")
        with (output / "dataset_overlap_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value", "scope"])
            writer.writeheader()
            writer.writerows(dict(metric=key, value=value, scope="complete Part1 EOF scan; both Part2 splits") for key, value in summary.items())
        (output / "unmatched_barcodes.json").write_text(json.dumps(scan["unmatched_barcodes"], indent=2), encoding="utf-8")
        (output / "integration_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        (output / "provenance.json").write_text(json.dumps(dict(part1=scan, part2=manifest, summary=summary, quality=quality), indent=2), encoding="utf-8")
    return products, integrated_images, quality