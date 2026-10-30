"""CPU-only checkpointed inference on the exact locally integrated matched cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def aggregate(rows):
    """Micro-average exact span counts, never average per-image F1."""
    tp = sum(r['true_positive_spans'] for r in rows)
    predicted = sum(r['predicted_spans'] for r in rows)
    gold = sum(r['gold_spans'] for r in rows)
    support = sum(r['token_support'] for r in rows)
    precision = tp / predicted if predicted else 0.
    recall = tp / gold if gold else 0.
    return dict(images=len(rows), true_positive_spans=tp, predicted_spans=predicted,
                gold_spans=gold, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                token_support=support,
                token_accuracy=sum(r['correct_tokens'] for r in rows)/support if support else None)


def run(root=ROOT, limit=None):
    import msvcrt
    import pandas as pd
    from PIL import Image
    from nutrition_extraction.linked_data import load_linked, MATCHED
    from part2.config import DatasetConfig
    from part2.dataset import HuggingFaceNutritionDataset
    from part2.layoutlm_bio import (INFERENCE_SCHEMA_VERSION, MODEL_ID, MODEL_REVISION,
                                  available_ram_bytes, evaluate_sequences,
                                  load_pretrained_model, predict_record)

    root = Path(root)
    output = root / 'outputs/full_cohort_bio'
    output.mkdir(parents=True, exist_ok=True)
    samples = output / 'samples'
    samples.mkdir(exist_ok=True)
    lock = (output / 'run.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('A full-cohort worker already holds the run lock.')
    directory = root / 'outputs/part1_part2_integration'
    products, images, _ = load_linked(directory)
    cohort = images.loc[images['barcode'].isin(products.loc[products['match_status'].eq(MATCHED), 'barcode'])].copy()
    # Finish the official test cohort first, without changing its membership.
    cohort = cohort.sort_values(['dataset_split', 'source_row'])
    if len(cohort) != 3075 or cohort['image_key'].nunique() != 3075:
        raise ValueError('Expected exactly 3,075 unique matched image records.')
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.parquet')}
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    annotations = pd.read_parquet(directory / 'integrated_annotations.parquet')
    groups = {key: group.sort_values('token_index') for key, group in annotations.groupby('image_key', sort=False)}
    completed, failures = {}, {}
    model = processor = id2label = None
    runtime = None
    started = time.monotonic()
    status = 'running'

    def publish():
        rows = list(completed.values())
        summary = dict(schema_version=1, inference_schema_version=INFERENCE_SCHEMA_VERSION,
                       model_id=MODEL_ID, model_revision=MODEL_REVISION,
                       input_fingerprint=fingerprint, source_sha256=source_hashes,
                       updated_at_utc=datetime.now(timezone.utc).isoformat(), status=status,
                       target_images=len(cohort), completed_images=len(rows),
                       failed_images=len(failures), pending_images=len(cohort)-len(rows)-len(failures),
                       target_splits=cohort.groupby('dataset_split').size().to_dict(),
                       metrics_by_split={split: aggregate([r for r in rows if r['split']==split]) for split in ['train','test']},
                       runtime=runtime, session_elapsed_seconds=time.monotonic()-started,
                       reference='Original dataset BIO annotations, not independent visual transcription.',
                       caveat='Train scores are descriptive, not held-out generalization. Test selection is all matched official test rows. Supplied OCR; no OCR run. No metadata is used as gold.')
        atomic_json(output / 'summary.json', summary)
        atomic_json(output / 'failures.json', list(failures.values()))
        temporary = output / 'image_results.tmp'
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(output / 'image_results.csv')

    try:
        for row in cohort.to_dict('records'):
            key = row['image_key']
            destination = samples / f"{key.replace(':','_')}.json"
            if destination.exists():
                cached = json.loads(destination.read_text(encoding='utf-8'))
                if cached.get('input_fingerprint') == fingerprint and cached.get('inference_schema_version') == INFERENCE_SCHEMA_VERSION:
                    completed[key] = cached['summary_row']
        publish()
        attempts = 0
        for row in cohort.to_dict('records'):
            key = row['image_key']
            if key in completed:
                continue
            if limit is not None and attempts >= limit:
                status = 'paused_limit'; break
            if available_ram_bytes() < .75 * 1024**3:
                status = 'paused_low_memory'; break
            attempts += 1
            image = None
            try:
                group = groups[key]
                if group['token_index'].tolist() != list(range(len(group))):
                    raise ValueError('Noncontiguous original token indices.')
                tokens = group['ocr_token'].tolist()
                boxes = [list(map(int, box)) for box in group['bounding_box']]
                tags = group['gold_bio_id'].astype(int).tolist()
                raw = dict(tokens=tokens, bboxes=boxes, ner_tags=tags,
                           meta=json.loads(row['meta_json']), image={'src':row['dataset_image_url']})
                loader = object.__new__(HuggingFaceNutritionDataset)
                loader.config = DatasetConfig(split=row['dataset_split'], image_cache=root/'data/part2_cache/images')
                loader.config.image_cache.mkdir(parents=True, exist_ok=True)
                loader.timeout = 45
                try:
                    image_path = loader._image_path(raw)
                except RuntimeError:
                    # Signed asset links expire: refresh this original official row,
                    # not a possibly newer OFF packaging image.
                    fresh = HuggingFaceNutritionDataset(loader.config, row_indices=[int(row['source_row'])])
                    fresh_raw = fresh.records[0]
                    if fresh_raw['tokens'] != tokens or fresh_raw['bboxes'] != boxes or fresh_raw['ner_tags'] != tags:
                        raise ValueError('Refreshed official row differs from integrated annotations.')
                    image_path = fresh._image_path(fresh_raw)
                with Image.open(image_path) as source:
                    if source.size != (int(row['image_width']), int(row['image_height'])):
                        raise ValueError('Cached source dimensions differ from annotated source dimensions.')
                    image = source.convert('RGB')
                record = dict(image=image, image_path=str(image_path), tokens=tokens, bboxes=boxes,
                              ner_tags=tags, barcode=str(row['barcode']))
                if model is None:
                    model, processor, id2label, runtime = load_pretrained_model(root/'part2/weights')
                prediction = predict_record(record, model=model, processor=processor, id2label=id2label,
                                            cache_dir=root/'data/part2_cache/layoutlm_bio')
                gold = group['gold_bio_label'].tolist()
                if prediction['gold_labels'] != gold or len(prediction['predicted_labels']) != len(tokens) or not all(prediction['coverage_mask']) or prediction['covered_token_count'] != len(tokens):
                    raise ValueError('Prediction/gold alignment or full token coverage failed.')
                metrics = evaluate_sequences([gold], [prediction['predicted_labels']], aligned=True,
                                             coverage_masks=[prediction['coverage_mask']])
                result_row = dict(image_key=key, barcode=str(row['barcode']), split=row['dataset_split'],
                                  token_support=len(tokens), correct_tokens=sum(a==b for a,b in zip(gold,prediction['predicted_labels'])),
                                  **{name:metrics[name] for name in ['true_positive_spans','predicted_spans','gold_spans','precision','recall','f1','token_accuracy']},
                                  inference_seconds=prediction['inference_seconds'], inference_windows=prediction['inference_windows'],
                                  cache_hit=prediction['cache_hit'], token_coverage=1.)
                payload = dict(input_fingerprint=fingerprint, inference_schema_version=INFERENCE_SCHEMA_VERSION,
                               image_key=key, source_row=int(row['source_row']), split=row['dataset_split'],
                               barcode=str(row['barcode']), image_path=str(image_path),
                               image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                               tokens=tokens, bboxes=boxes, prediction=prediction, summary_row=result_row)
                atomic_json(samples/f"{key.replace(':','_')}.json", payload)
                completed[key] = result_row
                print(f"{len(completed)}/3075 {key}: {prediction['inference_seconds']:.2f}s cache={prediction['cache_hit']}", flush=True)
            except Exception as error:
                failures[key] = dict(image_key=key, barcode=str(row['barcode']), split=row['dataset_split'], error=f'{type(error).__name__}: {error}')
                print(f'FAILED {key}: {error}', flush=True)
                # Resource/load failures must not cascade through 3,075 images.
                if model is None and image is not None:
                    status = 'paused_model_load_error'; break
            finally:
                if image is not None:
                    image.close()
                publish()
        if status == 'running':
            status = 'complete' if len(completed)==len(cohort) else 'incomplete_with_failures'
    except BaseException:
        status = 'interrupted'
        raise
    finally:
        publish()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
    return json.loads((output/'summary.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(limit=args.limit), indent=2))