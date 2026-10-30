"""Pre-inference-only evidence audit. Never reads model predictions or runs OCR/models."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

VERSION = '1.0'
DATASET = 'openfoodfacts/nutrient-detection-layout'
QUALITY_RULES = {'short_side_review_px': 400, 'dark_mean_gray': 60, 'bright_mean_gray': 220,
                 'low_gray_std': 30, 'low_laplacian_variance': 100, 'laplacian_measure_max_side': 1000,
                 'high_box_iou': 0.8, 'large_margin_ocr_envelope_fraction': 0.4}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def labels(value):
    if value is None or pd.isna(value):
        return []
    return sorted({part.strip().casefold() for part in str(value).split(',') if part.strip().casefold() not in {'', 'null', 'undefined', 'none', 'nan'}})


def box_diagnostics(boxes):
    """Original normalized geometry only; no clipping/deduplication/correction."""
    invalid, out_of_range, zero_area, reversed_count = [], [], [], []
    valid = []
    for index, box in enumerate(boxes):
        if box is None or len(box) != 4 or any(value is None or not np.isfinite(value) or int(value) != value for value in box):
            invalid.append(index)
            continue
        x0, y0, x1, y1 = map(int, box)
        if any(value < 0 or value > 1000 for value in box):
            out_of_range.append(index)
        if x1 < x0 or y1 < y0:
            reversed_count.append(index)
        if x1 == x0 or y1 == y0:
            zero_area.append(index)
        if 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000:
            valid.append((index, x0, y0, x1, y1))
    duplicates = sum(count-1 for count in Counter(tuple(box) for box in boxes if box is not None).values())
    overlap_pairs = high_overlap = 0
    # Per-image pairwise comparisons; bounded image sequence sizes, no dataset-wide N².
    for i, left in enumerate(valid):
        for right in valid[i+1:]:
            intersection = max(0, min(left[3], right[3])-max(left[1], right[1])) * max(0, min(left[4], right[4])-max(left[2], right[2]))
            if intersection:
                overlap_pairs += 1
                union = (left[3]-left[1])*(left[4]-left[2]) + (right[3]-right[1])*(right[4]-right[2]) - intersection
                high_overlap += int(intersection/union >= QUALITY_RULES['high_box_iou'])
    envelope = (max(row[3] for row in valid)-min(row[1] for row in valid))*(max(row[4] for row in valid)-min(row[2] for row in valid))/1e6 if valid else None
    # Backward y jumps are reading-order review proxies, not proof of wrong order in columns.
    backwards = sum(right[2] < left[2]-30 for left, right in zip(valid, valid[1:]))
    return dict(invalid_box_count=len(invalid), out_of_range_box_count=len(out_of_range), zero_area_boxes=len(zero_area),
                reversed_boxes=len(reversed_count), duplicate_box_excess=duplicates, overlapping_pairs=overlap_pairs,
                high_iou_pairs=high_overlap, ocr_envelope_area_fraction=envelope, backward_y_jumps=backwards,
                invalid_box_indices=json.dumps(sorted(set(invalid+out_of_range+zero_area+reversed_count))))


def token_diagnostics(tokens):
    decimal_comma = decimal_point = merged_unit = multiple_numeric = malformed = long = 0
    examples = []
    for index, token in enumerate(tokens):
        text = str(token or '')
        issues = []
        decimal_comma += bool(re.search(r'\d,\d', text))
        decimal_point += bool(re.search(r'\d\.\d', text))
        if re.search(r'\d\s*(?:kcal|kj|mg|[µμu]g|g|ml)\b', text, flags=re.I):
            merged_unit += 1
            issues.append('number_unit_in_same_OCR_token')
        if len(re.findall(r'\d+(?:[,.]\d+)?', text)) > 1:
            multiple_numeric += 1
            issues.append('multiple_numeric_mentions_segmentation_review')
        if re.search(r'(?:\d[,.]){2,}\d|[Oo][gG]\b|\d+[,.]\d+[,.]|\d\s*(?:pg|kcat)\b', text):
            malformed += 1
            issues.append('numeric_glyph_or_separator_review')
        if len(text) > 30:
            long += 1
            issues.append('long_merged_token_review')
        if issues:
            examples.append(dict(token_index=index, ocr_token=text, issue='|'.join(issues)))
    return dict(decimal_comma_tokens=decimal_comma, decimal_point_tokens=decimal_point,
                merged_number_unit_tokens=merged_unit, multiple_numeric_tokens=multiple_numeric,
                malformed_numeric_review_tokens=malformed, long_tokens=long,
                empty_tokens=sum(not str(value or '').strip() for value in tokens),
                mean_token_characters=float(np.mean([len(str(value or '')) for value in tokens])) if tokens else 0,
                max_token_characters=max((len(str(value or '')) for value in tokens), default=0)), examples


def bio_diagnostics(gold_labels, allowed):
    """Do not repair invalid I- transitions; preserve raw entities as annotated."""
    previous, entities, invalid, multi = 'O', [], [], 0
    active = None
    for index, label in enumerate([*gold_labels, 'O']):
        prefix, _, kind = str(label).partition('-')
        if index < len(gold_labels) and label not in allowed:
            invalid.append(dict(token_index=index, label=str(label), issue='unknown_gold_label'))
        if prefix == 'I' and previous not in {f'B-{kind}', f'I-{kind}'}:
            invalid.append(dict(token_index=index, label=str(label), issue='I_without_same_entity_predecessor'))
        if active is not None and (prefix != 'I' or kind != active['label']):
            active['end'] = index
            multi += int(index-active['start'] > 1)
            entities.append(active)
            active = None
        if prefix in {'B', 'I'} and active is None:
            active = dict(label=kind, start=index)
        previous = label
    return dict(entity_count=len(entities), multi_token_entities=multi, invalid_bio_transitions=len(invalid),
                per_100g_entities=sum(row['label'].endswith('_100G') for row in entities),
                per_serving_entities=sum(row['label'].endswith('_SERVING') for row in entities),
                serving_size_entities=sum(row['label'] == 'SERVING_SIZE' for row in entities)), entities, invalid


def local_image_path(root, record):
    stable_url = record.get('image_url') or record.get('dataset_image_url')
    text = f"{DATASET}|{record['barcode']}|{record['image_id']}|{stable_url}"
    path = Path(root) / 'data/part2_cache/images' / (hashlib.sha256(text.encode('utf-8')).hexdigest()+'.jpg')
    return path if path.is_file() else None


def image_measurements(path):
    """Luminance and fixed-scale Laplacian proxies; not calibrated OCR quality."""
    import cv2
    with Image.open(path) as source:
        raw_size = source.size
        orientation = int(source.getexif().get(274, 1))
        image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = image.size
        original_mode = source.mode
    gray = ImageOps.grayscale(image)
    scaled = gray.copy()
    scaled.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    pixels = np.asarray(scaled, dtype=np.uint8)
    return dict(local_width=width, local_height=height, raw_width=raw_size[0], raw_height=raw_size[1],
                exif_orientation=orientation, source_mode=original_mode,
                brightness_mean=float(pixels.mean()), contrast_std=float(pixels.std()),
                laplacian_variance=float(cv2.Laplacian(pixels, cv2.CV_64F).var()),
                near_black_fraction=float((pixels < 5).mean()), near_white_fraction=float((pixels > 250).mean()),
                measure_width=scaled.width, measure_height=scaled.height)


def image_flags(measurement):
    flags = []
    if measurement.get('exif_orientation', 1) not in {0, 1}:
        flags.append('exif_orientation_transform_needed')
    if measurement.get('brightness_mean', 128) < 60:
        flags.append('dark_luminance_review')
    if measurement.get('brightness_mean', 128) > 220:
        flags.append('bright_luminance_review')
    if measurement.get('contrast_std', 100) < 30:
        flags.append('low_contrast_proxy')
    if measurement.get('laplacian_variance', 1000) < 100:
        flags.append('low_sharpness_proxy')
    return flags


def tokenizer_lengths(tokenizer, tokens, boxes):
    """Local tokenizer only: no processor weights, pixels, model or inference."""
    encoded = tokenizer(text=tokens, boxes=boxes, truncation=False, padding=False, add_special_tokens=True)
    ids = encoded['input_ids']
    word_ids = encoded.word_ids()
    # Conservative single-pass coverage of first 512 positions, not an actual forward pass.
    # Reserve both BOS and EOS: a 512-position LayoutLM text input has at most
    # 510 text subwords. Audit word-start coverage, not complete last-word pieces.
    covered = {word for word in word_ids[1:511] if word is not None}
    max_word_pieces = max(Counter(word for word in word_ids if word is not None).values(), default=0)
    return dict(subword_tokens_with_specials=len(ids), needs_sliding_window=len(ids) > 512,
                first_512_word_coverage=len(covered), words_missed_by_single_pass=len(tokens)-len(covered),
                max_subwords_per_word=max_word_pieces)


def run(root, output=None):
    root = Path(root).resolve()
    for path in [root, root / 'src']:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from transformers import LayoutLMv3TokenizerFast
    from nutrition_extraction.linked_data import load_linked
    from part2.config import PreprocessingConfig
    from part2.preprocessing import load_image, preprocess_image
    output = Path(output) if output else root / 'outputs/preprocessing_readiness'
    output.mkdir(parents=True, exist_ok=True)
    directory = root / 'outputs/part1_part2_integration'
    products, images, integration = load_linked(directory)
    annotations = pd.read_parquet(directory / 'integrated_annotations.parquet')
    label_map = json.loads((root / 'part2/weights/config.json').read_text(encoding='utf-8-sig'))['id2label']
    allowed = set(label_map.values())
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(str(root / 'part2/weights'), local_files_only=True, only_label_first_subword=True)
    tokenizer.model_max_length = 1000000  # Audit untruncated length; the deployed model limit remains 512.
    protected = [*directory.glob('*.parquet'), root / 'part2/preprocessing.py', root / 'part2/config.py',
                 *[p for p in root.rglob('*.ipynb') if p.name != 'part2_preprocessing_eda.ipynb' and '.venv' not in str(p)]]
    before = {str(path): sha(path) for path in protected}
    by_product = products.set_index('barcode').to_dict('index')
    by_tokens = {key: group.sort_values('token_index') for key, group in annotations.groupby('image_key', sort=False)}
    rows, token_examples, transition_examples, entity_rows, label_rows, image_errors = [], [], [], [], [], []
    for record in images.to_dict('records'):
        key = record['image_key']
        group = by_tokens.get(key, annotations.iloc[:0])
        tokens = group['ocr_token'].tolist()
        boxes = group['bounding_box'].tolist()
        gold = group['gold_bio_label'].tolist()
        product = by_product.get(str(record['barcode']), {})
        base = dict(image_key=key, barcode=record['barcode'], image_id=record['image_id'], split=record['dataset_split'],
                    match_status=record['match_status'], country=product.get('country'), category=product.get('category'),
                    image_url=record['image_url'], width=record['image_width'], height=record['image_height'],
                    aspect_ratio=record['image_width']/record['image_height'] if record['image_height'] else None,
                    token_count=len(tokens), annotation_available=record['annotation_available'],
                    token_box_alignment=bool(len(tokens)==record['token_count']==record['bbox_count']==record['tag_count'] and not group['ocr_token'].isna().any() and not group['gold_bio_label'].isna().any()),
                    token_index_contiguous=group['token_index'].tolist() == list(range(len(group))),
                    label_ids_consistent=all(str(label_map.get(str(tag))) == label for tag, label in zip(group['gold_bio_id'], gold)))
        geometry = box_diagnostics(boxes)
        token_stats, examples = token_diagnostics(tokens)
        bio_stats, entities, transitions = bio_diagnostics(gold, allowed)
        base.update(geometry, **token_stats, **bio_stats)
        base['orientation_metadata'] = 'landscape' if base['aspect_ratio'] and base['aspect_ratio'] > 1.5 else 'portrait' if base['aspect_ratio'] and base['aspect_ratio'] < .75 else 'near_square'
        base['short_side_review'] = bool(min(base['width'], base['height']) < 400)
        base['large_margin_crop_review'] = bool(geometry['ocr_envelope_area_fraction'] is not None and geometry['ocr_envelope_area_fraction'] < .4)
        base['nutrients'] = '|'.join(sorted({re.sub(r'_(100G|SERVING)$', '', entity['label']).lower() for entity in entities}))
        # Fast tokenization does not use spatial values to split text. Preserve raw
        # boxes for this audit even when their geometry requires model-input review.
        valid_tokenizer_input = base['token_box_alignment'] and not geometry['invalid_box_count']
        if valid_tokenizer_input:
            base.update(tokenizer_lengths(tokenizer, tokens, [list(box) for box in boxes]))
            base['tokenizer_status'] = 'measured_local_tokenizer'
        else:
            base.update(subword_tokens_with_specials=None, needs_sliding_window=None, first_512_word_coverage=None, words_missed_by_single_pass=None, max_subwords_per_word=None)
            base['tokenizer_status'] = 'review_invalid_alignment_or_geometry_not_silently_repaired'
        path = local_image_path(root, record)
        base['local_image_path'] = str(path) if path else None
        base['pixel_status'] = 'not_cached_not_measured'
        base['image_hash'] = None
        flags = ['short_side_lt400_review'] if base['short_side_review'] else []
        if path:
            try:
                base['image_hash'] = sha(path)
                base.update(image_measurements(path))
                base['pixel_status'] = 'measured_cached_original'
                flags.extend(image_flags(base))
                base['dimension_metadata_mismatch'] = (base['raw_width'], base['raw_height']) != (base['width'], base['height'])
            except Exception as error:
                base['pixel_status'] = 'cached_unreadable'
                image_errors.append(dict(image_key=key, error=f'{type(error).__name__}: {error}'))
        if base['large_margin_crop_review']:
            flags.append('ocr_envelope_small_crop_review_not_table_localization')
        base['quality_flags'] = '|'.join(flags)
        rows.append(base)
        token_examples.extend({**{name: base[name] for name in ['image_key', 'barcode', 'split']}, **example} for example in examples)
        transition_examples.extend(dict(image_key=key, barcode=base['barcode'], split=base['split'], **example) for example in transitions)
        entity_rows.extend(dict(image_key=key, barcode=base['barcode'], split=base['split'], **entity) for entity in entities)
        label_rows.extend(dict(image_key=key, split=base['split'], label=label, count=count) for label, count in Counter(gold).items())
    image_table = pd.DataFrame(rows)
    # Product readiness uses the same original values, no annotation/model prediction input.
    metadata_rows = []
    mappings = {'fat': ('fat_100g','g'), 'saturated_fat': ('saturated-fat_100g','g'), 'carbohydrates': ('carbohydrates_100g','g'),
                'sugars': ('sugars_100g','g'), 'proteins': ('proteins_100g','g'), 'salt': ('salt_100g','g'),
                'sodium': ('sodium_100g','g'), 'fiber': ('fiber_100g','g'), 'energy_kcal': ('energy-kcal_100g','kcal'),
                'energy_kj': ('energy-kj_100g','kj'), 'calcium': ('calcium_100g','g'), 'iron': ('iron_100g','g'),
                'potassium': ('potassium_100g','g'), 'vitamin_d': ('vitamin-d_100g','g')}
    for product in products.to_dict('records'):
        raw_nutrients = json.loads(product['structured_nutrients'])
        serving = str(product.get('serving_size') or '').strip()
        for nutrient, (field, unit) in mappings.items():
            value = pd.to_numeric(pd.Series([raw_nutrients.get(field)]), errors='coerce').iloc[0]
            finite = pd.notna(value) and np.isfinite(value)
            flag = ''
            if finite:
                if value < 0:
                    flag = 'negative_quantity_review'
                elif value > (1000 if unit=='kcal' else 4200 if unit=='kj' else 100):
                    flag = 'unit_or_basis_implausible_quantity_review'
            metadata_rows.append(dict(barcode=product['barcode'], match_status=product['match_status'], nutrient=nutrient,
                                      source_field=field, source_unit_convention=unit, basis_convention='per_100g',
                                      raw_value=raw_nutrients.get(field), numeric_value=float(value) if finite else None,
                                      finite_numeric=bool(finite), review_flag=flag,
                                      unit_verified_from_packaging=False, serving_size=serving,
                                      serving_size_available=serving.casefold() not in {'', 'nan', 'none', 'null', 'undefined'},
                                      serving_mass_or_volume_parseable=bool(re.search(r'\d+(?:[,.]\d+)?\s*(?:g|ml)\b', serving, re.I))))
    tables = {'images': image_table, 'token_examples': pd.DataFrame(token_examples),
              'bio_transition_examples': pd.DataFrame(transition_examples, columns=['image_key','barcode','split','token_index','label','issue']),
              'entities': pd.DataFrame(entity_rows), 'label_counts': pd.DataFrame(label_rows), 'metadata': pd.DataFrame(metadata_rows)}
    # Small non-destructive preprocessing proxy experiment; do NOT call this OCR validation.
    experiments = []
    measured = image_table.loc[image_table['pixel_status'].eq('measured_cached_original')].sort_values('image_key').head(8)
    for item in measured.to_dict('records'):
        source_image = load_image(item['local_image_path'])
        # Same adapter as pipeline, on an in-memory copy, saved only to audit outputs.
        changed = preprocess_image(source_image, PreprocessingConfig(max_side=2000, contrast=1.15, grayscale=False, resize=True))
        preview = output / 'preprocessing_proxy_previews'
        preview.mkdir(exist_ok=True)
        transformed_path = preview / (item['image_key'].replace(':','_')+'_contrast_resize.png')
        changed.save(transformed_path)
        changed_metrics = image_measurements(transformed_path)
        experiments.append(dict(image_key=item['image_key'], barcode=item['barcode'],
                                transformation='existing max_side=2000 + contrast=1.15 on a COPY; no OCR run',
                                before_brightness=item['brightness_mean'], after_brightness=changed_metrics['brightness_mean'],
                                before_contrast=item['contrast_std'], after_contrast=changed_metrics['contrast_std'],
                                before_sharpness=item['laplacian_variance'], after_sharpness=changed_metrics['laplacian_variance'],
                                original_width=item['local_width'], original_height=item['local_height'],
                                transformed_width=changed.width, transformed_height=changed.height,
                                original_path=item['local_image_path'], transformed_path=str(transformed_path),
                                ocr_quality_before=None, ocr_quality_after=None,
                                conclusion='Pixel proxies only; no evidence of improved OCR accuracy. Need matched manual transcription and controlled OCR experiment before adoption.'))
    tables['proxy_experiments'] = pd.DataFrame(experiments)
    for name, table in tables.items():
        table.to_parquet(output / f'{name}.parquet', index=False)
    recommendations = build_recommendations(image_table, tables['metadata'])
    recommendations.to_csv(output / 'recommendations.csv', index=False)
    all_sources = {str(path): sha(path) for path in protected}
    if before != all_sources:
        raise ValueError('Protected input changed externally during readiness analysis.')
    for item in image_table.loc[image_table['pixel_status'].eq('measured_cached_original')].to_dict('records'):
        if sha(item['local_image_path']) != item['image_hash']:
            raise ValueError('Original cached image changed during proxy experiment.')
    manifest = dict(version=VERSION, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    total_annotated_images=len(image_table), matched_products=int(products['match_status'].eq('barcode_matched_image_unverified').sum()),
                    unmatched_images=int(image_table['match_status'].ne('barcode_matched_image_unverified').sum()),
                    split_counts=image_table.groupby('split').size().to_dict(), token_annotations=len(annotations),
                    pixel_measured_images=int(image_table['pixel_status'].eq('measured_cached_original').sum()),
                    cached_unreadable_images=int(image_table['pixel_status'].eq('cached_unreadable').sum()),
                    pixel_sampling='All locally cached original Part2 images identifiable using the existing dataset loader cache hash. Availability/convenience sample, NOT random or representative. No uncached images downloaded.',
                    measured_pixel_split_counts=image_table.loc[image_table['pixel_status'].eq('measured_cached_original')].groupby('split').size().to_dict(),
                    tokenizer_measured_images=int(image_table['tokenizer_status'].eq('measured_local_tokenizer').sum()),
                    tokenizer_assets={name: sha(root / 'part2/weights' / name) for name in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']},
                    sliding_window_images=int(image_table['needs_sliding_window'].eq(True).sum()),
                    rules=QUALITY_RULES, proxy_experiment_images=len(experiments), source_sha256=before,
                    measured_image_sha256={item['local_image_path']: item['image_hash'] for item in image_table.loc[image_table['pixel_status'].eq('measured_cached_original')].to_dict('records')},
                    image_errors=image_errors, prediction_caches_read=0, model_inference_runs=0, ocr_runs=0,
                    existing_sources_unchanged=True,
                    limitations=['Pixel indicators use cached availability sample only, not all images.',
                                 'OCR diagnostics concern supplied corpus OCR, not unseen-image EasyOCR error rates.',
                                 'Aspect ratio is not text rotation; EXIF orientation alone cannot detect sideways/upside-down text.',
                                 'OCR envelope is not a nutrition-table annotation; crop flags cannot justify automatic crops.',
                                 'Sharpness/brightness/contrast thresholds are explicit uncalibrated review screens, not learned quality labels.',
                                 'Tokenizer lengths are exact local-tokenizer text lengths with untouched boxes, not geometry approval or model forward passes; malformed box structures/alignment remain unmeasured.',
                                 'Per-100g database field units/basis are conventions, not packaging-verified truth.',
                                 'No prediction, F1, performance or post-inference output influences readiness counts.',
                                 'Contrast/resize test measures pixel proxies only; no OCR benefit has been demonstrated.'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report = ['# Pre-inference readiness: measured evidence', '', json.dumps({key:manifest[key] for key in ['total_annotated_images','matched_products','unmatched_images','token_annotations','pixel_measured_images','tokenizer_measured_images','sliding_window_images','proxy_experiment_images']}, indent=2), '']
    for recommendation in recommendations.to_dict('records'):
        percentage = f"{recommendation['affected_percent']:.2f}%" if recommendation['denominator'] else 'not measured'
        report.extend([f"## {recommendation['observed_issue']}", f"- Affected: {recommendation['affected_count']}/{recommendation['denominator']} ({percentage}) {recommendation['denominator_scope']}", f"- Class: {recommendation['decision_class']}", f"- Action: {recommendation['preprocessing_required']}", f"- Benefit: {recommendation['expected_benefit']}", f"- Validation: {recommendation['validation_method']}", ''])
    report.extend(['## Limits', *['- '+text for text in manifest['limitations']]])
    (output / 'readiness_findings.md').write_text('\n'.join(report), encoding='utf-8')
    return manifest, tables, recommendations


def build_recommendations(images, metadata):
    n = len(images)
    pixels = images.loc[images['pixel_status'].eq('measured_cached_original')]
    measured = len(pixels)
    output = []
    def add(issue, count, denominator, scope, classification, action, benefit, validation):
        output.append(dict(observed_issue=issue, affected_count=int(count), denominator=int(denominator),
                           affected_percent=100*count/denominator if denominator else None,
                           denominator_scope=scope, decision_class=classification, preprocessing_required=action,
                           expected_benefit=benefit, validation_method=validation))
    add('Input longer than 512 LayoutLMv3 subwords including specials', images['needs_sliding_window'].eq(True).sum(), images['subword_tokens_with_specials'].notna().sum(), 'tokenizer-measured annotated images', 'Required preprocessing',
        'Retain existing overlapping sliding windows and word_ids alignment; no single-pass truncation for affected images.', 'Full original-word coverage rather than silent truncation.', 'Assert every original word is covered; validate boundary spans and overlap merging without tuning gold.')
    bad = (~images['token_box_alignment']) | (~images['token_index_contiguous']) | (~images['label_ids_consistent']) | images[['invalid_box_count','out_of_range_box_count','zero_area_boxes','reversed_boxes']].gt(0).any(axis=1)
    add('Invalid/alignment-risk token-box-label records', bad.sum(), n, 'all annotated images', 'Required preprocessing',
        'Validate aligned tokens/boxes/labels and [0,1000] nondegenerate integer geometry before model encoding; quarantine failures, never silently clamp/reorder gold.', 'Prevent invalid model inputs while preserving annotations.', 'Read-only alignment, coordinate and tokenizer word_ids checks. If zero affected, retain checks; no corrective transform justified.')
    add('EXIF orientation transformation indicated', pixels['exif_orientation'].isin([2,3,4,5,6,7,8]).sum() if measured else 0, measured, 'cached pixel-measured images only', 'Required preprocessing',
        'Retain EXIF-aware image loading for new OCR inputs; if transforming already annotated images, coordinate-transform boxes consistently or preserve original geometry.', 'Avoid metadata-directed orientation mismatch.', 'Compare original/oriented dimensions and transformed box overlays; do not infer rotation from aspect ratio.')
    add('Short image side below 400 pixels', images['short_side_review'].sum(), n, 'all images with source dimensions', 'Data-quality review flag',
        'Inspect legibility; do not assume upscaling restores text. Prefer original higher-resolution source only if already available.', 'Identify likely text-detail limitations before OCR.', 'Inspect representative panels and validate OCR against manual text, not sharpness proxies alone.')
    add('Source maximum side exceeds existing 2000px OCR cap', images[['width','height']].max(axis=1).gt(2000).sum(), n, 'all images with source dimensions', 'Optional preprocessing',
        'Retain aspect-preserving maximum-side resize as a resource-control option, not an accuracy claim. Do not upscale small sources; preserve the model processor saved image settings.',
        'Bound CPU OCR workload while avoiding unnecessary upscaling or distorted aspect ratios.', 'Compare before/after legibility and manual-reference OCR quality; verify scaled geometry/normalized boxes. Existing corpus model inputs must retain image/box alignment.')
    for issue, flag, action in [('Low grayscale contrast proxy', 'low_contrast_proxy', 'Consider contrast normalization only on reviewed low-contrast inputs; default 1.15 contrast is not justified for every image.'),
                                ('Low sharpness/Laplacian proxy', 'low_sharpness_proxy', 'Review small/blurred panels; do not apply unconditional sharpening or blur correction.'),
                                ('Dark luminance proxy', 'dark_luminance_review', 'Consider brightness correction only if visual review confirms underexposure.'),
                                ('Bright luminance proxy', 'bright_luminance_review', 'Review white-background versus overexposure; avoid automatic darkening from mean brightness alone.')]:
        add(issue, pixels['quality_flags'].str.contains(flag, regex=False).sum() if measured else 0, measured, 'cached pixel-measured convenience sample only', 'Optional preprocessing', action,
            'Potential readability benefit is a hypothesis; current experiment measures proxies only.', 'Paired before/after OCR on manually transcribed held-out images; report CER/WER with identical OCR settings before adoption.')
    add('Small OCR envelope within image (<40% of normalized area)', images['large_margin_crop_review'].sum(), n, 'all annotated images with valid boxes', 'Data-quality review flag',
        'Inspect candidate margins; do not auto-crop from all-text OCR envelope. Use verified nutrition-table localization if a crop is later justified, remapping boxes.', 'Potentially reduce irrelevant packaging text without deleting nutrient context.', 'Manually verify crop retains every target span; coordinate transform/inverse tests; OCR quality comparison.')
    for issue, column, action in [
        ('Decimal-comma numeric text', 'decimal_comma_tokens', 'Retain decimal-comma normalization in downstream numeric parser; never replace the original OCR/gold text.'),
        ('Multiple numeric mentions in one OCR token', 'multiple_numeric_tokens', 'Preserve OCR token geometry and flag segmentation for review; do not split annotated gold words without an explicit reversible mapping.'),
        ('Number and unit merged in one OCR token', 'merged_number_unit_tokens', 'Use unit-aware parsing that accepts merged values/units; retain original word_ids and raw OCR.'),
        ('Malformed numeric glyph/separator review', 'malformed_numeric_review_tokens', 'Flag raw strings; do not replace O→0 or g→9 automatically. Validate any correction using image evidence.'),
        ('Backward vertical reading-order jumps', 'backward_y_jumps', 'Inspect multi-column reading order before any sorting. Preserve original corpus order for aligned annotation evaluation.'),
        ('Duplicate bounding boxes', 'duplicate_box_excess', 'Inspect shared/duplicated detection geometry; never discard gold tokens solely because boxes repeat.'),
        ('High-IoU overlapping box pairs', 'high_iou_pairs', 'Review segmentation/shared-box context; no automatic NMS or box merge on gold annotations.'),
        ('Invalid BIO continuation transitions', 'invalid_bio_transitions', 'Quarantine/adjudicate labels for review; retain original tags, never auto-repair evaluation gold.')]:
        classification = 'Required preprocessing' if column in {'decimal_comma_tokens', 'merged_number_unit_tokens'} else 'Data-quality review flag'
        add(issue, images[column].gt(0).sum(), n, 'all annotated images (at least one occurrence)', classification, action,
            'Avoid unsupported OCR/annotation changes and retain traceable input alignment.', 'Inspect token/box/image examples; record reversible proposals separately and keep official evaluation untouched.')
    matched = metadata.loc[metadata['match_status'].eq('barcode_matched_image_unverified')]
    products = matched.drop_duplicates('barcode')
    add('Missing serving-size context', (~products['serving_size_available']).sum(), len(products), 'unique barcode-matched products', 'Data-quality review flag',
        'Keep per-serving and per-100g values separate; never infer a serving mass or convert without explicit evidence.', 'Avoid ungrounded measurement-basis conversion before/after extraction.', 'Require explicit parsed serving mass/volume and verify packaging basis before normalization.')
    flagged_products = matched.loc[matched['review_flag'].ne(''), 'barcode'].nunique()
    add('Implausible/negative structured product quantities', flagged_products, len(products), 'unique barcode-matched products', 'Post-inference validation',
        'Use unit/basis-aware validation as a downstream review guard, not a reason to alter image pixels or gold labels.', 'Avoid treating unverified database anomalies as preprocessing targets or image truth.', 'Check units, basis and packaging version manually; no automatic replacement of extracted values.')
    return pd.DataFrame(output)


def filtered_images(frame, splits=(), countries=(), nutrients=(), quality_flags=()):
    mask = pd.Series(True, index=frame.index)
    if splits:
        mask &= frame['split'].isin(splits)
    if countries:
        mask &= frame['country'].map(lambda value: bool(set(labels(value)) & set(countries)))
    if nutrients:
        mask &= frame['nutrients'].map(lambda value: bool(set(str(value).split('|')) & set(nutrients)))
    if quality_flags:
        mask &= frame['quality_flags'].map(lambda value: bool(set(str(value).split('|')) & set(quality_flags)))
    return frame.loc[mask].copy()
