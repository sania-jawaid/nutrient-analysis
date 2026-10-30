"""Offline Part 1 → Part 2 evidence audit. No inference, downloads or gold edits."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VERSION = '1.1'
BACKGROUND_VERSION = '1.1'
PLACEHOLDERS = {'', 'null', 'none', 'nan', '<na>', 'undefined', 'n/a'}
FIELDS = {
    'energy_kcal': ('energy-kcal_100g', 'kcal', 1000.0),
    'energy_kj': ('energy-kj_100g', 'kj', 4200.0),
    'fat': ('fat_100g', 'g', 100.0), 'saturated_fat': ('saturated-fat_100g', 'g', 100.0),
    'carbohydrates': ('carbohydrates_100g', 'g', 100.0), 'sugars': ('sugars_100g', 'g', 100.0),
    'fiber': ('fiber_100g', 'g', 100.0), 'proteins': ('proteins_100g', 'g', 100.0),
    'salt': ('salt_100g', 'g', 100.0), 'sodium': ('sodium_100g', 'g', 100.0),
    'calcium': ('calcium_100g', 'g', 100.0), 'iron': ('iron_100g', 'g', 100.0),
    'potassium': ('potassium_100g', 'g', 100.0), 'vitamin_d': ('vitamin-d_100g', 'g', 100.0),
    'trans_fat': ('trans-fat_100g', 'g', 100.0), 'added_sugars': ('added-sugars_100g', 'g', 100.0),
    'cholesterol': ('cholesterol_100g', 'g', 100.0),
}
META = ['product_name', 'brands', 'categories', 'countries', 'serving_size', 'image_nutrition_url']
MIN_IMAGES, MIN_SPANS = 5, 5


def labels(value):
    if value is None or pd.isna(value):
        return []
    return sorted({part.strip().casefold() for part in str(value).split(',') if part.strip().casefold() not in PLACEHOLDERS})


def nonempty(series):
    return ~series.fillna('').astype(str).str.strip().str.casefold().isin(PLACEHOLDERS)


def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def update_reservoir(previous, old_count, incoming, rng, capacity=20000):
    """Uniform bounded sample across chunks; quantiles are explicitly approximate."""
    incoming = np.asarray(incoming, dtype=float)
    total = old_count + len(incoming)
    size = min(capacity, total)
    if not size:
        return np.empty(0)
    retained = int(rng.hypergeometric(old_count, len(incoming), size))
    old = rng.choice(previous, retained, replace=False) if retained else np.empty(0)
    new = rng.choice(incoming, size-retained, replace=False) if size-retained else np.empty(0)
    return np.concatenate([old, new])


def scan_full_background(root, output, *, force=False):
    """Read local complete gzip in chunks. Exact counts; bounded-sample quantiles."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = root / 'data/full_off/en.openfoodfacts.org.products.csv.gz'
    marker = output / 'full_background_manifest.json'
    source_state = fingerprint(source)
    if marker.exists() and not force:
        cached = json.loads(marker.read_text(encoding='utf-8'))
        if cached['source'] == source_state and cached['version'] == BACKGROUND_VERSION:
            for name in ['representativeness.parquet', 'label_distributions.parquet', 'metadata_coverage.parquet', 'distribution_bins.parquet']:
                if not (output / name).exists():
                    raise ValueError('Background cache incomplete; rebuild explicitly.')
            return cached
    wanted = ['code', *META, *[value[0] for value in FIELDS.values()]]
    header = pd.read_csv(source, sep='\t', compression='gzip', nrows=0).columns
    columns = [column for column in wanted if column in header]
    if 'code' not in columns:
        raise ValueError('Complete Part 1 source lacks code.')
    state = {name: {'count': 0, 'negative': 0, 'upper_review': 0, 'min': None, 'max': None,
                    'sample': np.empty(0), 'histogram': np.zeros(42, dtype=np.int64)} for name in FIELDS}
    metadata_counts = Counter()
    label_counts = {'countries': Counter(), 'categories': Counter()}
    rng = np.random.default_rng(42)
    rows = 0
    for chunk in pd.read_csv(source, sep='\t', compression='gzip', usecols=columns, dtype=str,
                             keep_default_na=False, chunksize=20000, on_bad_lines='error'):
        rows += len(chunk)
        for name in META:
            if name in chunk:
                metadata_counts[name] += int(nonempty(chunk[name]).sum())
        for column in label_counts:
            if column in chunk:
                for value in chunk[column]:
                    label_counts[column].update(labels(value))
        for name, (column, unit, upper) in FIELDS.items():
            if column not in chunk:
                continue
            values = pd.to_numeric(chunk[column], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            item = state[name]
            item['sample'] = update_reservoir(item['sample'], item['count'], values, rng)
            item['count'] += len(values)
            item['negative'] += int((values < 0).sum())
            item['upper_review'] += int((values > upper).sum())
            if len(values):
                item['min'] = min(item['min'], float(values.min())) if item['min'] is not None else float(values.min())
                item['max'] = max(item['max'], float(values.max())) if item['max'] is not None else float(values.max())
            edges = np.concatenate(([-np.inf], np.linspace(0, upper, 41), [np.inf]))
            item['histogram'] += np.histogram(values, bins=edges)[0]
        if rows % 500000 == 0:
            print(f'Background scan: {rows:,} full-source rows; no row limit', flush=True)
    integration = json.loads((root / 'outputs/part1_part2_integration/provenance.json').read_text(encoding='utf-8'))
    if rows != integration['summary']['part1_product_rows'] or fingerprint(source) != source_state:
        raise ValueError('Full-source count/stability differs from integration. No mixed snapshot is certified.')
    from nutrition_extraction.linked_data import load_linked, products_for_eda
    products, _, _ = load_linked(root / 'outputs/part1_part2_integration')
    matched = products_for_eda(products)
    broad50 = pd.read_csv(root / 'data/openfoodfacts_eda_enriched.csv', dtype={'code': 'string'})
    representations, metadata_rows, label_rows, histogram_rows = [], [], [], []
    def describe(name, dataset, values, denominator, approximate=False, exact=None, field_available=True):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        column, unit, upper = FIELDS[name]
        quantiles = np.quantile(values, [.01, .25, .5, .75, .99]) if len(values) else [None]*5
        count = exact['count'] if exact is not None else len(values)
        representations.append(dict(dataset=dataset, nutrient=name, source_field=column, unit=unit,
            products_or_rows=denominator, numeric_count=count, coverage=count/denominator if denominator and field_available else None,
            source_field_available=field_available,
            p01=quantiles[0], q25=quantiles[1], median=quantiles[2], q75=quantiles[3], p99=quantiles[4],
            quantile_sample_size=len(values), quantiles_approximate=approximate,
            negative_count=exact['negative'] if exact is not None else int((values < 0).sum()),
            upper_review_count=exact['upper_review'] if exact is not None else int((values > upper).sum()),
            min=exact['min'] if exact is not None else float(values.min()) if len(values) else None,
            max=exact['max'] if exact is not None else float(values.max()) if len(values) else None))
        counts = exact['histogram'] if exact is not None else np.histogram(values, bins=np.concatenate(([-np.inf], np.linspace(0, upper, 41), [np.inf])))[0]
        for index, count_bin in enumerate(counts):
            histogram_rows.append(dict(dataset=dataset, nutrient=name, bin_index=index,
                bin_label='<0' if index == 0 else f'>{upper:g}' if index == 41 else f'{(index-1)*upper/40:g}–{index*upper/40:g}',
                count=int(count_bin), numeric_denominator=count, source_field_available=field_available, fraction=float(count_bin/count) if count else None))
    for name in FIELDS:
        describe(name, 'Full Part 1 snapshot (all rows)', state[name]['sample'], rows, True, state[name])
    datasets = [('Full Part 1 snapshot (all rows)', None, rows), ('Broad 50K EDA sample', broad50, len(broad50)), ('Matched annotated products', matched, len(matched))]
    for dataset_name, frame, denominator in datasets:
        for column in META:
            count = metadata_counts[column] if frame is None else int(nonempty(frame[column]).sum()) if column in frame else 0
            metadata_rows.append(dict(dataset=dataset_name, field=column, count=count, denominator=denominator, fraction=count/denominator))
        for column in label_counts:
            counter = label_counts[column] if frame is None else Counter(value for raw in frame[column] for value in labels(raw)) if column in frame else Counter()
            for value, count in counter.items():
                label_rows.append(dict(dataset=dataset_name, dimension=column, label=value, count=count, denominator=denominator, fraction=count/denominator))
        if frame is not None:
            for name, (column, _, _) in FIELDS.items():
                values = pd.to_numeric(frame[column], errors='coerce').dropna().to_numpy() if column in frame else []
                describe(name, dataset_name, values, denominator, field_available=column in frame)
    for filename, records in [('representativeness', representations), ('label_distributions', label_rows), ('metadata_coverage', metadata_rows), ('distribution_bins', histogram_rows)]:
        pd.DataFrame(records).to_parquet(output / f'{filename}.parquet', index=False)
    manifest = dict(version=BACKGROUND_VERSION, source=source_state, full_rows=rows, matched_products=len(matched), broad_sample_rows=len(broad50),
                    quantile_method='seed42 bounded uniform reservoir, at most 20,000 finite observations per nutrient; full quantiles approximate, counts/histograms exact',
                    duplicate_note='Full distributions are product-row weighted (56 repeated barcode rows); matched subset is unique-product weighted.',
                    generated_at_utc=datetime.now(timezone.utc).isoformat())
    marker.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def nutrient_key(label):
    from part2.layoutlm_bio import _label_semantics
    name, basis, unit = _label_semantics(label)
    return (f'energy_{unit}' if name == 'energy' else name), basis


def rule_flags(nutrient, value, unit, basis):
    """Review flags, not automatic corrections. Never apply 100g limits per serving."""
    from part2.normalization import canonicalize, normalize_unit
    if value is None or not np.isfinite(value):
        return ['missing_or_nonfinite']
    flags = ['negative_quantity'] if value < 0 else []
    unit = normalize_unit(unit)
    if basis != 'per_100g':
        return flags
    if unit in {'mg', 'ug', 'g'}:
        mass, normalized = canonicalize(value, unit)
        if normalized == 'g' and mass > 100:
            flags.append('mass_above_100g_per_100g_review')
    elif nutrient == 'energy_kcal' and unit == 'kcal' and value > 1000:
        flags.append('energy_above_1000kcal_per_100g_review')
    elif nutrient == 'energy_kj' and unit == 'kj' and value > 4200:
        flags.append('energy_above_4200kj_per_100g_review')
    return flags


def load_aligned_predictions(root, images, annotations):
    """Use the declared seeded official-test bundle only; reject unproven caches."""
    from part2.layoutlm_bio import INFERENCE_SCHEMA_VERSION, MODEL_REVISION
    path = Path(root) / 'data/part2_cache/layoutlm_bio/latest_test_bundle.json'
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('inference_schema_version') != INFERENCE_SCHEMA_VERSION or bundle.get('model_revision') != MODEL_REVISION or bundle.get('split') != 'test':
        raise ValueError('Official-test prediction bundle schema/revision/split mismatch.')
    grouped = {key: group.sort_values('token_index') for key, group in annotations.groupby('image_key', sort=False)}
    accepted, rejected = {}, []
    for sample in bundle['samples']:
        key = f"test:{sample.get('source_index')}"
        image = images.loc[images['image_key'].eq(key)]
        gold = grouped.get(key)
        reason = None
        if len(image) != 1 or gold is None:
            reason = 'missing_or_ambiguous_image_key'
        elif str(image.iloc[0]['barcode']) != str(sample['barcode']):
            reason = 'barcode_mismatch'
        elif sample['tokens'] != gold['ocr_token'].tolist() or sample['bboxes'] != [list(box) for box in gold['bounding_box']]:
            reason = 'token_or_box_alignment_mismatch'
        elif sample['gold_labels'] != gold['gold_bio_label'].tolist():
            reason = 'gold_label_mismatch'
        elif len(sample['predicted_labels']) != len(gold) or len(sample.get('coverage_mask', [])) != len(gold) or not all(sample['coverage_mask']):
            reason = 'incomplete_token_coverage'
        elif key in accepted:
            reason = 'duplicate_cache_sample'
        if reason:
            rejected.append(dict(image_key=key, reason=reason))
        else:
            accepted[key] = sample
    individual_files = list((path.parent).glob('*.json'))
    return accepted, dict(bundle_sha256=file_hash(path), bundle_path=str(path), bundle_images=len(bundle['samples']),
                           accepted_images=len(accepted), rejected=rejected, selection=bundle.get('evaluation_metadata'),
                           individual_cache_files_not_pooled=len(individual_files)-1,
                           policy='Only declared seeded official-test cohort pooled. Standalone smoke/upload/old-schema caches not pooled without equivalent provenance and alignment.')


def metric_summary(rows, min_images=MIN_IMAGES, min_spans=MIN_SPANS):
    """Micro exact-span counts, image denominator and conservative display gate."""
    totals = {key: int(rows[key].sum()) if key in rows else 0 for key in ['tp', 'pred_spans', 'gold_spans', 'parsed_predictions', 'pred_entities', 'token_errors', 'token_support']}
    if 'token_count' in rows:
        totals['token_support'] = int(rows.drop_duplicates('image_key')['token_count'].sum())
    if 'token_error_indices' in rows:
        totals['token_errors'] = sum(len({int(index) for positions in group['token_error_indices'] for index in positions}) for _, group in rows.groupby('image_key'))
    images = rows['image_key'].nunique() if 'image_key' in rows else 0
    contributing_images = rows.loc[rows['gold_spans'].gt(0) | rows['pred_spans'].gt(0), 'image_key'].nunique() if not rows.empty else 0
    sufficient = contributing_images >= min_images and totals['gold_spans'] >= min_spans
    p = totals['tp']/totals['pred_spans'] if totals['pred_spans'] else 0.0
    r = totals['tp']/totals['gold_spans'] if totals['gold_spans'] else 0.0
    parse_support = totals['pred_entities']
    return {**totals, 'images': int(images), 'entity_contributing_images': int(contributing_images), 'sufficient_support': sufficient,
            'precision': p if sufficient else None, 'recall': r if sufficient else None,
            'f1': (2*p*r/(p+r) if p+r else 0.0) if sufficient else None,
            'parsing_success': totals['parsed_predictions']/parse_support if contributing_images >= min_images and parse_support >= min_spans else None,
            'note': f'Metrics suppressed below {min_images} distinct images or {min_spans} gold spans; parsing requires {min_spans} predicted entities. All counts remain visible.'}


def evaluate_cached(root, output):
    from nutrition_extraction.linked_data import load_linked, MATCHED
    from part2.layoutlm_bio import _entity_spans, decode_bio_entities, analyze_bio_errors
    from part2.normalization import canonicalize, normalize_unit
    root, output = Path(root), Path(output)
    directory = root / 'outputs/part1_part2_integration'
    products, images, provenance = load_linked(directory)
    matched = products.loc[products['match_status'].eq(MATCHED)].copy()
    images = images.loc[images['barcode'].isin(matched['barcode'])].copy()
    annotations = pq.read_table(directory / 'integrated_annotations.parquet').to_pandas()
    predictions, cache_audit = load_aligned_predictions(root, images, annotations)
    product_map = matched.set_index('barcode').to_dict('index')
    tokens_by_key = {key: group.sort_values('token_index') for key, group in annotations.groupby('image_key', sort=False) if key in set(images['image_key'])}
    performance, coverage, anomalies, agreements, image_rows, errors = [], [], [], [], [], []
    for image in images.to_dict('records'):
        key, code = image['image_key'], str(image['barcode'])
        product = product_map[code]
        structured = json.loads(product['structured_nutrients'])
        tokens = tokens_by_key[key]['ocr_token'].tolist()
        gold = tokens_by_key[key]['gold_bio_label'].tolist()
        sample = predictions.get(key)
        predicted = sample['predicted_labels'] if sample else None
        gold_entities = _entity_spans(gold)
        predicted_entities = _entity_spans(predicted) if sample else []
        decoded_gold = decode_bio_entities(tokens, gold)
        decoded_pred = decode_bio_entities(tokens, predicted) if sample else {'parsed_fields': [], 'uncertain_fields': []}
        groups = set(FIELDS) | {'serving_size'} | {nutrient_key(item['label'])[0] for item in gold_entities + predicted_entities}
        base = dict(image_key=key, barcode=code, split=image['dataset_split'],
                    country=product.get('country'), category=product.get('category'),
                    width=image['image_width'], height=image['image_height'], token_count=len(tokens),
                    aspect_ratio=image['image_width']/image['image_height'] if image['image_height'] else None,
                    metadata_fields_present=sum(bool(nonempty(pd.Series([product.get(field)])).iloc[0]) for field in ['product_name', 'brand', 'category', 'country', 'serving_size']),
                    evaluated=sample is not None)
        image_rows.append(base)
        if sample:
            errors.extend({**row, 'image_key': key} for row in analyze_bio_errors([{'tokens': tokens, 'barcode': code}], [sample]))
        for name in groups:
            info = FIELDS.get(name)
            field, unit, _ = info if info else (None, None, None)
            raw = structured.get(field) if field else None
            numeric = pd.to_numeric(pd.Series([raw]), errors='coerce').iloc[0]
            finite = pd.notna(numeric) and np.isfinite(numeric)
            selected_gold = [item for item in gold_entities if nutrient_key(item['label'])[0] == name]
            selected_pred = [item for item in predicted_entities if nutrient_key(item['label'])[0] == name]
            parsed_pred = [item for item in decoded_pred['parsed_fields'] if nutrient_key(item['label'])[0] == name]
            uncertain_pred = [item for item in decoded_pred['uncertain_fields'] if nutrient_key(item['label'])[0] == name]
            coverage.append({**base, 'nutrient': name, 'structured_present': raw not in (None, ''),
                             'structured_mapping_available': info is not None,
                             'structured_numeric': bool(finite), 'gold_present': bool(selected_gold),
                             'prediction_present': bool(selected_pred) if sample else None,
                             'gold_spans': len(selected_gold), 'pred_entities': len(selected_pred) if sample else None,
                             'parsed_predictions': len(parsed_pred) if sample else None,
                             'parsing_failures': len(uncertain_pred) if sample else None,
                             'metadata_state': 'not_mapped_to_structured_nutrient_field' if info is None else 'available_numeric' if finite else 'missing' if raw in (None, '') else 'present_not_numeric',
                             'annotation_state': 'gold_annotated' if selected_gold else 'not_annotated_not_proof_absent',
                             'prediction_state': 'not_evaluated' if not sample else 'no_predicted_entity' if not selected_pred else 'parsed' if parsed_pred else 'parsing_failed'})
            if finite:
                flags = rule_flags(name, float(numeric), unit, 'per_100g')
                parent = {'sugars': 'carbohydrates_100g', 'saturated_fat': 'fat_100g', 'trans_fat': 'fat_100g'}.get(name)
                if parent:
                    parent_value = pd.to_numeric(pd.Series([structured.get(parent)]), errors='coerce').iloc[0]
                    if pd.notna(parent_value) and np.isfinite(parent_value) and float(numeric) > float(parent_value) + 0.1:
                        flags.append('subnutrient_exceeds_parent_same_basis_review')
                anomalies.append({**base, 'nutrient': name, 'source_field': field, 'value': float(numeric), 'unit': unit,
                                  'basis': 'per_100g', 'rules': '|'.join(flags), 'physical_review': bool(flags),
                                  'rule_scope': 'source _100g convention; unit/nutrient-specific screening, not verified error'})
            if not sample:
                continue
            gold_set = {(item['label'], item['start'], item['end']) for item in selected_gold}
            pred_set = {(item['label'], item['start'], item['end']) for item in selected_pred}
            # Errors restricted to this nutrient on either side, retaining exact fine label/basis.
            involved = [index for index, (left, right) in enumerate(zip(gold, predicted))
                        if left != right and ((left != 'O' and nutrient_key(left[2:])[0] == name)
                                              or (right != 'O' and nutrient_key(right[2:])[0] == name))]
            performance.append({**base, 'nutrient': name, 'tp': len(gold_set & pred_set), 'gold_spans': len(gold_set),
                                'pred_spans': len(pred_set), 'pred_entities': len(selected_pred),
                                'parsed_predictions': len(parsed_pred), 'parsing_failures': len(uncertain_pred),
                                'token_errors': len(involved), 'token_support': len(tokens),
                                'token_error_indices': involved,
                                'parsing_reasons': '|'.join(sorted({reason for item in uncertain_pred for reason in item['uncertainty_reasons']})),
                                'structured_mapping_available': info is not None,
                                'structured_numeric': bool(finite), 'gold_present': bool(selected_gold)})
            for candidate in parsed_pred:
                _, basis = nutrient_key(candidate['label'])
                candidate_unit = normalize_unit(candidate.get('unit'))
                value = float(candidate['value'])
                if candidate_unit in {'g', 'mg', 'ug'}:
                    value, candidate_unit = canonicalize(value, candidate_unit)
                compatible = basis == 'per_100g' and finite and candidate_unit == unit and not candidate.get('qualifier')
                reasons = []
                if field is None:
                    reasons.append('nutrient_source_field_unmapped')
                if not finite:
                    reasons.append('structured_numeric_metadata_missing')
                if basis != 'per_100g':
                    reasons.append('serving_or_unknown_basis_not_comparable_to_100g')
                if candidate_unit != unit:
                    reasons.append('unit_incompatible')
                if candidate.get('qualifier'):
                    reasons.append('qualified_bound_not_point_value')
                delta = value-float(numeric) if compatible else None
                agreement = bool(np.isclose(value, numeric, rtol=0.01, atol=1e-8)) if compatible else None
                agreements.append({**base, 'nutrient': name, 'raw_text': candidate['raw_text'],
                                   'basis': basis, 'unit': candidate_unit, 'predicted_value': value,
                                   'product_value': float(numeric) if finite else None, 'source_field': field,
                                   'compatible': compatible, 'difference': delta, 'within_one_percent': agreement,
                                   'incompatibility_reasons': '|'.join(reasons),
                                   'prediction_validation_flags': '|'.join(rule_flags(name, value, candidate_unit, basis)),
                                   'comparison_status': 'compatible_agreement' if compatible and agreement else 'compatible_discrepancy' if compatible else 'not_comparable_basis_unit_qualifier_or_missing_metadata',
                                   'reference_is_gold': False,
                                   'interpretation': 'Field-name/basis/unit compatible only; current product/packaging identity and density not verified.'})
    frames = {'performance_rows': pd.DataFrame(performance), 'coverage_rows': pd.DataFrame(coverage),
              'anomaly_rows': pd.DataFrame(anomalies), 'agreement_rows': pd.DataFrame(agreements),
              'image_rows': pd.DataFrame(image_rows), 'error_examples': pd.DataFrame(errors)}
    anomaly = frames['anomaly_rows']
    thresholds = []
    for name, group in anomaly.groupby('nutrient'):
        p1, q1, median, q3, p99 = group['value'].quantile([.01, .25, .5, .75, .99])
        iqr = q3-q1
        mask = anomaly['nutrient'].eq(name)
        anomaly.loc[mask, 'iqr_outlier'] = (anomaly.loc[mask, 'value'] < q1-1.5*iqr) | (anomaly.loc[mask, 'value'] > q3+1.5*iqr)
        anomaly.loc[mask, 'tail_p01_p99'] = (anomaly.loc[mask, 'value'] < p1) | (anomaly.loc[mask, 'value'] > p99)
        thresholds.append(dict(nutrient=name, numeric_products=len(group), p01=p1, q25=q1, median=median, q75=q3, p99=p99,
                               iqr_lower=q1-1.5*iqr, iqr_upper=q3+1.5*iqr, thresholds_scope='matched product metadata descriptive only; no model tuning'))
    frames['anomaly_thresholds'] = pd.DataFrame(thresholds)
    for name, frame in frames.items():
        frame.to_parquet(output / f'{name}.parquet', index=False)
    cache_audit.update(evaluated_matched_images=len(predictions), matched_images=len(images), model_inference_runs=0)
    (output / 'prediction_audit.json').write_text(json.dumps(cache_audit, indent=2), encoding='utf-8')
    return frames, cache_audit, provenance


def filtered_rows(frame, *, nutrients=(), countries=(), categories=(), splits=()):
    mask = pd.Series(True, index=frame.index)
    if nutrients and 'nutrient' in frame:
        mask &= frame['nutrient'].isin(nutrients)
    for column, wanted in [('country', countries), ('category', categories)]:
        if wanted:
            mask &= frame[column].map(lambda value: bool(set(labels(value)) & set(wanted))) if column in frame else False
    if splits:
        mask &= frame['split'].isin(splits)
    return frame.loc[mask].copy()


def performance_groups(rows, dimension):
    """One group contribution per image/nutrient; multi-label memberships disclosed."""
    frame = rows.copy()
    if dimension in {'country', 'category'}:
        frame['group'] = frame[dimension].map(lambda value: labels(value) or ['(metadata missing)'])
        frame = frame.explode('group')
    elif dimension == 'token_count':
        frame['group'] = pd.cut(frame['token_count'], [0, 60, 120, np.inf], right=True, include_lowest=True, labels=['≤60', '61–120', '>120']).astype(str)
    elif dimension == 'aspect_ratio':
        frame['group'] = pd.cut(frame['aspect_ratio'], [0, .75, 1.5, np.inf], include_lowest=True, labels=['portrait <0.75', 'near-square 0.75–1.5', 'landscape >1.5']).astype(str)
    elif dimension == 'width':
        frame['group'] = pd.cut(frame['width'], [0, 1000, 2000, np.inf], include_lowest=True, labels=['≤1000 px', '1001–2000 px', '>2000 px']).astype(str)
    elif dimension == 'metadata_fields_present':
        frame['group'] = np.where(frame[dimension] >= 4, '4–5 metadata fields', '0–3 metadata fields')
    elif dimension == 'structured_numeric':
        if 'structured_mapping_available' in frame:
            frame = frame.loc[frame['structured_mapping_available']].copy()
        frame['group'] = frame[dimension].map({True: 'numeric product metadata present', False: 'numeric product metadata missing'})
    else:
        frame['group'] = frame[dimension]
    output = []
    for group, subset in frame.groupby('group', observed=True):
        if subset['gold_spans'].sum() or subset['pred_spans'].sum():
            output.append(dict(dimension=dimension, group=str(group), **metric_summary(subset)))
    return pd.DataFrame(output)


def coverage_summary(rows):
    result = []
    for nutrient, group in rows.groupby('nutrient'):
        evaluated = group.loc[group['evaluated']]
        n, e = len(group), len(evaluated)
        mapped = bool(group['structured_mapping_available'].all()) if 'structured_mapping_available' in group else True
        for source, count, denominator in [
            ('Structured numeric product records', int(group['structured_numeric'].sum()), n),
            ('Gold BIO annotation present', int(group['gold_present'].sum()), n),
            ('Structured numeric (evaluated cohort)', int(evaluated['structured_numeric'].sum()), e),
            ('Gold BIO (evaluated cohort)', int(evaluated['gold_present'].sum()), e),
            ('Model BIO prediction (evaluated only)', int(evaluated['prediction_present'].eq(True).sum()), e)]:
            supported = mapped or not source.startswith('Structured numeric')
            result.append(dict(nutrient=nutrient, source=source, images_with_value=count if supported else None,
                               denominator=denominator, coverage=count/denominator if denominator and supported else None,
                               state='measured' if supported else 'source_field_not_mapped_not_missing'))
    return pd.DataFrame(result)


def write_findings(output, frames, background, audit, provenance):
    """Five observations computed from actual counts; actions are not causal claims."""
    output = Path(output)
    meta = pd.read_parquet(output / 'metadata_coverage.parquet')
    perf = frames['performance_rows']
    cov = frames['coverage_rows']
    agreement = frames['agreement_rows']
    evaluated = cov.loc[cov['evaluated']]
    failures = int(evaluated['parsing_failures'].fillna(0).sum())
    entities = int(evaluated['pred_entities'].fillna(0).sum())
    mapped_cov = cov.loc[cov['structured_mapping_available']]
    missing_gold = int((mapped_cov['gold_present'] & ~mapped_cov['structured_numeric']).sum())
    gold_rows = int(mapped_cov['gold_present'].sum())
    comparable = agreement.loc[agreement['compatible']]
    discrepancies = int((~comparable['within_one_percent'].astype(bool)).sum())
    full_name = 'Full Part 1 snapshot (all rows)'
    matched_name = 'Matched annotated products'
    def metadata(field, dataset):
        row = meta.loc[meta['field'].eq(field) & meta['dataset'].eq(dataset)].iloc[0]
        return f"{row['fraction']:.1%} ({int(row['count']):,}/{int(row['denominator']):,})"
    overall = metric_summary(perf)
    token_groups = performance_groups(perf, 'token_count')
    token_description = '; '.join(f"{row['group']} words: F1 {row['f1']:.4f}, n={row['images']} images" for _, row in token_groups.loc[token_groups['sufficient_support']].iterrows())
    physical = frames['anomaly_rows'].loc[frames['anomaly_rows']['physical_review']]
    findings = [
        dict(eda_finding=f"Annotation selection: {background['matched_products']:,} matched products out of {provenance['summary']['part1_unique_barcodes']:,} unique full-snapshot barcodes. Serving-size completeness: full {metadata('serving_size', full_name)}, matched {metadata('serving_size', matched_name)}.",
             impact='Observed metadata-distribution difference demonstrates that cohort composition differs; it does not prove a causal extraction effect.',
             action='Keep broad EDA and annotation-selected analysis separate; report serving-basis and selection denominators.', evidence_type='observed descriptive association', support=f"{background['full_rows']:,} full rows; {background['matched_products']:,} matched products"),
        dict(eda_finding=f"{missing_gold:,}/{gold_rows:,} image–nutrient pairs have a gold BIO span but no finite structured product value in the mapped field.",
             impact='Missing product metadata can coexist with annotated nutrition text; metadata absence must not be counted as model failure or used to drop gold examples.',
             action='Preserve gold annotations independently; label missing metadata, absent annotations and unrun predictions separately.', evidence_type='observed paired-source availability', support=f'{gold_rows:,} gold-positive image–nutrient pairs'),
        dict(eda_finding=f"{failures}/{entities} predicted BIO spans in the aligned cached cohort remain uncertain after conservative numerical parsing.",
             impact='High span tagging accuracy does not ensure complete usable nutrition JSON; units, duplicates and multi-number spans need separate review.',
             action='Report parsing support beside BIO PRF and keep uncertain raw spans instead of guessing corrections.', evidence_type='observed cached-output diagnostic', support=f"{audit['accepted_images']} evaluated images; {entities} predicted spans"),
        dict(eda_finding=f"{discrepancies}/{len(comparable)} nutrient-identity/basis/unit-compatible extracted fields disagree with current product values beyond the predeclared 1% numeric tolerance. Product metadata also has {len(physical)} flagged quantities across {physical['barcode'].nunique()} matched products under declared physical/hierarchy review rules.",
             impact='Database records and extracted image values are not interchangeable gold; packaging changes, OCR and annotation/model mistakes are competing hypotheses.',
             action='Prioritize discrepancy examples for image/version adjudication, not automatic model correction.', evidence_type='observed cross-source discrepancy; causes unverified', support=f'{len(comparable)} comparable fields'),
           dict(eda_finding=f"Observed OCR-length strata: {token_description}. Only {audit['accepted_images']}/{background['matched_products']} matched images are evaluated; overall F1 {overall['f1']:.4f} on {overall['gold_spans']} gold spans. Sparse groups are suppressed.",
               impact='Image text/layout characteristics coincide with different scores in this small cohort, but nutrient mix and individual difficult examples can confound the association. Category/country strata are often too sparse; no causal driver is established.',
             action='Maintain the frozen official-test cohort, show every group denominator, and validate hypotheses on additional independently evaluated images before changing preprocessing.', evidence_type='observed sample limitation; driver claims exploratory', support=f"{audit['accepted_images']} official-test images; zero new inference runs"),
    ]
    pd.DataFrame(findings).to_csv(output / 'key_findings.csv', index=False)
    report_lines = ['# Part 1 EDA → Part 2 Extraction: measured observational findings', '',
                    'No model inference or dataset download was run. Existing gold annotations, prior notebooks and cached predictions are unchanged.', '',
                    f"Full background: {background['full_rows']:,} product rows. Matched cohort: {background['matched_products']:,} unique products. Evaluated: {audit['accepted_images']} aligned cached official-test images.", '',
                    f"Exact BIO span precision/recall/F1: {overall['precision']:.4f}/{overall['recall']:.4f}/{overall['f1']:.4f}. Numerical parsing success: {overall['parsed_predictions']}/{overall['pred_entities']} ({overall['parsing_success']:.2%}).", '']
    for index, finding in enumerate(findings, 1):
        report_lines.extend([f'## Finding {index}', '', finding['eda_finding'], '',
                             '**Part 2 implication:** ' + finding['impact'], '', '**Action:** ' + finding['action'], '',
                             '**Evidence type/support:** ' + finding['evidence_type'] + ' — ' + finding['support'], ''])
    report_lines.extend(['## Limits and interpretation', '',
                         '- Full counts and fixed-bin distributions are exact row-weighted quantities. Full P1/P25/P50/P75/P99 are approximate from a seed42 uniform reservoir capped at 20,000 finite values per nutrient; matched and 50K quantiles are exact.',
                         '- Missing, null, undefined and other explicit placeholder metadata are excluded from completeness/label counts. Multilingual labels are not forced into one inferred country/category.',
                         '- Model scores concern supplied corpus OCR, not unseen-image EasyOCR accuracy. Twenty selected test images cannot establish a causal or general population error driver.',
                         '- Groups below five entity-contributing images or five gold spans have no reported PRF. Country/category labels overlap; groups are not independent.',
                         '- Cross-source discrepancies use identity/unit/basis field conventions and a fixed 1% numeric tolerance. Packaging version, density and image identity are unverified. Product records are never treated as image gold.',
                         '- Per-100g physical review bounds are not applied to serving quantities. Statistical tails and hierarchy flags trigger review, not data correction or test-set tuning.',
                         '- Standalone smoke/upload prediction caches are excluded; only the declared seeded official-test bundle passes strict image-key/barcode/token/box/gold/coverage checks.'])
    (output / 'analysis_findings.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    # Persist aggregations for fast initial display; interactive filters aggregate small cached rows.
    coverage_summary(cov).to_parquet(output / 'coverage_summary.parquet', index=False)
    for dimension in ['nutrient', 'country', 'category', 'token_count', 'aspect_ratio', 'width', 'metadata_fields_present', 'structured_numeric']:
        performance_groups(perf, dimension).to_parquet(output / f'performance_{dimension}.parquet', index=False)
    return findings


def run_analysis(root, output=None, *, force_background=False):
    root = Path(root).resolve()
    for path in [root, root / 'src']:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    output = Path(output) if output else root / 'outputs/eda_extraction_impact'
    output.mkdir(parents=True, exist_ok=True)
    protected_paths = [path for path in root.rglob('*.ipynb') if path.name != 'part1_eda_impact_on_part2.ipynb' and '.venv' not in str(path)]
    protected_paths += list((root / 'outputs/part1_part2_integration').glob('*.parquet'))
    protected_paths += list((root / 'data/part2_cache/layoutlm_bio').glob('*.json'))
    before = {str(path): file_hash(path) for path in protected_paths}
    background = scan_full_background(root, output, force=force_background)
    frames, audit, provenance = evaluate_cached(root, output)
    findings = write_findings(output, frames, background, audit, provenance)
    after = {str(path): file_hash(path) for path in protected_paths}
    if before != after:
        raise ValueError('Protected notebooks/integrated data/prediction cache changed during analysis.')
    manifest = dict(version=VERSION, generated_at_utc=datetime.now(timezone.utc).isoformat(), background=background,
                    integration_summary=provenance['summary'], prediction_audit=audit,
                    matched_products=background['matched_products'], evaluated_images=audit['accepted_images'],
                    minimum_images=MIN_IMAGES, minimum_gold_spans=MIN_SPANS, model_inference_runs=0,
                    existing_notebooks_and_inputs_unchanged=True, input_sha256=before,
                    limitations=['Observational audit; no causal claims.', 'Only the seeded cached official-test cohort is evaluated; train predictions are unavailable, not zero.',
                                 'Full numeric quantiles approximate from seed42 bounded uniform reservoirs; matched/50K quantiles exact.',
                                 'Full dataset row weighting includes 56 duplicates; coverage/labels count available source rows, not unique products.',
                                 'Cross-source values share field-name basis conventions, not verified packaging identity; beverage per-volume ambiguity may remain.',
                                 'Multi-label country/category groups overlap; groups are not independent.',
                                 'Thresholds describe metadata, never tune model or gold labels against test data.'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'data_dictionary.md').write_text('''# EDA → Extraction cached output dictionary

- representativeness: one source cohort/nutrient; finite coverage counts, min/max, P1/P25/P50/P75/P99 and quantile sample size/approximation flag. source_field_available distinguishes missing columns from zero available product values.
- distribution_bins: exact same-unit fixed-bin counts/fractions per source cohort/nutrient; includes underflow/overflow bins.
- label_distributions and metadata_coverage: source cohort label/field counts and row/product denominator; labels may overlap.
- image_rows: one matched original image key with untouched split, product context, source dimensions/token count and evaluated flag.
- coverage_rows: matched image/nutrient availability states, mapping flag, evaluated flag and separate metadata/gold/model/parsing counts. Unevaluated predictions are null, not zero failure; serving_size has no structured nutrient mapping.
- performance_rows: evaluated image/nutrient exact gold/predicted/true-positive span counts; predicted parseable/uncertain counts/reasons; error token positions retained for union-based deduplicated token totals. Source split is always official test.
- performance_<dimension>: micro aggregation with distinct image count, entity-contributing image support, span counts and sparse-score suppression flag. Multi-label groups overlap.
- anomaly_rows and anomaly_thresholds: raw mapped product values/units/bases, review flags, matched-cohort IQR/P1/P99 thresholds. No values corrected; no thresholds tune the model.
- agreement_rows: parsed predicted span versus current source value; identity/basis/unit/qualifier comparability reasons, numeric difference, predeclared 1% tolerance, reference_is_gold=False. No pooled mixed-unit MAE.
- error_examples: actual original-token gold/prediction BIO disagreements; OCR transcription not independently verified.
- key_findings.csv and analysis_findings.md: five computed observations with interpretation, actions, evidence and support.
- manifest.json/prediction_audit.json/full_background_manifest.json: model-cache cohort provenance, strict alignment checks, protected-input hashes, full-source fingerprint, version, quantile method, no-inference guarantee and limitations.
''', encoding='utf-8')
    return manifest, frames, findings
