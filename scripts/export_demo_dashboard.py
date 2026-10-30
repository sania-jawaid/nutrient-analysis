"""Build a portable, offline presentation snapshot. No model loads or downloads."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]


def records(frame):
    return json.loads(frame.to_json(orient='records', force_ascii=False))


def script_json(value):
    """Keep even untrusted OCR/product strings inside their JSON script element."""
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def build_payload(root=ROOT):
    root = Path(root)
    def read_json(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))
    linked = root / 'outputs/part1_part2_integration'
    products = pd.read_parquet(linked/'integrated_products.parquet')
    images = pd.read_parquet(linked/'integrated_images.parquet')
    readiness = pd.read_parquet(root/'outputs/preprocessing_readiness/images.parquet')
    payload = dict(exported_at=datetime.now(timezone.utc).isoformat(),
                   provenance=read_json('outputs/part1_part2_integration/provenance.json'),
                   readiness_manifest=read_json('outputs/preprocessing_readiness/manifest.json'),
                   recommendations=records(pd.read_csv(root/'outputs/preprocessing_readiness/recommendations.csv')),
                   readiness=records(readiness[['image_key','split','width','aspect_ratio','token_count','subword_tokens_with_specials','zero_area_boxes','quality_flags']]),
                   completeness=records(pd.read_parquet(root/'outputs/eda_extraction_impact/metadata_coverage.parquet')),
                   products=records(products[['barcode','product_name','brand','country','category','serving_size','match_status']]),
                   splits=images.groupby('dataset_split').size().to_dict(), examples=[])
    manifest = payload['readiness_manifest']
    payload['provenance'] = {'summary': payload['provenance']['summary']}
    payload['readiness_manifest'] = {k:v for k,v in manifest.items() if k not in ['source_sha256','measured_image_sha256','tokenizer_assets']}
    bundle = read_json('data/part2_cache/layoutlm_bio/latest_test_bundle.json')
    payload['seeded'] = {k:bundle[k] for k in ['fine_grained_metrics','coarse_model_metrics','coarse_rule_metrics']}
    summary_path = root/'outputs/full_cohort_bio/summary.json'
    payload['cohort'] = read_json('outputs/full_cohort_bio/summary.json') if summary_path.exists() else None
    payload['cohort_rows'] = []
    if payload['cohort']:
        payload['cohort'].pop('runtime', None)
        # Capture one consistent snapshot from immutable per-image checkpoints,
        # rather than racing the worker's independently published CSV/summary.
        fingerprint = payload['cohort']['input_fingerprint']
        selected_samples = []
        for path in sorted((summary_path.parent/'samples').glob('*.json')):
            sample = json.loads(path.read_text(encoding='utf-8'))
            if sample.get('input_fingerprint') != fingerprint:
                continue
            payload['cohort_rows'].append(sample['summary_row'])
            if len(selected_samples) < 6:
                selected_samples.append(sample)
        sys.path.insert(0,str(root))
        from scripts.full_cohort_bio import aggregate
        rows = payload['cohort_rows']
        payload['cohort']['completed_images'] = len(rows)
        payload['cohort']['pending_images'] = max(0, payload['cohort']['target_images']-len(rows)-payload['cohort']['failed_images'])
        payload['cohort']['metrics_by_split'] = {split:aggregate([r for r in rows if r['split']==split]) for split in ['train','test']}
        for sample in selected_samples:
            prediction = sample['prediction']
            image_path = Path(sample['image_path'])
            preview = None
            if image_path.is_file():
                with Image.open(image_path) as source:
                    image = source.convert('RGB')
                    image.thumbnail((600,600))
                    stream = BytesIO()
                    image.save(stream,format='JPEG',quality=78)
                    preview = 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')
            payload['examples'].append(dict(image_key=sample['image_key'], barcode=sample['barcode'],
                split=sample['split'], image=preview, tokens=sample['tokens'],
                predicted_labels=prediction['predicted_labels'],gold_labels=prediction['gold_labels'],
                fields=prediction['structured_output'].get('parsed_fields',[]),
                uncertain_fields=prediction['structured_output'].get('uncertain_fields',[])))
    return payload


def export(root=ROOT):
    root = Path(root)
    payload = build_payload(root)
    template = (root/'src/nutrition_extraction/demo_dashboard.html').read_text(encoding='utf-8')
    html = template.replace('/*__PLOTLY__*/', get_plotlyjs()).replace('__SNAPSHOT_JSON__', script_json(payload))
    destination = root/'outputs/presentation/nutrition_dashboard_demo.html'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(html,encoding='utf-8')
    print(f'Created {destination} ({destination.stat().st_size/1024/1024:.1f} MiB)')
    print(f"Snapshot: {len(payload['cohort_rows'])}/3075 full-cohort predictions; no inference was run.")
    return destination


if __name__ == '__main__':
    export()