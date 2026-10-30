from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.preprocessing_readiness import (
    box_diagnostics, token_diagnostics, bio_diagnostics, image_measurements,
    image_flags, filtered_images, tokenizer_lengths, local_image_path,
)


def test_geometry_checks_do_not_change_original_boxes():
    boxes = [[0,0,100,100],[0,0,100,100],[200,200,200,250],[-1,0,5,5],[50,60,40,70]]
    snapshot = [row.copy() for row in boxes]
    result = box_diagnostics(boxes)
    assert result['duplicate_box_excess'] == 1
    assert result['zero_area_boxes'] == 1
    assert result['out_of_range_box_count'] == 1
    assert result['reversed_boxes'] == 1
    assert result['high_iou_pairs'] == 1
    assert boxes == snapshot


def test_text_diagnostics_and_bio_review_preserve_raw_values():
    tokens = ['6,2g', '12.3', '4g/10g', 'Og', '1,2,3']
    stats, examples = token_diagnostics(tokens)
    assert stats['decimal_comma_tokens'] == 2
    assert stats['multiple_numeric_tokens'] == 2
    assert stats['malformed_numeric_review_tokens'] >= 2
    assert tokens[0] == '6,2g'
    gold = ['I-FAT_100G','I-FAT_100G','O','B-SALT_SERVING']
    stats, entities, invalid = bio_diagnostics(gold, set(gold) | {'B-FAT_100G'})
    assert stats['invalid_bio_transitions'] == 1
    assert stats['multi_token_entities'] == 1
    assert stats['per_100g_entities'] == 1
    assert stats['per_serving_entities'] == 1
    assert gold[0] == 'I-FAT_100G'


def test_pixel_measures_are_non_destructive_and_flags_not_quality_truth(tmp_path):
    path = tmp_path / 'dark.png'
    Image.new('RGB',(100,80),(10,10,10)).save(path)
    original = path.read_bytes()
    result = image_measurements(path)
    assert result['brightness_mean'] == 10
    assert result['contrast_std'] == 0
    assert result['laplacian_variance'] == 0
    assert 'dark_luminance_review' in image_flags(result)
    assert 'low_sharpness_proxy' in image_flags(result)
    assert path.read_bytes() == original


def test_readiness_filters_and_unknown_pixel_scope():
    table = pd.DataFrame([dict(image_key='train:0', split='train', country='France, Belgium', nutrients='fat|salt', quality_flags='short_side_lt400_review', pixel_status='not_cached_not_measured'),
                          dict(image_key='test:0', split='test', country=None, nutrients='sugars', quality_flags='', pixel_status='measured_cached_original')])
    selected = filtered_images(table, splits=['train'], countries=['belgium'], nutrients=['fat'], quality_flags=['short_side_lt400_review'])
    assert selected['image_key'].tolist() == ['train:0']
    assert selected.iloc[0]['pixel_status'] == 'not_cached_not_measured'
    assert len(table) == 2


def test_exact_length_audit_uses_no_model_forward_pass():
    class Encoded(dict):
        def word_ids(self):
            return [None, *range(520), None]
    class Tokenizer:
        def __call__(self, **kwargs):
            assert kwargs['truncation'] is False
            return Encoded(input_ids=list(range(522)))
    result = tokenizer_lengths(Tokenizer(), ['word']*520, [[0,0,1,1]]*520)
    assert result['subword_tokens_with_specials'] == 522
    assert result['needs_sliding_window']
    assert result['words_missed_by_single_pass'] == 10


def test_actual_readiness_counts_no_prediction_dependency_and_full_split_retention():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'outputs/preprocessing_readiness'
    if not (directory / 'manifest.json').exists():
        pytest.skip('Readiness artifacts not bundled')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    images = pd.read_parquet(directory / 'images.parquet')
    assert len(images) == manifest['total_annotated_images'] == 3083
    assert images.groupby('split').size().to_dict() == {'train':2884,'test':199}
    assert images['match_status'].ne('barcode_matched_image_unverified').sum() == 8
    assert manifest['prediction_caches_read'] == manifest['model_inference_runs'] == manifest['ocr_runs'] == 0
    assert not {'f1','prediction','predicted_labels','precision','recall'} & set(images.columns)
    assert images['needs_sliding_window'].eq(True).sum() == manifest['sliding_window_images']
    assert images['pixel_status'].eq('measured_cached_original').sum() == manifest['pixel_measured_images']
    assert images.loc[images['pixel_status'].eq('not_cached_not_measured'), 'brightness_mean'].isna().all()
    decisions = pd.read_csv(directory / 'recommendations.csv')
    assert set(decisions['decision_class']) == {'Required preprocessing','Optional preprocessing','Data-quality review flag','Post-inference validation'}
    assert decisions['affected_count'].le(decisions['denominator']).all()
