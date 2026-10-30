import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.eda_extraction_impact import (
    update_reservoir, rule_flags, metric_summary, filtered_rows, coverage_summary,
    performance_groups, load_aligned_predictions,
    labels, nonempty,
)


def test_reservoir_is_bounded_and_repeatable_across_chunks():
    def run():
        rng = np.random.default_rng(42)
        sample, count = np.empty(0), 0
        for chunk in [np.arange(100), np.arange(100, 300)]:
            sample = update_reservoir(sample, count, chunk, rng, capacity=25)
            count += len(chunk)
        return sample
    assert len(run()) == 25
    assert np.array_equal(run(), run())
    assert len(set(run())) == 25
    assert set(run()).issubset(range(300))


def test_unit_basis_specific_flags_never_apply_100g_bounds_to_servings():
    assert rule_flags('energy_kcal', 120, 'kcal', 'per_100g') == []
    assert rule_flags('fat', 101, 'g', 'per_100g')
    assert rule_flags('salt', 200, 'mg', 'per_100g') == []
    assert rule_flags('fat', 150, 'g', 'per_serving') == []
    assert rule_flags('iron', -1, 'mg', 'per_serving') == ['negative_quantity']
    assert rule_flags('energy_kj', 4201, 'kj', 'per_100g')
    assert not rule_flags('energy_kcal', 900, 'kcal', 'per_100g')


def rows():
    return pd.DataFrame([dict(image_key=f'test:{i}', nutrient='fat', split='test', country='France, Belgium', category='Snacks',
                             tp=1, pred_spans=1, gold_spans=1, parsed_predictions=1, pred_entities=1,
                             token_errors=0, token_support=10, token_count=10, aspect_ratio=1.0, width=1000,
                             metadata_fields_present=4, structured_numeric=True) for i in range(5)])


def test_micro_metrics_and_sparse_groups_suppressed_by_distinct_images():
    data = rows()
    assert metric_summary(data)['f1'] == 1
    assert metric_summary(data.iloc[:4])['f1'] is None
    duplicated = pd.concat([data.iloc[:1]]*10)
    assert metric_summary(duplicated)['images'] == 1
    assert metric_summary(duplicated)['f1'] is None
    # 5 evaluated images, but only 1 entity-positive image is insufficient.
    data.loc[1:, ['tp', 'pred_spans', 'gold_spans', 'pred_entities', 'parsed_predictions']] = 0
    assert metric_summary(data)['f1'] is None
    assert metric_summary(data)['entity_contributing_images'] == 1
    assert metric_summary(data.iloc[:0])['parsing_success'] is None


def test_filters_preserve_official_splits_and_multilabel_group_counts():
    data = rows()
    original = data.copy(deep=True)
    assert len(filtered_rows(data, countries=['belgium'], categories=['snacks'], splits=['test'])) == 5
    assert filtered_rows(data, splits=['train']).empty
    assert filtered_rows(data, nutrients=['salt']).empty
    grouped = performance_groups(data, 'country')
    assert set(grouped['group']) == {'france', 'belgium'}
    assert grouped['images'].tolist() == [5, 5]
    pd.testing.assert_frame_equal(data, original)


def test_placeholder_metadata_is_missing_not_a_real_category():
    assert labels('null, Snacks, undefined, SNACKS') == ['snacks']
    assert nonempty(pd.Series([None, '', 'null', 'undefined', 'France'])).tolist() == [False, False, False, False, True]


def test_unmapped_fields_not_included_in_missing_metadata_driver():
    data = rows().assign(structured_mapping_available=False, structured_numeric=False)
    assert performance_groups(data, 'structured_numeric').empty


def test_token_totals_not_multiplied_by_nutrient_rows():
    data = pd.concat([rows().assign(token_error_indices=[[1, 2]]*5), rows().assign(nutrient='salt', token_error_indices=[[2, 3]]*5)])
    summary = metric_summary(data)
    assert summary['token_support'] == 50
    assert summary['token_errors'] == 15


def test_unmapped_serving_size_is_not_missing_nutrient_metadata():
    data = pd.DataFrame([dict(nutrient='serving_size', evaluated=True, structured_numeric=False,
                             structured_mapping_available=False, gold_present=True, prediction_present=True)])
    result = coverage_summary(data).set_index('source')
    assert pd.isna(result.loc['Structured numeric product records', 'coverage'])
    assert result.loc['Gold BIO annotation present', 'coverage'] == 1


def test_prediction_coverage_denominator_excludes_unevaluated_images():
    data = pd.DataFrame([
        dict(nutrient='fat', evaluated=True, structured_numeric=False, gold_present=True, prediction_present=True),
        dict(nutrient='fat', evaluated=False, structured_numeric=True, gold_present=True, prediction_present=None)])
    result = coverage_summary(data).set_index('source')
    assert result.loc['Model BIO prediction (evaluated only)', 'denominator'] == 1
    assert result.loc['Model BIO prediction (evaluated only)', 'coverage'] == 1
    assert result.loc['Structured numeric product records', 'denominator'] == 2
    assert result.loc['Structured numeric product records', 'coverage'] == .5


def test_real_cached_impact_kpis_and_exact_counts_match_integration():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'outputs/eda_extraction_impact'
    if not (directory / 'manifest.json').exists():
        pytest.skip('Offline analysis artifacts not bundled')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    integration = json.loads((root / 'outputs/part1_part2_integration/provenance.json').read_text(encoding='utf-8'))
    assert manifest['matched_products'] == integration['summary']['overlapping_barcodes']
    assert manifest['model_inference_runs'] == 0
    data = pd.read_parquet(directory / 'performance_rows.parquet')
    assert set(data['split']) == {'test'}
    summary = metric_summary(data)
    bundle = json.loads((root / 'data/part2_cache/layoutlm_bio/latest_test_bundle.json').read_text(encoding='utf-8'))
    assert summary['tp'] == bundle['fine_grained_metrics']['true_positive_spans']
    assert summary['gold_spans'] == bundle['fine_grained_metrics']['gold_spans']
    assert summary['f1'] == pytest.approx(bundle['fine_grained_metrics']['f1'])
    assert data['image_key'].nunique() == manifest['evaluated_images']
    assert len(pd.read_csv(directory / 'key_findings.csv')) == 5
    coverage = pd.read_parquet(directory / 'coverage_rows.parquet')
    assert coverage.groupby('nutrient')['image_key'].nunique().eq(manifest['matched_products']).all()


def test_cache_alignment_mismatch_rejected_not_silently_scored(tmp_path):
    from part2.layoutlm_bio import INFERENCE_SCHEMA_VERSION, MODEL_REVISION
    cache = tmp_path / 'data/part2_cache/layoutlm_bio'
    cache.mkdir(parents=True)
    sample = dict(source_index=0, barcode='001', tokens=['WRONG'], bboxes=[[0,0,1,1]],
                  gold_labels=['O'], predicted_labels=['O'], coverage_mask=[True])
    (cache / 'latest_test_bundle.json').write_text(json.dumps(dict(inference_schema_version=INFERENCE_SCHEMA_VERSION,
        model_revision=MODEL_REVISION, split='test', samples=[sample])))
    images = pd.DataFrame([dict(image_key='test:0', barcode='001')])
    tokens = pd.DataFrame([dict(image_key='test:0', token_index=0, ocr_token='actual', bounding_box=[0,0,1,1], gold_bio_label='O')])
    accepted, audit = load_aligned_predictions(tmp_path, images, tokens)
    assert not accepted
    assert audit['rejected'][0]['reason'] == 'token_or_box_alignment_mismatch'
