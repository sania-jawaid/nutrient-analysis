from scripts.full_cohort_bio import aggregate, atomic_json
import json
import pytest


def test_micro_counts_not_mean_f1():
    result = aggregate([
        dict(true_positive_spans=1, predicted_spans=2, gold_spans=1, token_support=3, correct_tokens=2),
        dict(true_positive_spans=3, predicted_spans=3, gold_spans=5, token_support=7, correct_tokens=6),
    ])
    assert result['precision'] == 4/5
    assert result['recall'] == 4/6
    assert result['f1'] == pytest.approx(8/11)
    assert result['token_accuracy'] == .8
    assert result['images'] == 2


def test_empty_cohort_has_no_token_accuracy():
    assert aggregate([])['token_accuracy'] is None


def test_atomic_checkpoint(tmp_path):
    path = tmp_path/'sample.json'
    atomic_json(path, {'barcode':'00123', 'done':1})
    assert json.loads(path.read_text())['barcode'] == '00123'
    assert not path.with_suffix('.tmp').exists()