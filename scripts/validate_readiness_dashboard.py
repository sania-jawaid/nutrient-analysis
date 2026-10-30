"""Readiness UI regression: zero inference, all-source counts and supported filters."""
from pathlib import Path
import sys
import pandas as pd
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))
app = AppTest.from_file(str(root / 'app.py'), default_timeout=120).run()
assert not app.exception
assert 'Pre-Inference EDA & Readiness' in app.radio[0].options
assert 'Post-Inference Diagnostics' in app.radio[0].options
app.radio[0].set_value('Pre-Inference EDA & Readiness').run()
assert not app.exception, app.exception
assert any(item.label == 'Annotated images audited' and item.value == '3,083' for item in app.metric)
assert any(item.label == 'Barcode-matched products' and item.value == '3,075' for item in app.metric)
assert any(item.label == 'Overlength tokenizer inputs' and item.value == '455' for item in app.metric)
assert not any('F1' in item.label for item in app.metric)
assert len(app.get('plotly_chart')) >= 6
print('PASS: whole-inventory readiness renders without model scores', flush=True)

def multi(name):
    return next(item for item in app.multiselect if item.label == name)

multi('Readiness official split').set_value(['train']).run()
assert not app.exception, app.exception
assert any('2,884 source images' in item.value for item in app.caption)
print('PASS: official train filter', flush=True)
multi('Readiness nutrient').set_value(['fat']).run()
assert not app.exception, app.exception
multi('Readiness country').set_value(['france']).run()
assert not app.exception, app.exception
print('PASS: nutrient/country filters', flush=True)
multi('Readiness quality flag').set_value(['low_contrast_proxy']).run()
assert not app.exception, app.exception
print('PASS: quality flag, including empty-state safety', flush=True)
# Post-inference content preserved, not mixed with readiness.
app.radio[0].set_value('Post-Inference Diagnostics').run()
assert not app.exception, app.exception
assert any(item.label == 'Evaluated images' and item.value == '20' for item in app.metric)
print('PASS: retained post-inference page', flush=True)
for name in ['Data Overview','Data Understanding','Part 2 Dataset EDA','BIO Entity Tagging','View Raw Data']:
    app.radio[0].set_value(name).run()
    assert not app.exception, (name, app.exception)
    print('PASS existing page:', name, flush=True)
print('PASS: readiness and post-inference separation, actual KPIs, chart/filter support, source preservation and old dashboard views.')
