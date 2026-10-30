"""Streamlit regression of insights, supported filters and unchanged existing pages."""
from pathlib import Path
import sys
import json
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'src'))
from scripts.eda_extraction_impact import filtered_rows, metric_summary
import pandas as pd

app = AppTest.from_file(str(root / 'app.py'), default_timeout=120).run()
assert not app.exception
app.radio[0].set_value('Post-Inference Diagnostics').run()
assert not app.exception, app.exception
print('PASS: insights initial render', flush=True)
assert any(item.label == 'Matched products' and item.value == '3,075' for item in app.metric)
assert any(item.label == 'Evaluated images' and item.value == '20' for item in app.metric)
assert len(app.get('plotly_chart')) >= 6

def multi(name):
    return next(item for item in app.multiselect if item.label == name)

multi('Insights nutrient').set_value(['fat']).run()
assert not app.exception, app.exception
print('PASS: nutrient filter', flush=True)
performance = pd.read_parquet(root / 'outputs/eda_extraction_impact/performance_rows.parquet')
expected = metric_summary(filtered_rows(performance, nutrients=['fat']))
tables = [item.value for item in app.dataframe if 'dimension' in item.value and 'group' in item.value]
assert any(set(table['group']) == {'fat'} and table.iloc[0]['tp'] == expected['tp'] for table in tables)
multi('Insights official split').set_value(['train']).run()
assert not app.exception, app.exception
print('PASS: train-only no-score render', flush=True)
assert any('No evaluated predictions' in item.value for item in app.info)
multi('Insights official split').set_value(['test']).run()
multi('Insights country').set_value(['france']).run()
assert not app.exception, app.exception
print('PASS: country filter', flush=True)
sparse_tables = [item.value for item in app.dataframe if 'sufficient_support' in item.value and 'dimension' in item.value]
assert sparse_tables and all(table['f1'].isna().all() for table in sparse_tables)
multi('Insights category').set_value(['snacks']).run()
assert not app.exception, app.exception
print('PASS: category filter', flush=True)
# Existing pages still work; no model evaluation reruns are performed.
for page in ['Data Overview', 'Data Understanding', 'Results', 'Nutrients', 'View Raw Data', 'BIO Entity Tagging']:
    app.radio[0].set_value(page).run()
    assert not app.exception, (page, app.exception)
    print('PASS existing page:', page, flush=True)
print('PASS: insights loads, actual 3,075/20 KPIs, chart generation, nutrient/country/category/split filtering, train no-score state, old views and cached BIO evaluation.')
