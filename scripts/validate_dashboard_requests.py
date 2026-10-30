"""Interactive regression checks for the requested dashboard layout and filters."""
from pathlib import Path
import sys
import pandas as pd
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from nutrition_extraction.dashboard_data import filter_products, label_support, annotation_rows
from nutrition_extraction.linked_data import load_linked, products_for_eda

app = AppTest.from_file(str(root / 'app.py'), default_timeout=90).run()
def widget(collection, label):
    return next(item for item in collection if item.label == label)
def navigate(label):
    app.radio[0].set_value(label).run()
    assert not app.exception, app.exception
    print('PASS page:', label, flush=True)

expected = ['Data Overview', 'Data Understanding', 'Part 1 + Part 2: Common product subset', 'Results']
assert app.radio[0].options[:4] == expected
assert 'View Raw Data' in app.radio[0].options
source = widget(app.selectbox, 'Sample')
source.select(next(option for option in source.options if 'Common product subset' in option)).run()
widget(app.multiselect, 'Country labels').set_value(['France']).run()
widget(app.multiselect, 'Category labels').set_value(['Snacks']).run()
assert not app.exception, app.exception
products, images, _ = load_linked(root / 'outputs/part1_part2_integration')
filtered = filter_products(products_for_eda(products), ['France'], ['Snacks'])
assert 0 < len(filtered) < 3075
print('Actual common products after France + Snacks:', len(filtered))
for page in ['Data Overview', 'Nutrients', 'Data quality', 'Images & alignment', 'Part 1 + Part 2: Common product subset', 'Part 2 Dataset EDA', 'View Raw Data', 'Data Understanding', 'Results', 'BIO Entity Tagging']:
    navigate(page)
    if page == 'Nutrients':
        assert any(metric.label == 'Mode' for metric in app.metric)
    if page == 'Part 2 Dataset EDA':
        assert any(item.value == 'Entity-tagging input table · original Part 2 annotations' for item in app.subheader)
        allowed = images.loc[images['barcode'].isin(filtered['code'])]
        expected_support = label_support(annotation_rows(root / 'outputs/part1_part2_integration', allowed['image_key']))
        support_tables = [item.value for item in app.dataframe if 'train_token_support' in item.value.columns]
        assert support_tables
        assert support_tables[-1]['total_support'].sum() == expected_support['total_support'].sum()
        assert any('View image' in str(item.proto) for item in app.dataframe), 'Image links not configured'
    if page == 'View Raw Data':
        assert any(item.value == 'Merged Part 1 + Part 2 table · separate linked view' for item in app.subheader)
    if page == 'BIO Entity Tagging':
        assert any('1,551/1,551' in item.value for item in app.caption)
        assert all('RAM' not in metric.label and 'RSS' not in metric.label for metric in app.metric)
        assert all('RAM' not in item.value and 'RSS' not in item.vaFine-grained BIO model
Coarse model projection
Coarse rule baselinelue for item in app.caption)
# Reset and restore broad default.
navigate('Data Overview')
widget(app.multiselect, 'Country labels').set_value([]).run()
widget(app.multiselect, 'Category labels').set_value([]).run()
source = widget(app.selectbox, 'Sample')
source.select(next(option for option in source.options if '50,000-row enriched' in option)).run()
navigate('Data Overview')
assert any('50,000' in item.value for item in app.markdown)
print('PASS: navigation, France+Snacks on analytical pages, filtered Part 2 gold support, modes, image links, merged raw view, no RAM telemetry, unchanged official-test results and broad default.')
