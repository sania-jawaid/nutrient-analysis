"""Local UI regression check; does not load OCR/models or change datasets."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(root / 'app.py'), default_timeout=90).run()
def source_widget():
	return next(widget for widget in app.selectbox if widget.label == 'Sample')

assert not app.exception, app.exception
assert source_widget().value.endswith('(50,000-row enriched sample)')
common_label = next(label for label in source_widget().options if 'Common product subset' in label)
source_widget().select(common_label).run()
assert not app.exception, app.exception
assert any('3,075' in item.value for item in app.markdown), 'Common EDA ribbon missing expected count'
app.radio[0].set_value('Nutrients').run()
assert not app.exception, app.exception
app.radio[0].set_value('Part 1 + Part 2: Common product subset').run()
assert not app.exception, app.exception
assert any(item.label == 'Common products' and item.value == '3,075' for item in app.metric)
app.radio[0].set_value('Part 2 Dataset EDA').run()
assert not app.exception, app.exception
app.radio[0].set_value('BIO Entity Tagging').run()
assert not app.exception, app.exception
assert any('1,551/1,551' in item.value for item in app.caption)
broad_label = next(label for label in source_widget().options if '50,000-row enriched' in label)
source_widget().select(broad_label).run()
app.radio[0].set_value('Data Overview').run()
assert not app.exception, app.exception
assert any('50,000' in item.value for item in app.markdown)
print('PASS: broad default, common source (3,075), nutrient EDA, common audit, Part 2 linked context, unchanged cached test evaluation, return to 50K.')
