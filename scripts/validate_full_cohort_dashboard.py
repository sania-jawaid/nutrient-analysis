from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root/'src'))
app = AppTest.from_file(str(root/'app.py'), default_timeout=120).run()
app.radio[0].set_value('BIO Entity Tagging').run()
assert not app.exception, app.exception
assert any(m.label == 'Full-cohort completed' and int(m.value.replace(',','')) > 0 for m in app.metric)
assert any(m.label == 'Full-cohort status' for m in app.metric)
assert any('predicted_bio' in table.value for table in app.dataframe)
assert any(item.label == 'Full-cohort image result' for item in app.selectbox)
next(item for item in app.selectbox if item.label == 'Full-cohort result split').set_value('test').run()
assert not app.exception, app.exception
assert any('token_coverage' in table.value and table.value['token_coverage'].eq(1).all() for table in app.dataframe)
print('PASS: full-cohort live progress, split metrics/filter, token predictions, structured fields and preserved existing BIO page.')