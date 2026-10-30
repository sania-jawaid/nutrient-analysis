import json
from scripts.export_demo_dashboard import script_json


def test_embedded_json_cannot_close_script_or_inject_markup():
    value = {'text': '</script><img src=x onerror=alert(1)>', 'barcode':'00123'}
    serialized = script_json(value)
    assert '</script>' not in serialized
    assert '<img' not in serialized
    assert json.loads(serialized) == value