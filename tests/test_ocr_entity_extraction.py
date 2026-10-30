from nutrition_extraction.evaluation import character_error_rate, word_error_rate
from nutrition_extraction.extraction import canonicalize_measurement, extract_record, tag_ocr_tokens
from nutrition_extraction.ocr import OCRToken
from nutrition_extraction.schema import NutritionRecord


def test_tagging_matches_specific_nutrient_aliases() -> None:
    tokens = [
        OCRToken("Saturated fat", (0, 0, 80, 20), 0.9),
        OCRToken("Fat", (0, 30, 30, 50), 0.9),
        OCRToken("200 kcal", (100, 0, 170, 20), 0.9),
    ]
    tagged = tag_ocr_tokens(tokens)
    assert tagged[0]["normalized"]["nutrient"] == "saturated_fat"
    assert tagged[1]["normalized"]["nutrient"] == "fat"
    assert tagged[2]["entity_type"] == "VALUE"
    assert tagged[2]["normalized"] == {"value": 200.0, "unit": "kcal"}


def test_record_serialization_round_trip() -> None:
    tokens = [
        OCRToken("Energy", (10, 10, 70, 30), 0.99),
        OCRToken("200 kcal", (100, 10, 170, 30), 0.98),
    ]
    record = extract_record("test-product", tokens, "panel.jpg")
    restored = NutritionRecord.from_dict(record.to_dict())
    assert restored.product_id == record.product_id
    assert restored.source_image == "panel.jpg"
    assert restored.fields[0].nutrient == "energy"


def test_error_rate_metrics() -> None:
    assert character_error_rate("abc", "abc") == 0
    assert word_error_rate("one two", "one") == 0.5


def test_french_nutrient_aliases_and_canonical_units() -> None:
    tokens = [
        OCRToken("Énergie", (0, 0, 70, 20), 0.9),
        OCRToken("237", (80, 0, 110, 20), 0.9),
        OCRToken("kcal", (115, 0, 150, 20), 0.9),
        OCRToken("Matières grasses", (0, 30, 120, 50), 0.9),
        OCRToken("9,5", (130, 30, 155, 50), 0.9),
        OCRToken("g", (160, 30, 170, 50), 0.9),
    ]
    record = extract_record("fr-product", tokens)
    assert [field.nutrient for field in record.fields] == ["energy", "fat"]
    assert record.fields[0].normalized_unit == "kcal"
    assert record.fields[1].normalized_value == 9.5
    assert canonicalize_measurement(992, "kJ") == (992 / 4.184, "kcal")
    assert canonicalize_measurement(350, "mg") == (0.35, "g")
