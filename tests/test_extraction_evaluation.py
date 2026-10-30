from nutrition_extraction.evaluation import character_error_rate, entity_prf, structured_metrics, word_error_rate
from nutrition_extraction.extraction import normalize_measurement
from nutrition_extraction.schema import NutritionField, NutritionRecord


def test_error_rates_cover_perfect_and_empty_cases() -> None:
    assert character_error_rate("abc", "abc") == 0
    assert word_error_rate("one two", "one two") == 0
    assert character_error_rate("", "x") == 1
    assert word_error_rate("one two", "one") == 0.5


def test_entity_metrics_exact_spans() -> None:
    score = entity_prf([("NUTRIENT", 0, 1), ("VALUE", 2, 3)], [("NUTRIENT", 0, 1)])
    assert score["precision"] == 1
    assert score["recall"] == 0.5


def test_unit_normalization_decimal_comma_and_microgram() -> None:
    assert normalize_measurement("0,25", "µg") == (0.25, "ug")


def test_structured_metrics_exact_and_unit_scores() -> None:
    gold = NutritionRecord("p1", [NutritionField("protein", 3.0, "g", "100g")])
    pred = NutritionRecord("p1", [NutritionField("protein", 3.0, "g", "100g")])
    metrics = structured_metrics([gold], [pred])
    assert metrics["field_exact_match_accuracy"] == 1
    assert metrics["record_exact_match_accuracy"] == 1
    assert metrics["unit_accuracy"] == 1
    assert metrics["basis_accuracy"] == 1
    assert metrics["numeric_mae"] == 0


def test_structured_metrics_preserves_same_nutrient_in_two_units() -> None:
    gold = NutritionRecord("p2", [
        NutritionField("energy", 992.0, "kJ", "100g"),
        NutritionField("energy", 237.0, "kcal", "100g"),
    ])
    pred = NutritionRecord("p2", [
        NutritionField("energy", 992.0, "kJ", "100g"),
        NutritionField("energy", 237.0, "kcal", "100g"),
    ])
    metrics = structured_metrics([gold], [pred])
    assert metrics["field_support"] == 2
    assert metrics["field_exact_match_accuracy"] == 1
    assert metrics["record_exact_match_accuracy"] == 1


def test_numeric_mae_uses_canonical_energy_units() -> None:
    gold = NutritionRecord("p3", [NutritionField("energy", 992.0, "kJ", "100g")])
    pred = NutritionRecord("p3", [NutritionField("energy", 237.0, "kcal", "100g")])
    metrics = structured_metrics([gold], [pred])
    assert metrics["numeric_mae"] < 0.1
    assert metrics["unit_accuracy"] == 0
