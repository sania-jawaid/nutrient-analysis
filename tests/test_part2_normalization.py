from part2.normalization import canonicalize, nutrition_basis, parse_number, parse_value_unit


def test_decimal_separator_parsing() -> None:
    assert parse_number("9,5") == 9.5
    assert parse_number("1.234,5") == 1234.5
    assert parse_number("1,234.5") == 1234.5


def test_unit_normalization_and_canonicalization() -> None:
    parsed = parse_value_unit("0,35 mg")
    assert parsed is not None
    assert parsed.value == 0.35
    assert parsed.unit == "mg"
    assert canonicalize(1000, "mg") == (1.0, "g")
    converted, unit = canonicalize(418.4, "kJ")
    assert abs(converted - 100.0) < 1e-10
    assert unit == "kcal"
    assert canonicalize(1.5, "L") == (1500.0, "ml")


def test_basis_detection_and_missing_value() -> None:
    assert nutrition_basis("Average values per 100 g") == "per_100g"
    assert nutrition_basis("Pour une portion") == "per_serving"
    assert nutrition_basis("100 ml") == "per_100ml"
    assert parse_value_unit("not declared") is None
    assert parse_value_unit("") is None