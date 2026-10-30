import pytest

from part2.entity_extraction import extract_nutrients, match_nutrient_name
from part2.schemas import ExtractionResult, OCRDetection, validate_extraction_payload


def det(text: str, x0: int, y0: int, x1: int, y1: int, order: int = 0) -> OCRDetection:
    return OCRDetection(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.9, order)


def test_nutrient_name_alias_matching() -> None:
    assert match_nutrient_name("Matières grasses") is not None
    assert match_nutrient_name("Matières grasses")[0] == "fat"
    assert match_nutrient_name("of which saturates")[0] == "saturated_fat"
    assert match_nutrient_name("no nutrient here") is None


def test_unambiguous_row_association_and_basis_preservation() -> None:
    detections = [
        det("Per 100 g", 110, 10, 190, 28),
        det("Fat", 10, 60, 55, 80),
        det("9,5", 210, 60, 240, 80),
        det("g", 246, 60, 260, 80),
    ]
    result = extract_nutrients(detections, barcode="0012345678905", image_path="fixture.jpg", image_width=300)
    value = result.nutrients["per_100g"]["fat"][0]
    assert value.value == 9.5
    assert value.unit == "g"
    assert result.barcode == "0012345678905"


def test_split_basis_header_tokens_are_recognized() -> None:
    detections = [
        det("100", 110, 10, 138, 28),
        det("g", 142, 10, 152, 28),
        det("Protein", 10, 60, 75, 80),
        det("8", 210, 60, 225, 80),
        det("g", 230, 60, 242, 80),
    ]
    result = extract_nutrients(detections, barcode=None, image_path="fixture.jpg", image_width=300)
    assert result.nutrients["per_100g"]["proteins"][0].value == 8


def test_basis_header_does_not_apply_to_values_above_it() -> None:
    detections = [
        det("Sugar", 10, 80, 60, 100),
        det("21,7 g", 100, 80, 155, 100),
        det("Values per serving", 10, 130, 180, 152),
        det("Fat", 10, 170, 45, 190),
        det("8 g", 100, 170, 130, 190),
    ]
    result = extract_nutrients(detections, barcode=None, image_path="fixture.jpg", image_width=300)
    assert result.nutrients["unspecified"]["sugars"][0].basis == "unspecified"
    assert result.nutrients["per_serving"]["fat"][0].basis == "per_serving"


def test_ambiguous_values_remain_unresolved() -> None:
    detections = [
        det("Fat", 10, 40, 55, 60),
        det("9 g", 120, 40, 155, 60),
        det("12 g", 200, 40, 240, 60),
    ]
    result = extract_nutrients(detections, barcode=None, image_path="fixture.jpg", image_width=300)
    assert "fat" not in result.nutrients.get("unspecified", {})
    assert any("ambiguous" in item["reason"] for item in result.unresolved_fields)


def test_no_numeric_value_is_not_fabricated() -> None:
    result = extract_nutrients([det("Protein", 10, 40, 70, 60)], barcode=None, image_path="fixture.jpg", image_width=200)
    assert result.nutrients == {}
    assert result.unresolved_fields[0]["reason"] == "no numeric value on the same visual row"


def test_output_schema_validation() -> None:
    result = ExtractionResult(barcode="1", image_path="x.jpg")
    validate_extraction_payload(result.to_dict())
    invalid = result.to_dict()
    invalid["nutrients"] = {"per_package": {}}
    with pytest.raises(ValueError):
        validate_extraction_payload(invalid)