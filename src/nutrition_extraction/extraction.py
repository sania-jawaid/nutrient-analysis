"""Lightweight OCR-baseline nutrient tagging and spatial row association."""

from __future__ import annotations

import re
from typing import Iterable

from nutrition_extraction.ocr import OCRToken
from nutrition_extraction.preprocessing import normalize_nutrient_name, normalize_ocr_text
from nutrition_extraction.schema import NutritionField, NutritionRecord

VALUE_PATTERN = re.compile(
    r"(?P<number>[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+))\s*(?P<unit>kcal|kj|kJ|mg|mcg|µg|μg|ug|g|ml|mL|l|%|iu)?\b",
    re.IGNORECASE,
)
UNIT_PATTERN = re.compile(r"^(kcal|kj|mg|mcg|µg|μg|ug|g|ml|l|%)$", re.IGNORECASE)
NUTRIENT_ALIASES: dict[str, tuple[str, ...]] = {
    "energy": ("energy", "calories", "énergie", "energie"),
    "fat": ("fat", "total fat", "matières grasses", "matieres grasses", "matière grasse", "matiere grasse", "lipides"),
    "saturated_fat": ("saturates", "saturated fat", "of which saturates", "acides gras saturés", "acides gras satures", "dont acides gras saturés", "dont acides gras satures"),
    "carbohydrates": ("carbohydrate", "carbohydrates", "glucides"),
    "sugars": ("sugars", "sugar", "of which sugars", "sucres", "dont sucres"),
    "fiber": ("fiber", "fibre", "fibres alimentaires"),
    "proteins": ("protein", "proteins", "protéines", "proteines"),
    "salt": ("salt", "sel"),
    "sodium": ("sodium",),
}
BASIS_PATTERN = re.compile(r"(?:per\s*)?(100\s*g|100\s*ml|serving|portion)", re.IGNORECASE)


def normalize_measurement(number: str, unit: str | None) -> tuple[float, str | None]:
    """Convert decimal commas and common mass/volume aliases to stable unit strings."""
    amount = float(number.replace(",", "."))
    normalized_unit = unit.casefold() if unit else None
    aliases = {"µg": "ug", "μg": "ug", "mcg": "ug", "ml": "ml", "l": "l"}
    return amount, aliases.get(normalized_unit, normalized_unit)


def canonicalize_measurement(amount: float, unit: str | None) -> tuple[float | None, str | None]:
    """Convert common OCR units to canonical mass/volume/energy units; unknown units stay explicit."""
    if unit is None:
        return amount, None
    normalized = unit.strip().casefold().replace("μ", "µ")
    if normalized in {"g", "gram", "grams"}:
        return amount, "g"
    if normalized in {"mg", "milligram", "milligrams"}:
        return amount / 1000.0, "g"
    if normalized in {"ug", "µg", "mcg", "microgram", "micrograms"}:
        return amount / 1_000_000.0, "g"
    if normalized in {"ml", "milliliter", "milliliters", "millilitre", "millilitres"}:
        return amount, "ml"
    if normalized in {"l", "liter", "liters", "litre", "litres"}:
        return amount * 1000.0, "ml"
    if normalized in {"kj", "kilojoule", "kilojoules"}:
        return amount / 4.184, "kcal"
    if normalized in {"kcal", "calorie", "calories"}:
        return amount, "kcal"
    if normalized in {"%", "iu"}:
        return amount, normalized
    return amount, normalized


def _match_nutrient(text: str) -> str | None:
    """Find known nutrient names with longer aliases checked first to avoid substring clashes."""
    normalized = normalize_ocr_text(text).casefold().strip(" :;,.\t")
    matching_aliases = sorted(
        ((alias, canonical) for canonical, aliases in NUTRIENT_ALIASES.items() for alias in aliases),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for alias, canonical in matching_aliases:
        if normalized == alias or normalized.startswith(alias + " ") or normalized.startswith(alias + ":"):
            return canonical
    return None


def tag_ocr_tokens(tokens: Iterable[OCRToken]) -> list[dict[str, object]]:
    """Classify OCR tokens with lexical and numeric rules as a no-training baseline."""
    tagged: list[dict[str, object]] = []
    for token in tokens:
        nutrient = _match_nutrient(token.text)
        value_match = VALUE_PATTERN.search(token.text)
        basis_match = BASIS_PATTERN.search(token.text)
        if nutrient:
            entity_type = "NUTRIENT"
            normalized_value: dict[str, object] = {"nutrient": nutrient}
        elif basis_match:
            entity_type = "BASIS"
            normalized_value = {"basis": basis_match.group(1).lower().replace(" ", "")}
        elif UNIT_PATTERN.fullmatch(normalize_ocr_text(token.text).strip().casefold()):
            entity_type = "UNIT"
            normalized_value = {"unit": normalize_ocr_text(token.text).strip().casefold()}
        elif value_match:
            amount, unit = normalize_measurement(value_match.group("number"), value_match.group("unit"))
            entity_type = "VALUE"
            normalized_value = {"value": amount, "unit": unit}
        else:
            entity_type = "O"
            normalized_value = {}
        tagged.append({
            "text": token.text,
            "box": token.box,
            "confidence": token.confidence,
            "entity_type": entity_type,
            "normalized": normalized_value,
        })
    return tagged


def associate_rows(tagged_tokens: list[dict[str, object]], y_tolerance: int | None = None) -> list[dict[str, object]]:
    """Group tokens by visual baseline and associate nearby nutrient/value spans heuristically."""
    boxes = [item["box"] for item in tagged_tokens if item.get("box")]
    heights = [int(box[3]) - int(box[1]) for box in boxes]  # type: ignore[index]
    tolerance = y_tolerance or max(12, int((sorted(heights)[len(heights) // 2] if heights else 20) * 0.8))
    ordered = sorted(tagged_tokens, key=lambda item: (sum(item["box"][1::2]) / 2, item["box"][0]))  # type: ignore[index]
    rows: list[dict[str, object]] = []
    for item in ordered:
        box = item["box"]
        center_y = (int(box[1]) + int(box[3])) / 2  # type: ignore[index]
        row = next((candidate for candidate in reversed(rows) if abs(float(candidate["center_y"]) - center_y) <= tolerance), None)
        if row is None:
            row = {"center_y": center_y, "tokens": []}
            rows.append(row)
        row["tokens"].append(item)  # type: ignore[union-attr]
        members = row["tokens"]
        row["center_y"] = sum((int(member["box"][1]) + int(member["box"][3])) / 2 for member in members) / len(members)  # type: ignore[index]

    candidates: list[dict[str, object]] = []
    for row in rows:
        row_tokens = sorted(row["tokens"], key=lambda item: int(item["box"][0]))  # type: ignore[index]
        nutrients = [item for item in row_tokens if item["entity_type"] == "NUTRIENT"]
        values = [item for item in row_tokens if item["entity_type"] == "VALUE"]
        units = [item for item in row_tokens if item["entity_type"] == "UNIT"]
        bases = [item for item in row_tokens if item["entity_type"] == "BASIS"]
        if not nutrients:
            continue
        for index, nutrient_token in enumerate(nutrients):
            nutrient_x = int(nutrient_token["box"][0])  # type: ignore[index]
            next_nutrient_x = int(nutrients[index + 1]["box"][0]) if index + 1 < len(nutrients) else 10**9  # type: ignore[index]
            associated_values = [
                item for item in values
                if nutrient_x < int(item["box"][0]) < next_nutrient_x  # type: ignore[index]
            ]
            associated_units = [
                unit for unit in units
                if nutrient_x < int(unit["box"][0]) < next_nutrient_x  # type: ignore[index]
            ]
            normalized_values: list[dict[str, object]] = []
            used_units: set[int] = set()
            for value_token in associated_values:
                value_x1 = int(value_token["box"][2])  # type: ignore[index]
                next_value_x = min(
                    [int(other["box"][0]) for other in associated_values if int(other["box"][0]) > value_x1]  # type: ignore[index]
                    or [next_nutrient_x]
                )
                nearest_unit = next(
                    (unit for unit in associated_units
                     if id(unit) not in used_units
                     and value_x1 <= int(unit["box"][0]) < min(next_value_x, value_x1 + 160)),  # type: ignore[index]
                    None,
                )
                normalized = dict(value_token["normalized"])
                if nearest_unit is not None:
                    normalized["unit"] = nearest_unit["normalized"]["unit"]  # type: ignore[index]
                    used_units.add(id(nearest_unit))
                normalized_values.append(normalized)
            candidates.append({
                "nutrient": nutrient_token["normalized"]["nutrient"],  # type: ignore[index]
                "values": normalized_values,
                "basis": [item["normalized"]["basis"] for item in bases],  # type: ignore[index]
                "confidence": min(
                    [float(nutrient_token["confidence"])]
                    + [float(item["confidence"]) for item in associated_values]
                    + [float(item["confidence"]) for item in associated_units]
                ),
                "needs_review": len(associated_values) != 1 or len(nutrients) > 1,
                "source_text": " ".join(str(item["text"]) for item in row_tokens),
            })
    return candidates


def extract_record(
    product_id: str, tokens: list[OCRToken], source_image: str | None = None, confidence_threshold: float = 0.7
) -> NutritionRecord:
    """Create normalized fields from the deterministic OCR baseline."""
    tagged = tag_ocr_tokens(tokens)
    candidates = associate_rows(tagged)
    fields: list[NutritionField] = []
    warnings: list[str] = []
    for candidate in candidates:
        value_candidates = candidate["values"]
        if not value_candidates:
            warnings.append(f"No numeric value associated with {candidate['nutrient']} on row: {candidate['source_text']}")
            continue
        for value_candidate in value_candidates:
            value = value_candidate
            basis = candidate["basis"]
            canonical_value, canonical_unit = canonicalize_measurement(
                float(value["value"]), str(value["unit"]) if value.get("unit") is not None else None
            )
            fields.append(NutritionField(
                nutrient=str(candidate["nutrient"]),
                value=float(value["value"]),
                unit=value.get("unit"),
                basis=str(basis[0]) if basis else None,
                raw_text=str(candidate["source_text"]),
                confidence=float(candidate["confidence"]),
                normalized_value=canonical_value,
                normalized_unit=canonical_unit,
            ))
        if candidate["needs_review"]:
            warnings.append(f"Ambiguous spatial association: {candidate['source_text']}")
    return NutritionRecord(
        product_id=product_id,
        fields=fields,
        source_image=source_image,
        needs_review=bool(warnings) or any(
            field.confidence is not None and field.confidence < confidence_threshold for field in fields
        ),
        warnings=warnings,
    )
