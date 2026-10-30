"""Rule-based nutrition entity and spatial association baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from part2.normalization import ParsedValue, normalize_text, nutrition_basis, parse_value_unit
from part2.schemas import ExtractionResult, NutrientValue, OCRDetection


NUTRIENT_ALIASES: dict[str, tuple[str, ...]] = {
    "fat": ("total fat", "total fats", "fat", "fats", "lipides", "matiere grasse", "matieres grasses", "fett", "vetten", "grassi", "grasas"),
    "saturated_fat": ("of which saturates", "saturated fat", "saturated fats", "saturates", "saturate", "acides gras satures", "graisses saturees", "saturierte fettsauren", "verzadigde vetzuren"),
    "carbohydrates": ("carbohydrates", "carbohydrate", "glucides", "kohlenhydrate", "hidratos de carbono", "carboidrati", "koolhydraten"),
    "sugars": ("of which sugars", "sugars", "sugar", "sucres", "zucker", "azucares", "zuccheri", "suikers"),
    "proteins": ("proteins", "protein", "proteines", "eiweiss", "eiwit", "proteina", "proteine"),
    "salt": ("salt", "sel", "salz", "zout", "sale"),
    "sodium": ("sodium", "natrium", "sodio"),
    "fiber": ("dietary fiber", "dietary fibre", "fiber", "fibre", "fibres alimentaires", "ballaststoffe", "fibras"),
    "serving_size": ("serving size", "portion size", "serving", "portion", "porzione", "portionen"),
    "added_sugars": ("added sugars", "added sugar", "sucres ajoutes"),
    "calcium": ("calcium",),
    "cholesterol": ("cholesterol", "cholesterole"),
    "iron": ("iron", "fer", "eisen", "hierro"),
    "potassium": ("potassium", "kalium", "potasio"),
    "trans_fat": ("trans fat", "trans fats", "acides gras trans"),
    "vitamin_d": ("vitamin d", "vitamine d", "vitamin d3"),
    "energy": ("energy", "energie", "energetic value", "energy value", "calories", "calorie", "calorific value", "valeur energetique", "brennwert", "valore energetico"),
}
_ALIASES_SORTED = sorted(
    ((normalize_text(alias), nutrient) for nutrient, aliases in NUTRIENT_ALIASES.items() for alias in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
_VALUE_UNIT_ONLY = re.compile(r"^(kcal|kj|mg|g|ml|l|mcg|ug|µg|%)$", re.IGNORECASE)


@dataclass(frozen=True)
class NutrientMention:
    nutrient: str
    detection: OCRDetection
    matched_alias: str


@dataclass(frozen=True)
class ValueMention:
    parsed: ParsedValue
    detections: tuple[OCRDetection, ...]
    center_x: float
    center_y: float


def match_nutrient_name(text: str) -> tuple[str, str] | None:
    """Return canonical nutrient and alias for a conservative whole-phrase match."""
    normalized = normalize_text(text)
    normalized = re.sub(r"[^a-z0-9%]+", " ", normalized).strip()
    for alias, nutrient in _ALIASES_SORTED:
        if normalized == alias or normalized.startswith(alias + " ") or normalized.startswith(alias + " :"):
            return nutrient, alias
    return None


def _center(detection: OCRDetection) -> tuple[float, float]:
    return (
        sum(point[0] for point in detection.bbox) / len(detection.bbox),
        sum(point[1] for point in detection.bbox) / len(detection.bbox),
    )


def _row_groups(detections: Iterable[OCRDetection]) -> list[list[OCRDetection]]:
    items = list(detections)
    if not items:
        return []
    heights = sorted(max(1, max(p[1] for p in item.bbox) - min(p[1] for p in item.bbox)) for item in items)
    tolerance = max(8, heights[len(heights) // 2] * 0.65)
    ordered = sorted(items, key=lambda item: (_center(item)[1], _center(item)[0]))
    rows: list[list[OCRDetection]] = []
    centers: list[float] = []
    for item in ordered:
        y = _center(item)[1]
        match = next((i for i in range(len(rows) - 1, -1, -1) if abs(centers[i] - y) <= tolerance), None)
        if match is None:
            rows.append([item])
            centers.append(y)
        else:
            rows[match].append(item)
            centers[match] = sum(_center(member)[1] for member in rows[match]) / len(rows[match])
    return [sorted(row, key=lambda item: _center(item)[0]) for row in rows]


def _basis_anchors(detections: list[OCRDetection]) -> list[tuple[str, float, float, str]]:
    anchors: list[tuple[str, float, float, str]] = []
    for row in _row_groups(detections):
        for detection in row:
            basis = nutrition_basis(detection.text)
            if basis:
                center_x, center_y = _center(detection)
                anchors.append((basis, center_x, center_y, detection.text))
        # OCR often splits a header like "100 g" into separate boxes. Rejoin
        # only short adjacent windows on the same visual row.
        for start in range(len(row)):
            for width in (2, 3):
                window = row[start : start + width]
                if len(window) != width:
                    continue
                left = max(point[0] for point in window[0].bbox)
                right = min(point[0] for point in window[1].bbox)
                if right - left > 100:
                    continue
                text = " ".join(item.text for item in window)
                basis = nutrition_basis(text)
                if basis:
                    centers = [_center(item) for item in window]
                    anchors.append((basis, sum(point[0] for point in centers) / len(centers), sum(point[1] for point in centers) / len(centers), text))
    # Merge repeated header tokens of the same basis that describe the same column.
    unique: list[tuple[str, float, float, str]] = []
    for item in sorted(anchors, key=lambda entry: (entry[0], entry[2], entry[1])):
        if not any(item[0] == prev[0] and abs(item[1] - prev[1]) < 30 and abs(item[2] - prev[2]) < 24 for prev in unique):
            unique.append(item)
    return unique


def _basis_for_value(
    value: ValueMention,
    anchors: list[tuple[str, float, float, str]],
    image_width: int,
) -> tuple[str, bool]:
    if not anchors:
        return "unspecified", True
    # Basis headers at clearly separated y positions generally start a new
    # vertical section; never apply a serving header backward to rows above it.
    y_centers = sorted(item[2] for item in anchors)
    y_gap = max(30, int(image_width * 0.12))
    distinct_header_rows: list[float] = []
    for y_center in y_centers:
        if not distinct_header_rows or abs(y_center - distinct_header_rows[-1]) > y_gap:
            distinct_header_rows.append(y_center)
    if len(distinct_header_rows) > 1:
        preceding_rows = [y for y in distinct_header_rows if y <= value.center_y]
        if not preceding_rows:
            return "unspecified", True
        active_y = preceding_rows[-1]
        section_anchors = [item for item in anchors if abs(item[2] - active_y) <= y_gap]
        section_types = {item[0] for item in section_anchors}
        if len(section_types) == 1:
            return next(iter(section_types)), False
        anchors = section_anchors
    else:
        # A single detected header only applies to rows below it. Earlier rows
        # may be another basis whose header OCR missed; preserve as unspecified.
        if value.center_y < anchors[0][2]:
            return "unspecified", True
        basis_types = {item[0] for item in anchors}
        if len(basis_types) == 1:
            return next(iter(basis_types)), False
    distances = sorted((abs(value.center_x - x), basis, text) for basis, x, _, text in anchors)
    if len(distances) > 1 and abs(distances[0][0] - distances[1][0]) < max(12, image_width * 0.03):
        return "unspecified", True
    if distances[0][0] > image_width * 0.45:
        return "unspecified", True
    return distances[0][1], False


def _value_mentions(row: list[OCRDetection]) -> list[ValueMention]:
    values: list[ValueMention] = []
    ordered = sorted(row, key=lambda item: _center(item)[0])
    for index, detection in enumerate(ordered):
        parsed = parse_value_unit(detection.text)
        if parsed is None:
            continue
        used = [detection]
        unit = parsed.unit
        right_edge = max(point[0] for point in detection.bbox)
        if unit is None:
            candidates = []
            for other in ordered[index + 1 :]:
                left_edge = min(point[0] for point in other.bbox)
                if left_edge - right_edge > 90:
                    break
                if _VALUE_UNIT_ONLY.fullmatch(normalize_text(other.text).strip()):
                    candidates.append(other)
                elif parse_value_unit(other.text) is not None:
                    break
            if len(candidates) == 1:
                unit_detection = candidates[0]
                parsed = ParsedValue(parsed.value, normalize_text(unit_detection.text).strip(), f"{parsed.raw_text} {unit_detection.text}", parsed.qualifier)
                used.append(unit_detection)
        centers = [_center(item) for item in used]
        values.append(
            ValueMention(parsed, tuple(used), sum(p[0] for p in centers) / len(centers), sum(p[1] for p in centers) / len(centers))
        )
    return values


def extract_nutrients(
    detections: list[OCRDetection],
    *,
    barcode: str | None,
    image_path: str,
    image_width: int,
) -> ExtractionResult:
    """Associate row-aligned labels with values; ambiguity is kept unresolved."""
    result = ExtractionResult(barcode=barcode, image_path=image_path)
    anchors = _basis_anchors(detections)
    basis_types = sorted({basis for basis, _, _, _ in anchors})
    result.nutrition_basis = basis_types[0] if len(basis_types) == 1 else None
    mentions: list[NutrientMention] = []
    for detection in detections:
        match = match_nutrient_name(detection.text)
        if match:
            mentions.append(NutrientMention(match[0], detection, match[1]))
    rows = _row_groups(detections)
    row_of: dict[int, list[OCRDetection]] = {id(item): row for row in rows for item in row}

    for mention in mentions:
        row = row_of[id(mention.detection)]
        x0 = min(point[0] for point in mention.detection.bbox)
        next_labels = [
            min(point[0] for point in other.detection.bbox)
            for other in mentions
            if other is not mention and row_of.get(id(other.detection)) is row
            and min(point[0] for point in other.detection.bbox) > x0
        ]
        right_limit = min(next_labels) if next_labels else image_width
        candidates = [
            value for value in _value_mentions(row)
            if value.center_x > x0 and value.center_x < right_limit
            and all(item is not mention.detection for item in value.detections)
        ]
        # A same-token value (e.g. "Energy 120 kcal") is supported too.
        inline = parse_value_unit(mention.detection.text)
        if inline is not None:
            candidates.append(ValueMention(inline, (mention.detection,), _center(mention.detection)[0], _center(mention.detection)[1]))

        if not candidates:
            result.unresolved_fields.append({"nutrient": mention.nutrient, "reason": "no numeric value on the same visual row", "source_text": " ".join(item.text for item in row)})
            continue

        # Numeric candidates with no explicit unit remain unresolved. Energy is
        # split by unit; other fields require an unambiguous single value or
        # explicit distinct basis columns.
        usable: list[tuple[ValueMention, str, bool]] = []
        for candidate in candidates:
            if mention.nutrient == "energy":
                if candidate.parsed.unit == "kcal":
                    nutrient_key = "energy_kcal"
                elif candidate.parsed.unit == "kj":
                    nutrient_key = "energy_kj"
                else:
                    result.unresolved_fields.append({"nutrient": "energy", "reason": "energy number has no explicit kcal/kJ unit", "raw_text": candidate.parsed.raw_text})
                    continue
            else:
                nutrient_key = mention.nutrient
                if candidate.parsed.unit in {"kcal", "kj"}:
                    result.unresolved_fields.append({"nutrient": nutrient_key, "reason": "energy unit attached to non-energy entity", "raw_text": candidate.parsed.raw_text})
                    continue
            basis, basis_ambiguous = _basis_for_value(candidate, anchors, image_width)
            usable.append((candidate, nutrient_key, basis_ambiguous))

        if not usable:
            continue
        if len(usable) > 1:
            # Multiple energy values can be a valid kcal+kJ pair. Other repeats
            # are assigned only when each has a distinct, explicit basis column.
            bases = [_basis_for_value(candidate, anchors, image_width)[0] for candidate, _, _ in usable]
            energy_pair = mention.nutrient == "energy" and {candidate.parsed.unit for candidate, _, _ in usable} == {"kcal", "kj"}
            distinct_basis = len(set(bases)) == len(bases) and "unspecified" not in bases and len(anchors) > 1
            if not energy_pair and not distinct_basis:
                result.unresolved_fields.append({"nutrient": mention.nutrient, "reason": "multiple plausible values on the row; association is ambiguous", "raw_text": " | ".join(item.parsed.raw_text for item, _, _ in usable)})
                continue

        for candidate, nutrient_key, basis_ambiguous in usable:
            basis, _ = _basis_for_value(candidate, anchors, image_width)
            if basis_ambiguous and basis == "unspecified" and len(basis_types) > 1:
                result.unresolved_fields.append({"nutrient": nutrient_key, "reason": "could not assign value to a nutrition-basis column", "raw_text": candidate.parsed.raw_text})
                continue
            normalized_value, normalized_unit = (candidate.parsed.value, candidate.parsed.unit)
            # Energy uses the explicit unit to select its field; preserve its
            # original value/unit and include canonical kcal representation.
            from part2.normalization import canonicalize

            normalized_value, normalized_unit = canonicalize(candidate.parsed.value, candidate.parsed.unit)
            record_basis = basis if basis in {"per_100g", "per_100ml", "per_serving"} else "unspecified"
            result.nutrients.setdefault(record_basis, {}).setdefault(nutrient_key, []).append(
                NutrientValue(
                    value=candidate.parsed.value,
                    unit=candidate.parsed.unit,
                    basis=record_basis,
                    raw_text=candidate.parsed.raw_text,
                    ocr_confidence=min(item.confidence for item in (mention.detection, *candidate.detections)),
                    normalized_value=normalized_value,
                    normalized_unit=normalized_unit,
                )
            )
            if record_basis == "unspecified":
                result.warnings.append("Nutrition basis was not confidently identified; value retained as unspecified.")
    # Duplicate predictions are not silently deduplicated; mark possible overlap.
    for basis, nutrient_map in result.nutrients.items():
        for nutrient, values in nutrient_map.items():
            if len(values) > 1:
                result.warnings.append(f"Multiple {nutrient} candidates for {basis}; review before use.")
    result.warnings.append("OCR confidence is not calibrated confidence in nutrient-value association.")
    return result