"""Conservative parsing and normalization for OCR-read nutrient values."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


NUMBER_UNIT_RE = re.compile(
    r"(?P<qualifier><|>|≤|≥)?\s*(?P<number>[+-]?(?:\d[\d\s\u00a0\u202f.,]*|[.,]\d+))\s*"
    r"(?P<unit>kcal|kilocalories?|kj|kilojoules?|mg|mcg|μg|µg|ug|g|ml|mL|l|%)?",
    re.IGNORECASE,
)
UNIT_RE = re.compile(r"^(kcal|kj|mg|mcg|μg|µg|ug|g|ml|l|%)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedValue:
    value: float
    unit: str | None
    raw_text: str
    qualifier: str | None = None


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def parse_number(text: str) -> float | None:
    """Parse decimal comma/dot and common thousands grouping; reject junk."""
    value = text.strip().replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if not value:
        return None
    if "," in value and "." in value:
        # Last punctuation is treated as decimal; the other is a thousands separator.
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        value = value.replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def parse_value_unit(text: str) -> ParsedValue | None:
    """Parse a single value and optional unit from one OCR detection."""
    match = NUMBER_UNIT_RE.search(text)
    if not match:
        return None
    value = parse_number(match.group("number"))
    if value is None:
        return None
    unit = normalize_unit(match.group("unit"))
    raw = match.group(0).strip()
    return ParsedValue(value=value, unit=unit, raw_text=raw, qualifier=match.group("qualifier"))


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    value = normalize_text(unit).replace("μ", "µ").strip().casefold()
    aliases = {
        "kilocalorie": "kcal", "kilocalories": "kcal", "calories": "kcal",
        "kilojoule": "kj", "kilojoules": "kj", "µg": "ug", "μg": "ug", "mcg": "ug",
        "milligram": "mg", "milligrams": "mg", "gram": "g", "grams": "g",
        "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
        "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    }
    return aliases.get(value, value)


def canonicalize(value: float, unit: str | None) -> tuple[float, str | None]:
    """Convert common units to kcal, g, ml; retain unknown/no units explicitly."""
    normalized = normalize_unit(unit)
    if normalized == "kj":
        return value / 4.184, "kcal"
    if normalized == "mg":
        return value / 1000, "g"
    if normalized == "ug":
        return value / 1_000_000, "g"
    if normalized == "l":
        return value * 1000, "ml"
    return value, normalized


def nutrition_basis(text: str) -> str | None:
    normalized = normalize_text(text).replace(" ", "")
    if re.search(r"(?:per|pour|par|pro)?100g(?:ram)?", normalized):
        return "per_100g"
    if re.search(r"(?:per|pour|par|pro)?100ml", normalized):
        return "per_100ml"
    if re.search(r"serving|portion|portie|porzione|portionnement", normalized):
        return "per_serving"
    return None