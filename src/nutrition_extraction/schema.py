"""Typed nutrition extraction output and schema validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EntityType = Literal["NUTRIENT", "VALUE", "UNIT", "BASIS"]


@dataclass(frozen=True)
class NutritionField:
    """One normalized nutrient result, retaining source text and confidence."""

    nutrient: str
    value: float | None
    unit: str | None
    basis: str | None = None
    raw_text: str | None = None
    confidence: float | None = None
    normalized_value: float | None = None
    normalized_unit: str | None = None


@dataclass
class NutritionRecord:
    """All extracted values from one nutrition panel."""

    product_id: str
    fields: list[NutritionField] = field(default_factory=list)
    source_image: str | None = None
    needs_review: bool = False
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible primitives."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NutritionRecord:
        """Validate the required fields when consuming an inference result."""
        required = {"product_id", "fields"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"NutritionRecord missing required keys: {sorted(missing)}")
        fields = [NutritionField(**item) for item in payload["fields"]]
        return cls(
            product_id=str(payload["product_id"]),
            fields=fields,
            source_image=payload.get("source_image"),
            needs_review=bool(payload.get("needs_review", False)),
            warnings=list(payload.get("warnings", [])),
            schema_version=str(payload.get("schema_version", "1.0")),
        )
