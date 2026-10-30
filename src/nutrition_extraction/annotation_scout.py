"""Rank Open Food Facts records for manual nutrition-image annotation review.

Scores use product-level metadata only to prioritize human review. They do not
assert that image text matches metadata and must never be treated as gold labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_FIELDS = (
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g",
)
PLACEHOLDERS = {"", "undefined", "nan", "none", "null", "n/a", "na"}


def nonempty(series: pd.Series) -> pd.Series:
    """Treat null, blank, and common OFF placeholder values as missing."""
    normalized = series.astype("string").str.strip().str.casefold()
    return series.notna() & ~normalized.isin(PLACEHOLDERS)


def build_annotation_candidates(
    frame: pd.DataFrame,
    verified_product_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank products with nutrition-image URLs and available structured targets.

    Ranking is deterministic and prioritizes target-field coverage, product identity,
    country, and serving-size context. The CSV output is an annotation work queue;
    reviewers must verify every image/value relationship manually.
    """
    if frame.empty:
        raise ValueError("Input sample contains no product records.")
    targets = [field for field in TARGET_FIELDS if field in frame.columns]
    image_column = next((name for name in ("image_nutrition_url", "image_nutrition_small_url") if name in frame.columns), None)
    if image_column is None:
        raise ValueError("No nutrition image URL column is available in this sample.")
    if "code" not in frame.columns:
        raise ValueError("A product identifier column named 'code' is required for annotation review.")

    image_mask = nonempty(frame[image_column])
    if image_column == "image_nutrition_url" and "image_nutrition_small_url" in frame.columns:
        image_mask |= nonempty(frame["image_nutrition_small_url"])
    numeric_masks = {
        field: pd.to_numeric(frame[field], errors="coerce").notna()
        for field in targets
    }
    label_count = (
        pd.concat(numeric_masks, axis=1).sum(axis=1)
        if numeric_masks else pd.Series(0, index=frame.index)
    )
    candidate_mask = image_mask & label_count.gt(0) & nonempty(frame["code"])
    candidates = frame.loc[candidate_mask].copy()
    candidates["target_field_count"] = label_count.loc[candidate_mask].astype(int)
    candidates["target_fields_present"] = [
        ",".join(field for field in targets if numeric_masks[field].loc[index])
        for index in candidates.index
    ]
    candidates["nutrition_image_url"] = candidates[image_column].where(
        nonempty(candidates[image_column]),
        candidates.get("image_nutrition_small_url", pd.Series(index=candidates.index, dtype="string")),
    )
    candidates["nutrition_image_locale_hint"] = candidates["nutrition_image_url"].astype("string").str.extract(
        r"nutrition_([a-z]{2,3})(?:[._])", expand=False
    ).fillna("unknown")
    candidates["has_product_name"] = nonempty(candidates.get("product_name", pd.Series(index=candidates.index, dtype="string"))).astype(int)
    candidates["has_brand"] = nonempty(candidates.get("brands", pd.Series(index=candidates.index, dtype="string"))).astype(int)
    candidates["has_country"] = nonempty(candidates.get("countries", pd.Series(index=candidates.index, dtype="string"))).astype(int)
    candidates["has_serving_size"] = nonempty(candidates.get("serving_size", pd.Series(index=candidates.index, dtype="string"))).astype(int)
    candidates["annotation_priority_score"] = (
        candidates["target_field_count"] / max(len(targets), 1)
        + 0.10 * candidates["has_product_name"]
        + 0.05 * candidates["has_brand"]
        + 0.10 * candidates["has_country"]
        + 0.10 * candidates["has_serving_size"]
    ).round(4)
    verified_product_ids = verified_product_ids or set()
    candidates["alignment_verified"] = candidates["code"].astype(str).isin(verified_product_ids)
    candidates["review_status"] = candidates["alignment_verified"].map(
        {True: "record_level_verified_token_tags_pending", False: "pending_manual_visual_review"}
    )
    candidates["reviewer"] = ""
    candidates["review_notes"] = ""
    candidates = candidates.sort_values(
        ["annotation_priority_score", "target_field_count", "code"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    candidates.insert(0, "review_rank", range(1, len(candidates) + 1))

    summary = {
        "input_rows": int(len(frame)),
        "input_columns": int(frame.shape[1]),
        "nutrition_image_url_candidates": int(image_mask.sum()),
        "ranked_candidates_with_any_target": int(len(candidates)),
        "target_fields_scored": list(targets),
        "candidate_count_by_target_support": {
            f"at_least_{count}_of_{len(targets)}_targets": int((candidates["target_field_count"] >= count).sum())
            for count in (1, 3, 5, 7, len(targets))
        },
        "candidate_locale_hints": candidates["nutrition_image_locale_hint"].value_counts().to_dict(),
        "record_level_alignment_verified_count": int(candidates["alignment_verified"].sum()),
        "token_level_annotations_complete_count": 0,
        "warning": "Ranking uses metadata only and is not confidence. Only explicitly marked product IDs have separate manual record-level verification; all others require visual review.",
    }
    return candidates, summary


def build_diverse_review_batch(
    candidates: pd.DataFrame,
    size: int = 100,
    per_locale: int = 3,
) -> pd.DataFrame:
    """Choose a small locale-diverse pilot, then fill remaining slots by priority.

    Locale is inferred from the URL filename only and is a weak heuristic. Reviewers
    must inspect the actual language and table. The output remains a review queue.
    """
    if size < 1 or per_locale < 1:
        raise ValueError("size and per_locale must be positive")
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values(
        ["annotation_priority_score", "target_field_count", "code"],
        ascending=[False, False, True], kind="stable",
    )
    chosen_indices: list[int] = []
    if "nutrition_image_locale_hint" in ordered:
        locales = ordered["nutrition_image_locale_hint"].value_counts().index.tolist()
        for locale in locales:
            locale_indices = ordered.index[ordered["nutrition_image_locale_hint"].eq(locale)].tolist()
            chosen_indices.extend(locale_indices[:per_locale])
            if len(chosen_indices) >= size:
                break
    chosen_indices = list(dict.fromkeys(chosen_indices))[:size]
    if len(chosen_indices) < size:
        chosen_set = set(chosen_indices)
        chosen_indices.extend(index for index in ordered.index if index not in chosen_set)
        chosen_indices = chosen_indices[:size]
    batch = ordered.loc[chosen_indices].copy()
    batch.insert(0, "pilot_order", range(1, len(batch) + 1))
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/openfoodfacts_eda_enriched.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/annotation_scout"))
    parser.add_argument("--top", type=int, default=500, help="Number of ranked candidates to export for a manageable review batch")
    parser.add_argument("--pilot-size", type=int, default=100, help="Size of locale-diverse annotation pilot")
    parser.add_argument("--per-locale", type=int, default=3, help="Minimum pilot slots per URL-language hint before filling by score")
    args = parser.parse_args()
    if args.top < 1:
        raise ValueError("--top must be a positive integer")
    frame = pd.read_csv(args.input, low_memory=False)
    gold_path = args.input.parent / "evaluation" / "gold_records.jsonl"
    verified_ids: set[str] = set()
    if gold_path.exists():
        verified_ids = {
            str(json.loads(line)["product_id"])
            for line in gold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    candidates, summary = build_annotation_candidates(frame, verified_product_ids=verified_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    candidates.head(args.top).to_csv(args.output_dir / "annotation_review_queue.csv", index=False)
    pilot = build_diverse_review_batch(candidates, size=args.pilot_size, per_locale=args.per_locale)
    pilot.to_csv(args.output_dir / "annotation_pilot_batch.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote top {min(args.top, len(candidates))} records to {args.output_dir / 'annotation_review_queue.csv'}")


if __name__ == "__main__":
    main()
