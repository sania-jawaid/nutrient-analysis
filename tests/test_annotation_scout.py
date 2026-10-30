import pandas as pd

from nutrition_extraction.annotation_scout import build_annotation_candidates


def test_scout_ranks_fuller_candidates_and_never_marks_alignment_verified() -> None:
    frame = pd.DataFrame({
        "code": ["a", "b", "c"],
        "product_name": ["A", "B", None],
        "brands": ["Brand A", None, None],
        "countries": ["France", None, None],
        "serving_size": ["90 g", None, None],
        "image_nutrition_url": ["https://x/nutrition_fr.1.jpg", "https://x/nutrition_en.1.jpg", ""],
        "energy-kcal_100g": [10, 20, None],
        "fat_100g": [1, None, None],
        "proteins_100g": [2, None, None],
    })
    candidates, summary = build_annotation_candidates(frame)
    assert candidates["code"].tolist() == ["a", "b"]
    assert candidates.iloc[0]["target_field_count"] == 3
    assert candidates["alignment_verified"].eq(False).all()
    assert summary["ranked_candidates_with_any_target"] == 2
    assert "fr" in summary["candidate_locale_hints"]
