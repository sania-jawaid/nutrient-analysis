from nutrition_extraction.splitting import make_splits


def test_group_never_leaks_between_splits() -> None:
    records = [{"product_id": f"product-{index}", "image_path": f"{index}.jpg"} for index in range(30)]
    splits = make_splits(records, seed=7)
    group_sets = {name: {row["product_id"] for row in values} for name, values in splits.items()}
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    assert sum(map(len, splits.values())) == len(records)


def test_same_product_group_stays_together() -> None:
    records = [{"product_id": f"product-{i // 2}", "image_path": f"{i}.jpg"} for i in range(30)]
    splits = make_splits(records, seed=7)
    product_locations: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        for row in rows:
            product_locations.setdefault(row["product_id"], set()).add(split_name)
    assert all(len(locations) == 1 for locations in product_locations.values())
