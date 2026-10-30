"""Create reproducible product-group-aware splits from labeled JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import GroupShuffleSplit


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records with line-numbered error messages."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
    return rows


def make_splits(
    records: list[dict[str, Any]],
    group_key: str = "product_id",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split without allowing records in the same group to leak across partitions."""
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 or value >= 1 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test fractions must be positive and sum to 1.")
    if len(records) < 3:
        raise ValueError("At least three annotated records are required to create three splits.")
    groups = [record.get(group_key) for record in records]
    if any(group is None or str(group).strip() == "" for group in groups):
        raise ValueError(f"Every record must have a non-empty {group_key!r} grouping value.")
    if len(set(groups)) < 3:
        raise ValueError("At least three distinct groups are required for leakage-safe three-way splits.")

    indices = list(range(len(records)))
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    remaining_indices, test_indices = next(test_splitter.split(indices, groups=groups))
    remaining_groups = [groups[index] for index in remaining_indices]
    validation_relative_fraction = validation_fraction / (train_fraction + validation_fraction)
    validation_splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_relative_fraction, random_state=seed + 1
    )
    train_local, validation_local = next(
        validation_splitter.split(remaining_indices, groups=remaining_groups)
    )
    train_indices = [remaining_indices[index] for index in train_local]
    validation_indices = [remaining_indices[index] for index in validation_local]

    return {
        "train": [records[index] for index in train_indices],
        "validation": [records[index] for index in validation_indices],
        "test": [records[index] for index in test_indices],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Labeled annotation JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-key", default="product_id")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits = make_splits(
        read_jsonl(args.input), args.group_key, args.train_fraction,
        args.validation_fraction, args.test_fraction, args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        destination = args.output_dir / f"{split_name}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(records)} records -> {destination}")


if __name__ == "__main__":
    main()
