"""Training extension entrypoint: intentionally no model or weight downloads.

The supervised LayoutLM-style training workflow is specified in
architecture.md as pseudocode. This module does not start training.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-design", action="store_true", help="Print where the pseudocode design is documented.")
    parser.parse_args()
    print("No model training was run and no pretrained weights were downloaded.")
    print("See part2/architecture.md for the clearly labeled LayoutLM-style training pseudocode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())