#!/usr/bin/env python3
"""Create a balanced target table for conditioned self-consistency sampling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--per-label", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.per_label <= 0:
        parser.error("--per-label must be positive")
    with args.dataset.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"sequence_align", args.label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {args.dataset}: {sorted(missing)}")
        representatives = {}
        for row in reader:
            label = row[args.label_column].strip()
            if label:
                representatives.setdefault(label, row["sequence_align"])

    labels = sorted(representatives)
    if len(labels) < 2:
        raise ValueError(f"Expected at least two labels in column {args.label_column}")

    rows = []
    for label in labels:
        # annaDCA uses this sequence only to establish alignment dimensions and
        # alphabet. sample_conditioned initializes every visible chain randomly.
        sequence = representatives[label]
        rows.extend(
            {
                "header": f"target_{label}_{index}",
                "sequence_align": sequence,
                "target_label": label,
            }
            for index in range(args.per_label)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["header", "sequence_align", "target_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} balanced targets ({args.per_label} per label) to {args.output}")


if __name__ == "__main__":
    main()
