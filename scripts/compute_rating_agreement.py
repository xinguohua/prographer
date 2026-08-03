"""Compute interval Krippendorff's alpha for human rating CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def krippendorff_alpha_interval(rows: list[list[float]]) -> float:
    values = [v for row in rows for v in row if v is not None]
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    de = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if de == 0:
        return 1.0

    observed = []
    for row in rows:
        vals = [v for v in row if v is not None]
        n = len(vals)
        if n < 2:
            continue
        observed.append(sum((a - b) ** 2 for a in vals for b in vals) / (n * (n - 1)))
    if not observed:
        return 1.0
    do = sum(observed) / len(observed)
    return 1.0 - (do / (2.0 * de))


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ratings_matrix(rows: list[dict]) -> list[list[float]]:
    return [[float(r["R1"]), float(r["R2"]), float(r["R3"])] for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=Path("data/human_ratings.csv"))
    args = parser.parse_args()
    rows = load_rows(args.ratings)

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["Condition"]].append(row)

    payload = {
        "rows": len(rows),
        "overall_alpha_interval": round(krippendorff_alpha_interval(ratings_matrix(rows)), 4),
        "condition_alpha_interval": {
            cond: round(krippendorff_alpha_interval(ratings_matrix(sub)), 4)
            for cond, sub in sorted(by_cond.items())
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
