"""Materialize the 62-feature challenger contract without touching feature values.

The challenger removes only the two passenger groups identified by the frozen
feature-ablation/pruning program:
  * nine raw pitcher pitch-usage proportions
  * batter season/career xwOBA-on-contact

The underlying trusted 73-feature parquet is not rewritten. Only a separate
feature-list contract is produced, which makes this a pure feature-selection
experiment.

2025 is not read.
"""
from __future__ import annotations

import json
import os

ROOT = "/workspace/hr_model"
SRC_LIST = os.path.join(ROOT, "features/v1.2_trusted/feature_list.json")
OUT_DIR = os.path.join(ROOT, "features/v1.2_pruned62")
OUT_LIST = os.path.join(OUT_DIR, "feature_list.json")
OUT_SUMMARY = os.path.join(OUT_DIR, "_summary.json")

PITCH_TYPES = ("FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS")
REMOVE = {f"pitcher_usage_{pt}_30d" for pt in PITCH_TYPES} | {
    "batter_xwoba_on_contact_season",
    "batter_xwoba_on_contact_career",
}


def main() -> None:
    with open(SRC_LIST) as fp:
        source = json.load(fp)
    if len(source) != 73:
        raise RuntimeError(f"expected frozen 73-feature source, got {len(source)}")

    missing = sorted(REMOVE - set(source))
    if missing:
        raise RuntimeError(f"expected prune features missing from source: {missing}")

    keep = [c for c in source if c not in REMOVE]
    if len(keep) != 62:
        raise RuntimeError(f"expected 62-feature challenger, got {len(keep)}")
    if set(keep) & REMOVE:
        raise RuntimeError("pruned feature leaked into challenger list")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_LIST, "w") as fp:
        json.dump(keep, fp, indent=2)
    with open(OUT_SUMMARY, "w") as fp:
        json.dump(
            {
                "source_feature_count": len(source),
                "challenger_feature_count": len(keep),
                "removed_feature_count": len(REMOVE),
                "removed_features": sorted(REMOVE),
                "holdout_2025_read": False,
                "purpose": "aggressive-retune challenger after frozen ablation/pruning screen",
            },
            fp,
            indent=2,
        )
    print(f"[pruned62] wrote {len(keep)}-feature challenger contract; removed {len(REMOVE)}")
    print("[pruned62] 2025 NOT READ")


if __name__ == "__main__":
    main()
