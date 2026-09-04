"""Run the exact trusted aggressive trainer on the 62-feature challenger.

This is intentionally a thin wrapper around train_v12_aggressive_candidate so
training methodology cannot drift between the 73-feature champion and the
62-feature challenger. Only the feature-list contract and output locations are
changed.

2025 is not read.
"""
from __future__ import annotations

import os

import trusted_v12.train_v12_aggressive_candidate as base

ROOT = "/workspace/hr_model"

base.FEAT_DIR = os.path.join(ROOT, "features/v1.2_pruned62")
# Feature values remain in the trusted 73-feature matrix; the 62-feature list
# determines which columns are consumed.
base.FEAT = os.path.join(ROOT, "features/v1.2_trusted/game_features.parquet")
base.FLIST = os.path.join(base.FEAT_DIR, "feature_list.json")
base.OUT = os.path.join(ROOT, "models/v1.2_pruned62")
base.REPORTS = os.path.join(ROOT, "reports/v1.2_pruned62")
base.TRIALS = os.path.join(base.OUT, "xgb_trials.csv")


if __name__ == "__main__":
    base.main()
