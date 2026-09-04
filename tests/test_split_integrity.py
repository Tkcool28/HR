"""
test_split_integrity.py

Verify the data split contract:
  - train ∩ val = ∅
  - val ∩ holdout = ∅
  - train ∪ val ∪ holdout = full (every game in game.parquet is in exactly one split)
  - 2025 only in holdout_ids
  - pa.parquet has >= 5,000,000 rows
  - Every batter_hand and pitcher_hand is in {L, R, S}
  - Every game_pk in pa.parquet has a matching row in game.parquet

Run: python3 /workspace/hr_model/tests/test_split_integrity.py
"""

import os
import sys
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/workspace/hr_model"
CUR = os.path.join(ROOT, "data/curated")
SPL = os.path.join(ROOT, "data/splits")

REQUIRED = {
    os.path.join(CUR, "pa.parquet"): 5_000_000,
    os.path.join(CUR, "game.parquet"): 1,
    os.path.join(CUR, "roster.parquet"): 1,
    os.path.join(SPL, "train_ids.parquet"): 1,
    os.path.join(SPL, "val_ids.parquet"): 1,
    os.path.join(SPL, "holdout_ids.parquet"): 1,
}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ok: {msg}")


def main() -> None:
    # 1) File presence + minimum row counts
    for path, min_rows in REQUIRED.items():
        if not os.path.exists(path):
            fail(f"missing file: {path}")
        md = pq.read_metadata(path)
        n = md.num_rows
        if n < min_rows:
            fail(f"{path} has {n} rows, need >= {min_rows}")
        ok(f"{os.path.basename(path)} has {n:,} rows (>= {min_rows:,})")

    # 2) Load the splits and game table
    games = pd.read_parquet(os.path.join(CUR, "game.parquet"))
    train = pd.read_parquet(os.path.join(SPL, "train_ids.parquet"))
    val = pd.read_parquet(os.path.join(SPL, "val_ids.parquet"))
    hold = pd.read_parquet(os.path.join(SPL, "holdout_ids.parquet"))

    train_set = set(int(x) for x in train["game_pk"])
    val_set = set(int(x) for x in val["game_pk"])
    hold_set = set(int(x) for x in hold["game_pk"])
    game_set = set(int(x) for x in games["game_pk"])

    # 3) Disjointness
    if train_set & val_set:
        fail(f"train ∩ val = {len(train_set & val_set)} games")
    ok("train ∩ val = ∅")
    if val_set & hold_set:
        fail(f"val ∩ holdout = {len(val_set & hold_set)} games")
    ok("val ∩ holdout = ∅")
    if train_set & hold_set:
        fail(f"train ∩ holdout = {len(train_set & hold_set)} games")
    ok("train ∩ holdout = ∅")

    # 4) Union covers all games
    union = train_set | val_set | hold_set
    missing = game_set - union
    extra = union - game_set
    if missing:
        fail(f"{len(missing)} game_pks in game.parquet are NOT in any split (first 5: {sorted(missing)[:5]})")
    if extra:
        fail(f"{len(extra)} game_pks in a split are NOT in game.parquet (first 5: {sorted(extra)[:5]})")
    ok(f"train ∪ val ∪ holdout == all games ({len(union):,} == {len(game_set):,})")

    # 5) Year assignment correctness
    games_year = games.copy()
    games_year["year"] = pd.to_datetime(games_year["game_date"]).dt.year
    gmap = dict(zip(games_year["game_pk"].astype("int64"), games_year["year"]))

    for sname, sset in [("train", train_set), ("val", val_set), ("holdout", hold_set)]:
        wrong = [gpk for gpk in sset if gpk in gmap and gmap[gpk] not in
                 {2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022} if sname == "train"]
        wrong = []
        for gpk in sset:
            if gpk not in gmap:
                continue
            y = gmap[gpk]
            if sname == "train" and not (2015 <= y <= 2022):
                wrong.append((gpk, y))
            elif sname == "val" and not (2023 <= y <= 2024):
                wrong.append((gpk, y))
            elif sname == "holdout" and y != 2025:
                wrong.append((gpk, y))
        if wrong:
            fail(f"{sname} split has {len(wrong)} games outside expected years (first 3: {wrong[:3]})")
        ok(f"all {sname} games are in expected year range")

    # 6) PA row count check + handedness
    pa = pd.read_parquet(os.path.join(CUR, "pa.parquet"), columns=["game_pk", "batter_hand", "pitcher_hand"])
    bh = set(pa["batter_hand"].dropna().unique())
    ph = set(pa["pitcher_hand"].dropna().unique())
    if not bh.issubset({"L", "R", "S"}):
        fail(f"batter_hand has unexpected values: {bh - {'L','R','S'}}")
    if not ph.issubset({"L", "R", "S"}):
        fail(f"pitcher_hand has unexpected values: {ph - {'L','R','S'}}")
    if pa["batter_hand"].isna().any() or pa["pitcher_hand"].isna().any():
        fail("null batter_hand or pitcher_hand present")
    ok(f"batter_hand values: {sorted(bh)}")
    ok(f"pitcher_hand values: {sorted(ph)}")

    # 7) Every pa.game_pk in game.parquet
    pa_gpks = set(int(x) for x in pa["game_pk"].unique())
    missing_in_games = pa_gpks - game_set
    if missing_in_games:
        fail(f"{len(missing_in_games)} pa.game_pks not in game.parquet (first 5: {sorted(missing_in_games)[:5]})")
    ok("every pa.game_pk has a matching game.parquet row")

    print("\nALL SPLIT INTEGRITY CHECKS PASSED")
    print(f"  games total: {len(game_set):,}")
    print(f"  train: {len(train_set):,}")
    print(f"  val: {len(val_set):,}")
    print(f"  holdout: {len(hold_set):,}")
    print(f"  pa rows: {len(pa):,}")


if __name__ == "__main__":
    main()
