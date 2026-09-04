"""Test: no-lookahead contract on game_features.parquet (v1.1).

The hard invariant: for every row, max_as_of_date (the latest date any
feature was sourced from) must be strictly less than the row's
game_date. This must hold even for holdout rows, where the upper bound
is the holdout cap (2024-12-31) rather than the row's own game_date.

v1.1 changes:
  - Reads features/v1.1/game_features.parquet
  - Updates UNAVAILABLE set to reflect v1.1's filled QoC + park features
  - park_hr_factor_by_hand is now real (no longer in UNAVAILABLE)
  - The 21 new features (barrel, xwOBA, EV, LA, hard-hit, FB, ISO +
    park_hr_factor_3yr, park_hr_factor_by_hand) must all be <5% null
    on train.
"""
import os
import sys
import pandas as pd
import numpy as np

ROOT = "/workspace/hr_model"
FEAT = os.path.join(ROOT, "features/v1.1/game_features.parquet")
HOLDOUT_CAP = pd.Timestamp("2024-12-31")


def main() -> int:
    f = pd.read_parquet(FEAT)
    print(f"loaded {len(f):,} rows, {f.shape[1]} cols from {FEAT}")

    asof_cols = [c for c in f.columns if c.endswith("_as_of")]
    print(f"  {len(asof_cols)} _as_of columns present")

    # Invariant 1: every row's max_as_of_date is strictly less than its
    # game_date.
    bad = f[f["max_as_of_date"] >= f["game_date"]]
    if len(bad) > 0:
        print(f"FAIL: {len(bad)} rows have max_as_of_date >= game_date", file=sys.stderr)
        print(bad.head().to_string(), file=sys.stderr)
        return 1
    print(f"PASS: all {len(f):,} rows have max_as_of_date < game_date")

    # Invariant 2: holdout rows' max_as_of_date is at or before the
    # holdout cap (2024-12-31) — i.e. no 2025 PAs leaked into the
    # holdout feature computation.
    hold = f[f["split"] == "holdout"]
    bad_hold = hold[hold["max_as_of_date"] > HOLDOUT_CAP]
    if len(bad_hold) > 0:
        print(f"FAIL: {len(bad_hold)} holdout rows have max_as_of_date > {HOLDOUT_CAP.date()}",
              file=sys.stderr)
        return 1
    print(f"PASS: all {len(hold):,} holdout rows have max_as_of_date <= {HOLDOUT_CAP.date()}")

    # Invariant 3: target hr_in_game is null for all holdout rows.
    if hold["hr_in_game"].notna().any():
        n = int(hold["hr_in_game"].notna().sum())
        print(f"FAIL: {n} holdout rows have non-null hr_in_game (target leaked)",
              file=sys.stderr)
        return 1
    print(f"PASS: all {len(hold):,} holdout rows have hr_in_game = NaN")

    # Invariant 4: per-feature null rate on train < 5% for the columns
    # the trainer will use.
    # In v1.1, the 16 v1 placeholder features with no v1.1 equivalent
    # are NO LONGER in the active feature list. The only columns that
    # could be 100% null are columns we explicitly didn't include
    # (e.g. batter_pull_pct_30d, which was dropped).
    UNAVAILABLE_V11 = {
        # weather (no source)
        "humidity_pct", "wind_component_out_mph", "temp_f",
        "roof_closed", "weather_hr_multiplier", "wind_speed_mph",
        # game context (not in savant)
        "day_or_night", "batting_order_slot",
        # advanced pitcher stats (no source)
        "fip_season", "xfip_season", "xera_season",
        "stuff_plus_season", "location_plus_season",
        # advanced BIP-derived (computed elsewhere, no source for these)
        "xiso_30d", "xslg_30d", "sweet_spot_pct_30d",
        "avg_ev_allowed_fl_30d", "avg_flyball_distance_30d",
        "bvp_avg", "form_xwoba_gap_30d",
        "opp_bullpen_quality_proxy_30d",
        # removed in v1.1 (still null if present)
        "batter_pull_pct_30d",
    }
    flist_path = os.path.join(ROOT, "features/v1.1/feature_list.json")
    import json
    with open(flist_path) as fp:
        feature_cols = json.load(fp)
    train = f[f["split"] == "train"]
    high_null = []
    for c in feature_cols:
        if c in UNAVAILABLE_V11:
            continue
        null_rate = train[c].isna().mean()
        if null_rate > 0.05:
            high_null.append((c, null_rate))
    if high_null:
        print(f"FAIL: {len(high_null)} non-nullable features have >5% null rate on train:",
              file=sys.stderr)
        for c, r in high_null[:20]:
            print(f"  {c}: {r:.2%}", file=sys.stderr)
        return 1
    print(f"PASS: all {len(feature_cols)} features have <5% null on train (excluding unavailable)")

    # Invariant 5 (v1.1 specific): all 21 NEW features (QoC + park) have
    # 0 nulls on every split.
    new_features = [
        "batter_barrel_rate_30d", "batter_barrel_rate_season", "batter_barrel_rate_career",
        "batter_xwoba_30d", "batter_xwoba_season", "batter_xwoba_career",
        "batter_avg_ev_30d", "batter_avg_ev_season",
        "batter_ev90_30d", "batter_avg_la_30d",
        "batter_hard_hit_pct_30d", "batter_fb_pct_30d", "batter_iso_30d",
        "pitcher_barrel_rate_allowed_30d", "pitcher_barrel_rate_allowed_season",
        "pitcher_xwoba_allowed_30d", "pitcher_hard_hit_pct_allowed_30d",
        "pitcher_avg_ev_allowed_30d", "pitcher_iso_allowed_30d",
        "park_hr_factor_3yr", "park_hr_factor_by_hand",
    ]
    for split in ["train", "val", "holdout"]:
        sub = f[f["split"] == split]
        for c in new_features:
            if c not in sub.columns:
                print(f"FAIL: new feature {c} not in {FEAT}", file=sys.stderr)
                return 1
            if sub[c].isna().any():
                n_null = int(sub[c].isna().sum())
                print(f"FAIL: {c} has {n_null} nulls in {split} "
                      f"({n_null/len(sub):.2%})", file=sys.stderr)
                return 1
    print(f"PASS: all {len(new_features)} v1.1 new features have 0 nulls on train/val/holdout")

    # Invariant 6 (v1.1 specific): v1 features are unchanged (compare against v1)
    # Load only the v1 join keys + v1 feature columns to save memory.
    # park_hr_factor_3yr is intentionally changed (v1 had constant 100.0; v1.1
    # uses the real BIP-derived 3-yr factor).
    v1_fl = json.load(open(os.path.join(ROOT, "features/v1/feature_list.json")))
    v1_fl_compare = [c for c in v1_fl if c != "park_hr_factor_3yr"]
    v1 = pd.read_parquet(
        os.path.join(ROOT, "features/v1/game_features.parquet"),
        columns=["batter_id", "game_pk"] + v1_fl_compare,
    )
    feat_v1 = f[["batter_id", "game_pk"] + v1_fl_compare].merge(
        v1, on=["batter_id", "game_pk"], how="inner", suffixes=("", "_v1")
    )
    del v1
    import gc as _gc
    _gc.collect()
    mismatches = 0
    for c in v1_fl_compare:
        a = feat_v1[c].to_numpy()
        b = feat_v1[c + "_v1"].to_numpy()
        # Compare with NaN-aware equality
        if not np.array_equal(np.where(pd.isna(a), -999, a), np.where(pd.isna(b), -999, b)):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH in {c}", file=sys.stderr)
    del feat_v1
    _gc.collect()
    if mismatches > 0:
        print(f"FAIL: {mismatches}/{len(v1_fl_compare)} v1 features changed in v1.1", file=sys.stderr)
        return 1
    print(f"PASS: all {len(v1_fl_compare)} v1 features unchanged in v1.1 "
          f"(park_hr_factor_3yr intentionally replaced with real value)")

    print("\nAll no-lookahead invariants satisfied (v1.1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
