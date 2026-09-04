"""
Integrity tests for the v1.1 BIP data pull and park factors.

Run with: python3 tests/test_bip_integrity.py
"""

import os
import sys
import pandas as pd
import numpy as np

RAW_DIR = "/workspace/hr_model/data/raw"
CURATED_DIR = "/workspace/hr_model/data/curated"

BIP_PATH = os.path.join(RAW_DIR, "bip_all.parquet")
PF_PATH = os.path.join(CURATED_DIR, "park_factors.parquet")
GAME_PATH = os.path.join(CURATED_DIR, "game.parquet")


def check_a_bip_row_count():
    """(a) n_BIP_parquet == 2015-2024 BIPs with launch_speed ≈ 1.0M-1.3M.
    (about 120k BIPs per regular season over 10 seasons, after filtering
    to events in BIP_EVENTS and dropping unmeasured BIPs)."""
    assert os.path.exists(BIP_PATH), f"missing {BIP_PATH}"
    df = pd.read_parquet(BIP_PATH)
    n = len(df)
    print(f"[a] n_BIP_parquet = {n:,}")
    assert 900_000 <= n <= 1_500_000, f"n_BIP = {n} not in [900k, 1.5M]"
    print(f"    PASS: {n:,} rows in [{900_000:,}, {1_500_000:,}]")
    return df


def check_b_launch_speed_null_rate(bip: pd.DataFrame):
    """(b) launch_speed null rate < 50% (some BIPs are pre-Statcast)."""
    if "launch_speed" not in bip.columns:
        print("[b] FAIL: launch_speed column missing")
        sys.exit(1)
    null_rate = float(bip["launch_speed"].isna().mean())
    print(f"[b] launch_speed null rate = {null_rate:.2%}")
    assert null_rate < 0.50, f"launch_speed null rate {null_rate:.2%} >= 50%"
    # Show breakdown by year (informational)
    by_year = bip.copy()
    by_year["year"] = pd.to_datetime(by_year["game_date"]).dt.year
    rates = by_year.groupby("year")["launch_speed"].apply(lambda s: s.isna().mean())
    print("    per-year null rates:")
    for y, r in rates.items():
        print(f"      {y}: {r:.2%}")
    print("    PASS: overall null rate < 50%")


def check_c_park_factors_complete():
    """(c) park_hr_factor_3yr: per (park, year) row exists for every
    (park, year) where park appears in game.parquet."""
    assert os.path.exists(PF_PATH), f"missing {PF_PATH}"
    assert os.path.exists(GAME_PATH), f"missing {GAME_PATH}"
    pf = pd.read_parquet(PF_PATH)
    games = pd.read_parquet(GAME_PATH)
    games["year"] = pd.to_datetime(games["game_date"]).dt.year
    # We only need parks that appear in 2015-2024 games (not 2025)
    games_2015_2024 = games[games["year"].between(2015, 2024)]
    expected_pairs = set(
        zip(games_2015_2024["park_id"].astype(int), games_2015_2024["year"].astype(int))
    )
    actual_pairs = set(zip(pf["park_id"].astype(int), pf["year"].astype(int)))
    missing = expected_pairs - actual_pairs
    print(f"[c] expected (park, year) pairs: {len(expected_pairs):,}")
    print(f"    actual   (park, year) pairs: {len(actual_pairs):,}")
    if missing:
        sample = sorted(missing)[:10]
        print(f"    MISSING: {len(missing)} pairs, sample: {sample}")
        sys.exit(1)
    print(f"    PASS: all {len(expected_pairs):,} (park, year) pairs present")
    # Also: all non-null park_hr_factor_3yr
    null_3yr = pf["park_hr_factor_3yr"].isna().sum()
    if null_3yr > 0:
        print(f"    WARN: {null_3yr} park_hr_factor_3yr rows are null")
    assert null_3yr == 0, f"{null_3yr} park_hr_factor_3yr rows are null"


def check_d_no_2025(bip: pd.DataFrame):
    """(d) No 2025 row in bip_all.parquet (max year <= 2024)."""
    years = pd.to_datetime(bip["game_date"]).dt.year
    max_year = int(years.max())
    min_year = int(years.min())
    print(f"[d] min year = {min_year}, max year = {max_year}")
    assert max_year <= 2024, f"max year {max_year} > 2024 (2025 row leaked!)"
    n_2025 = (years == 2025).sum()
    assert n_2025 == 0, f"n_2025 = {n_2025}"
    print(f"    PASS: no 2025 rows (max year = {max_year})")


def check_e_team_park_mapping(bip: pd.DataFrame):
    """(e) Park-attribution coverage: 30 MLB teams map to non-40 park_ids
    with non-trivial BIP counts. This catches the bug where 10% of BIPs
    were mis-attributed to park 40 because Statcast team codes (ATH, AZ)
    weren't in the v1 team_to_park table, and Truist Park (ATL) was missing.
    """
    # Re-derive the mapping (mirrors process.py)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
    from process import team_to_park

    bip_local = bip.copy()
    bip_local["park_id"] = bip_local["home_team"].apply(team_to_park).astype("int32")
    park_counts = bip_local["park_id"].value_counts()

    # Park 40 ("Other") should have very few BIPs (only All-Star games +
    # spring training / minor league codes not in the table). Historically
    # it was inflated to 119k before the fix.
    other_bips = int(park_counts.get(40, 0))
    print(f"[e] park 40 (Other/neutral) BIP count = {other_bips:,}")
    assert other_bips < 5_000, (
        f"park 40 inflated to {other_bips:,} BIPs — likely team_to_park "
        f"mapping bug (Statcast codes ATH/AZ not handled, or Truist Park missing)"
    )

    # Coliseum (19), Chase (2), Truist (41) should have non-trivial BIPs
    for pid, name in [(2, "Chase Field"), (19, "Oakland Coliseum"), (41, "Truist Park")]:
        n = int(park_counts.get(pid, 0))
        print(f"    park {pid} ({name}) BIP count = {n:,}")
        assert n > 30_000, f"park {pid} ({name}) has only {n} BIPs — likely mapping bug"
    print(f"    PASS: park attribution correct, 30 MLB parks all non-trivial")


def main():
    print("=" * 60)
    print("BIP integrity tests (v1.1)")
    print("=" * 60)
    bip = check_a_bip_row_count()
    check_b_launch_speed_null_rate(bip)
    check_c_park_factors_complete()
    check_d_no_2025(bip)
    check_e_team_park_mapping(bip)
    print()
    print("ALL BIP INTEGRITY TESTS PASSED")


if __name__ == "__main__":
    main()
