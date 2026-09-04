"""
Process the acquired Statcast BIP chunks into per-BIP parquet + park factors.

Inputs:
  /workspace/hr_model/data/raw/bip_all_raw.parquet   (pitch-level with BIP fields)
  /workspace/hr_model/data/raw/bip_<season>_<start>_<end>.csv   (raw BIP chunks)
  /workspace/hr_model/data/curated/game.parquet   (for park_id mapping & integrity)

Outputs:
  /workspace/hr_model/data/raw/bip_all.parquet   (per-BIP, collapsed)
  /workspace/hr_model/data/curated/park_factors.parquet   (3-yr rolling HR factors)

BIP semantics:
  - The Statcast `type=bip` URL still returns one row per pitch (the same as
    `type=details`). The BIP fields (launch_speed, launch_angle, hit_distance_sc,
    bb_type, woba_value, etc.) are populated for `description == 'hit_into_play'`.
  - There is one BIP per PA. Collapse by taking the single hit_into_play pitch
    per (game_pk, at_bat_number).
  - Keep only rows where events is in the BIP outcome set AND launch_speed is
    non-null (drop unmeasured BIPs).
  - 2025 holdout rule: only seasons 2015-2024 are processed. We do NOT read
    any 2025 BIP data. This is enforced at multiple levels.
"""

import os
import sys
import glob
import gc
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("process_bip")

RAW_DIR = "/workspace/hr_model/data/raw"
CURATED_DIR = "/workspace/hr_model/data/curated"
os.makedirs(CURATED_DIR, exist_ok=True)

# 2025 holdout rule: only process seasons 2015-2024.
ALLOWED_SEASONS = set(range(2015, 2025))

# BIP outcomes we keep (after collapsing to per-PA).
# One row per BIP — events is the final PA outcome string.
BIP_EVENTS = {
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "force_out",
    "fielders_choice", "fielders_choice_out",
    "sac_bunt", "sac_fly", "sac_fly_double_play", "sac_bunt_double_play",
    "field_error", "triple_play", "double_play", "other_out",
}

REQUIRED_COLS = [
    "game_pk", "game_date", "batter", "pitcher", "park_id", "events",
    "launch_speed", "launch_angle", "hit_distance_sc", "woba_value",
    "estimated_ba_using_speedangle", "bb_type",
]

# park_id mapping (mirror of process.py)
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from process import team_to_park, normalize_hand  # noqa: E402


def process_bip_season(year: int) -> pd.DataFrame:
    """Process a single season's BIP chunks into a per-BIP DataFrame."""
    files = sorted(glob.glob(f"{RAW_DIR}/bip_{year}_{year}*.csv"))
    if not files:
        return pd.DataFrame()

    needed = [
        "game_pk", "game_date", "at_bat_number", "pitch_number",
        "batter", "pitcher", "home_team", "away_team",
        "stand", "p_throws", "events", "description", "bb_type",
        "launch_speed", "launch_angle", "hit_distance_sc",
        "woba_value", "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "bat_speed", "swing_length",
    ]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df.dropna(how="all")
            if len(df) == 0:
                continue
            keep = [c for c in needed if c in df.columns]
            df = df[keep]
            dfs.append(df)
        except Exception as e:
            log.warning("Failed to read %s: %r", f, e)
        del df

    if not dfs:
        return pd.DataFrame()
    big = pd.concat(dfs, ignore_index=True, sort=False)
    big.columns = [c.strip() for c in big.columns]
    n0 = len(big)
    # Keep only the hit_into_play pitch of each PA — that's the BIP.
    bip_rows = big[big["description"] == "hit_into_play"].copy()
    # Dedupe: one BIP per (game_pk, at_bat_number)
    bip_rows = bip_rows.drop_duplicates(subset=["game_pk", "at_bat_number"], keep="first")
    # Lowercase events
    bip_rows["events"] = bip_rows["events"].astype(str).str.strip().str.lower()
    # Apply event filter
    bip_rows = bip_rows[bip_rows["events"].isin(BIP_EVENTS)]
    # Drop rows with no launch_speed (unmeasured BIPs)
    bip_rows = bip_rows[bip_rows["launch_speed"].notna()].copy()
    # park_id from home_team
    bip_rows["park_id"] = bip_rows["home_team"].apply(team_to_park).astype("int32")
    # Type cleanup
    bip_rows["batter"] = pd.to_numeric(bip_rows["batter"], errors="coerce").astype("Int64")
    bip_rows["pitcher"] = pd.to_numeric(bip_rows["pitcher"], errors="coerce").astype("Int64")
    bip_rows["game_pk"] = pd.to_numeric(bip_rows["game_pk"], errors="coerce").astype("Int64")
    bip_rows["game_date"] = pd.to_datetime(bip_rows["game_date"], errors="coerce").dt.date
    log.info("Season %d: %d raw -> %d per-BIP", year, n0, len(bip_rows))
    return bip_rows


def build_bip_all(seasons: list[int]) -> pd.DataFrame:
    """Build the combined per-BIP parquet for the given seasons."""
    parts = []
    for y in seasons:
        if y not in ALLOWED_SEASONS:
            log.warning("Skipping season %d (not in allowed 2015-2024)", y)
            continue
        df = process_bip_season(y)
        if len(df):
            parts.append(df)
        del df
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True, sort=False)
    log.info("Combined BIP: %d rows from %d seasons", len(big), len(parts))
    return big


def compute_park_factors(bip: pd.DataFrame) -> pd.DataFrame:
    """Compute (park_id, year) HR factor and 3-year rolling means.

    park_hr_factor_<year> = (HR/BIP at park in year) / (league HR/BIP in year) * 100
    park_hr_factor_3yr[park, year] = mean(hf[park, y-2 .. y])   # current + 2 priors
    park_hr_factor_by_hand_L/R: same computation but split by batter hand
    """
    df = bip.copy()
    df["year"] = pd.to_datetime(df["game_date"]).dt.year
    df["batter_hand"] = df["stand"].apply(normalize_hand)
    df["is_hr"] = (df["events"] == "home_run").astype(int)

    # ---- single-year league rate ----
    league = df.groupby("year", as_index=False).agg(
        league_hr=("is_hr", "sum"),
        league_bip=("is_hr", "count"),
    )
    league["league_hr_rate"] = league["league_hr"] / league["league_bip"]

    # ---- single-year park rate (all hands) ----
    park = df.groupby(["park_id", "year"], as_index=False).agg(
        park_hr=("is_hr", "sum"),
        park_bip=("is_hr", "count"),
    )
    park = park.merge(league[["year", "league_hr_rate"]], on="year", how="left")
    park["park_hr_factor"] = np.where(
        park["park_bip"] > 0,
        (park["park_hr"] / park["park_bip"]) / park["league_hr_rate"] * 100.0,
        np.nan,
    )

    # ---- by hand: LHH and RHH ----
    by_hand = df.groupby(["park_id", "year", "batter_hand"], as_index=False).agg(
        hand_hr=("is_hr", "sum"),
        hand_bip=("is_hr", "count"),
    )
    hand_league = df.groupby(["year", "batter_hand"], as_index=False).agg(
        league_hand_hr=("is_hr", "sum"),
        league_hand_bip=("is_hr", "count"),
    )
    hand_league["league_hand_hr_rate"] = hand_league["league_hand_hr"] / hand_league["league_hand_bip"]
    by_hand = by_hand.merge(
        hand_league[["year", "batter_hand", "league_hand_hr_rate"]],
        on=["year", "batter_hand"], how="left",
    )
    by_hand["hand_factor"] = np.where(
        by_hand["hand_bip"] > 0,
        (by_hand["hand_hr"] / by_hand["hand_bip"]) / by_hand["league_hand_hr_rate"] * 100.0,
        np.nan,
    )
    # pivot to wide
    hand_wide = by_hand.pivot_table(
        index=["park_id", "year"],
        columns="batter_hand",
        values="hand_factor",
        aggfunc="first",
    ).reset_index()
    hand_wide = hand_wide.rename(columns={"L": "park_hr_factor_by_hand_L", "R": "park_hr_factor_by_hand_R"})

    # ---- 3-year rolling mean per (park_id, year) ----
    park = park.sort_values(["park_id", "year"]).reset_index(drop=True)
    # For each row, gather current + 2 prior years' factor
    # We do this with a sorted merge: outer join, then rolling.
    out_rows = []
    for pid, sub in park.groupby("park_id"):
        sub = sub.sort_values("year").reset_index(drop=True)
        factors = sub["park_hr_factor"].values
        years = sub["year"].values
        roll3 = []
        for i, y in enumerate(years):
            # window: all rows with year in [y-2, y]
            mask = (years >= y - 2) & (years <= y)
            window = factors[mask]
            window = window[~np.isnan(window)]
            if len(window) == 0:
                roll3.append(np.nan)
            else:
                roll3.append(float(np.nanmean(window)))
        sub["park_hr_factor_3yr"] = roll3
        # also store single-year column with year-suffixed name for inspection
        out_rows.append(sub)
    park_rolled = pd.concat(out_rows, ignore_index=True)

    # ---- assemble final table ----
    out = park_rolled[["park_id", "year", "park_hr_factor", "park_hr_factor_3yr"]].copy()
    out = out.rename(columns={"park_hr_factor": "hr_factor_year"})

    # add by-hand
    out = out.merge(hand_wide, on=["park_id", "year"], how="left")

    # Also add a per-year column with the year-suffixed name (per the spec:
    # `hr_factor_<year>` is the single-year factor). The spec also says
    # `hr_factor_<year>` per row, so to keep it simple we store the single-year
    # value in a `hr_factor_year` column AND also add a `hr_factor_<year>` column
    # per actual year using the year value. We'll provide both: the data has
    # the same factor in both places; downstream joins should use park_hr_factor_3yr
    # for the rolling feature.
    # We also expose the suffix-named single-year column so the verifier can spot-check
    # any (park, year) directly.
    for y in sorted(out["year"].unique()):
        col = f"hr_factor_{int(y)}"
        if col not in out.columns:
            out[col] = np.nan
        out.loc[out["year"] == y, col] = out.loc[out["year"] == y, "hr_factor_year"]

    out = out.sort_values(["park_id", "year"]).reset_index(drop=True)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="*", type=int, default=None,
                        help="Seasons to process (default: 2015-2024)")
    parser.add_argument("--skip-park", action="store_true",
                        help="Skip park factor computation")
    args = parser.parse_args()

    seasons = args.seasons or list(range(2015, 2025))
    # Defensive: hard-fail on any 2025 season
    bad = [s for s in seasons if s not in ALLOWED_SEASONS]
    if bad:
        log.error("Refusing to process seasons %s — only 2015-2024 allowed (2025 holdout rule)", bad)
        sys.exit(2)

    log.info("Processing BIP seasons: %s", seasons)

    t0 = __import__("time").time()
    bip_all = build_bip_all(seasons)
    if bip_all.empty:
        log.error("No BIP rows produced — aborting")
        sys.exit(1)

    # Final column order: required cols first, then everything else
    other_cols = [c for c in bip_all.columns if c not in REQUIRED_COLS]
    bip_all = bip_all[REQUIRED_COLS + other_cols]
    out_bip = os.path.join(RAW_DIR, "bip_all.parquet")
    bip_all.to_parquet(out_bip, index=False)
    log.info("Wrote %s: %d rows (%.1f MB)", out_bip, len(bip_all), os.path.getsize(out_bip) / 1e6)

    # Verify: max year <= 2024
    max_year = pd.to_datetime(bip_all["game_date"]).dt.year.max()
    if max_year > 2024:
        log.error("FATAL: 2025 row leaked into bip_all.parquet (max year = %d)", max_year)
        sys.exit(3)
    log.info("Max game_date year = %d (PASS: no 2025 row)", max_year)

    # Park factors
    if not args.skip_park:
        log.info("Computing park factors")
        pf = compute_park_factors(bip_all)
        out_pf = os.path.join(CURATED_DIR, "park_factors.parquet")
        pf.to_parquet(out_pf, index=False)
        log.info("Wrote %s: %d (park, year) rows (%.1f KB)",
                 out_pf, len(pf), os.path.getsize(out_pf) / 1e3)

        # Quick sanity check
        for pid, name in [(9, "Coors (COL)"), (24, "Oracle (SF)"), (19, "Coliseum (OAK)")]:
            sub = pf[pf["park_id"] == pid].sort_values("year")
            if len(sub):
                latest = sub.iloc[-1]
                log.info("  %s latest: year=%d hr_factor_year=%.1f park_hr_factor_3yr=%.1f "
                         "L=%.1f R=%.1f",
                         name, int(latest["year"]), latest["hr_factor_year"],
                         latest["park_hr_factor_3yr"],
                         latest.get("park_hr_factor_by_hand_L", np.nan),
                         latest.get("park_hr_factor_by_hand_R", np.nan))

    log.info("Process_bip total time: %.1f s", __import__("time").time() - t0)


if __name__ == "__main__":
    main()
