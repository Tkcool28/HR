"""
Process the acquired Statcast chunks into the curated parquet store.

Inputs:
  /workspace/hr_model/data/raw/all_chunks.parquet   (pitch-level)
  /workspace/hr_model/data/raw/savant_<season>_<start>_<end>.csv   (raw chunks)

Outputs:
  /workspace/hr_model/data/curated/pa.parquet
  /workspace/hr_model/data/curated/game.parquet
  /workspace/hr_model/data/curated/roster.parquet
  /workspace/hr_model/data/splits/train_ids.parquet
  /workspace/hr_model/data/splits/val_ids.parquet
  /workspace/hr_model/data/splits/holdout_ids.parquet
"""

import os
import json
import datetime
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("process")

RAW_DIR = "/workspace/hr_model/data/raw"
CURATED_DIR = "/workspace/hr_model/data/curated"
SPLITS_DIR = "/workspace/hr_model/data/splits"
os.makedirs(CURATED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


# ---------- park_id mapping ----------
# Statcast does not include a park_id. We build a small lookup from the
# (home_team, game_year) -> park_id using a hard-coded mapping. This is
# deterministic and good enough for v1.
PARK_TABLE = [
    # (park_id, name, team_codes)
    (1, "Angel Stadium",              {"LAA", "ANA", "CAL"}),
    (2, "Chase Field",                {"ARI", "AZ"}),
    (3, "Camden Yards",               {"BAL"}),
    (4, "Fenway Park",                {"BOS"}),
    (5, "Wrigley Field",              {"CHC"}),
    (6, "Guaranteed Rate Field",      {"CWS", "CHW"}),
    (7, "Great American Ball Park",   {"CIN"}),
    (8, "Progressive Field",          {"CLE"}),
    (9, "Coors Field",                {"COL"}),
    (10, "Comerica Park",             {"DET"}),
    (11, "Kauffman Stadium",          {"KCR", "KC"}),
    (12, "Minute Maid Park",          {"HOU"}),
    (13, "Dodger Stadium",            {"LAD"}),
    (14, "loanDepot park",            {"MIA", "FLA"}),
    (15, "Miller Park",               {"MIL"}),
    (16, "Target Field",              {"MIN"}),
    (17, "Citi Field",                {"NYM"}),
    (18, "Yankee Stadium",            {"NYY"}),
    (19, "Oakland Coliseum",          {"OAK", "ATH"}),
    (20, "Citizens Bank Park",        {"PHI"}),
    (21, "PNC Park",                  {"PIT"}),
    (22, "Petco Park",                {"SD", "SDP"}),
    (23, "T-Mobile Park",             {"SEA"}),
    (24, "Oracle Park",               {"SF", "SFG"}),
    (25, "Busch Stadium",             {"STL"}),
    (26, "Tropicana Field",           {"TB", "TBR", "TBD"}),
    (27, "Globe Life Field",          {"TEX"}),
    (28, "Rogers Centre",             {"TOR"}),
    (29, "Nationals Park",            {"WSH", "WAS", "MON"}),
    (41, "Truist Park",               {"ATL"}),
    # Historical / occasional parks
    (30, "Tokyo Dome",                set()),
    (31, "Hiram Bithorn Stadium",     set()),
    (32, "TD Ameritrade Park (COL)",  set()),
    (33, "Fort Bragg Field",          set()),
    (34, "Williamsport (LLWS)",       set()),
    (35, "London Stadium",            set()),
    (36, "Field of Dreams (Dyersville)", set()),
    (37, "Estadio de Monterrey",      set()),
    (38, "Sahlen Field (Buf)",        {"BUF"}),
    (39, "Charles Schwab Field (OKC)", {"OKC"}),
    (40, "Other / neutral",           set()),
]

TEAM_TO_PARK = {}
for pid, _name, teams in PARK_TABLE:
    for t in teams:
        TEAM_TO_PARK[t] = pid


def team_to_park(team: str) -> int:
    if not isinstance(team, str):
        return 40
    return TEAM_TO_PARK.get(team, 40)


# ---------- PA outcome mapping ----------
def classify_pa_outcome(events: str | None) -> str | None:
    if not isinstance(events, str) or events == "" or events == "null":
        return None
    e = events.strip().lower()
    if e in ("home_run", "hr"):
        return "hr"
    # Hits that are not HR
    if e in ("single", "double", "triple"):
        return "hit"
    # Walks + HBP
    if e in ("walk", "intent_walk", "hit_by_pitch"):
        return "walk" if e in ("walk", "intent_walk") else "hbp"
    # Sacrifice (bunt or fly)
    if e in ("sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play", "sacrifice_bunt", "sacrifice_fly"):
        return "sacrifice"
    # Out-related
    if e in (
        "field_out", "force_out", "grounded_into_double_play", "double_play",
        "triple_play", "fielders_choice", "fielders_choice_out",
        "strikeout", "strikeout_double_play", "batter_interference",
        "catcher_interference", "fan_interference", "field_error",
        "other_out", "pickoff_caught_stealing_2b", "caught_stealing_2b",
        "pickoff_caught_stealing_3b", "caught_stealing_3b", "pickoff_caught_stealing_home",
        "caught_stealing_home",
    ):
        return "out"
    # Catcher interference counts as a PA in modern rules; treat as walk
    if e == "catcher_interference":
        return "walk"
    # Last resort - treat as out
    return "out"


# ---------- handedness normalization ----------
HAND_MAP = {
    "L": "L", "l": "L", "Left": "L", "left": "L",
    "R": "R", "r": "R", "Right": "R", "right": "R",
    "S": "S", "s": "S", "Both": "S", "Switch": "S", "B": "S",
}


def normalize_hand(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    if s in HAND_MAP:
        return HAND_MAP[s]
    if s[0].upper() in ("L", "R", "S"):
        return s[0].upper()
    return None


# ---------- main pipeline ----------
def build_pa_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """Reduce pitch-level to PA-level (one row per at_bat_number within a game_pk)."""
    log.info("Reducing %d pitch rows to PA-level", len(pitches))

    # Ensure required columns
    required = ["game_pk", "at_bat_number", "batter", "pitcher", "stand", "p_throws", "events", "game_date", "home_team", "away_team", "inning", "inning_topbot", "balls", "strikes", "outs_when_up"]
    missing = [c for c in required if c not in pitches.columns]
    if missing:
        log.warning("Missing columns: %s", missing)

    # Use the last pitch of each at_bat for the canonical events
    pitches = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"], ascending=[True, True, True])
    last = pitches.dropna(subset=["at_bat_number"]).groupby(["game_pk", "at_bat_number"], as_index=False).tail(1)
    # Some at_bat_numbers may be missing - coerce
    last["at_bat_number"] = last["at_bat_number"].astype("Int64")
    last["batter"] = last["batter"].astype("Int64")
    last["pitcher"] = last["pitcher"].astype("Int64")
    last["inning"] = last["inning"].astype("Int64")
    last["outs_when_up"] = last["outs_when_up"].astype("Int64")

    last["batter_id"] = last["batter"].astype("int64")
    last["pitcher_id"] = last["pitcher"].astype("int64")
    last["game_pk"] = last["game_pk"].astype("int64")
    last["game_date"] = pd.to_datetime(last["game_date"]).dt.date
    last["batter_hand"] = last["stand"].apply(normalize_hand)
    last["pitcher_hand"] = last["p_throws"].apply(normalize_hand)
    last["events"] = last["events"].where(last["events"].astype(str).str.len() > 0, None)
    last["pa_outcome"] = last["events"].apply(classify_pa_outcome)
    # Park id: home team -> park
    last["park_id"] = last["home_team"].apply(team_to_park).astype("int32")

    cols_out = [
        "game_pk", "game_date", "batter_id", "pitcher_id", "park_id",
        "batter_hand", "pitcher_hand", "pa_outcome", "events",
        "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
        "home_team", "away_team",
    ]
    out = last[cols_out].copy()
    out = out.dropna(subset=["batter_hand", "pitcher_hand", "pa_outcome"])
    out["batter_hand"] = out["batter_hand"].astype("string")
    out["pitcher_hand"] = out["pitcher_hand"].astype("string")
    out["pa_outcome"] = out["pa_outcome"].astype("string")
    log.info("PA table: %d rows", len(out))
    return out


def build_game_table(pitches: pd.DataFrame) -> pd.DataFrame:
    log.info("Building game table from %d pitch rows", len(pitches))
    g = pitches.groupby("game_pk", as_index=False).agg(
        game_date=("game_date", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
    )
    g["game_pk"] = g["game_pk"].astype("int64")
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date
    g["park_id"] = g["home_team"].apply(team_to_park).astype("int32")
    # game_time_utc: derive a placeholder noon UTC (we don't have it in Savant)
    g["game_time_utc"] = pd.to_datetime(g["game_date"].astype(str) + "T17:00:00Z", errors="coerce")
    # roof_closed: None (unknown from Savant alone)
    g["roof_closed"] = pd.NA
    out = g[["game_pk", "game_date", "home_team", "away_team", "park_id", "game_time_utc", "roof_closed"]].copy()
    log.info("Game table: %d rows", len(out))
    return out


def build_roster_table(pa: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (player_id, season) by aggregating handedness from PA table."""
    log.info("Building roster table from PA table")
    pa2 = pa.copy()
    pa2["season"] = pd.to_datetime(pa2["game_date"]).dt.year

    # Batter rows
    batters = pa2[["batter_id", "season", "batter_hand"]].rename(columns={"batter_id": "player_id", "batter_hand": "batter_hand_observed"})
    batters["pitcher_hand"] = pd.NA
    batters["primary_position"] = "batter"

    pitchers = pa2[["pitcher_id", "season", "pitcher_hand"]].rename(columns={"pitcher_id": "player_id", "pitcher_hand": "pitcher_hand_observed"})
    pitchers["batter_hand"] = pd.NA
    pitchers["primary_position"] = "pitcher"

    # Combine
    bat = batters.groupby(["player_id", "season"], as_index=False).agg(
        batter_hand=("batter_hand_observed", lambda s: s.mode().iloc[0] if len(s.mode()) else "R"),
        pitcher_hand=("pitcher_hand", "first"),
        primary_position=("primary_position", "first"),
    )
    pit = pitchers.groupby(["player_id", "season"], as_index=False).agg(
        batter_hand=("batter_hand", "first"),
        pitcher_hand=("pitcher_hand_observed", lambda s: s.mode().iloc[0] if len(s.mode()) else "R"),
        primary_position=("primary_position", "first"),
    )

    # Tag as batter/pitcher based on whether they appeared as a batter (b) or pitcher (p)
    # Union; later we may dedupe
    bat["team"] = "UNK"  # Savant doesn't give team of player per PA
    pit["team"] = "UNK"
    bat["name"] = pd.NA
    pit["name"] = pd.NA
    bat["role"] = "batter"
    pit["role"] = "pitcher"

    # First, find two-way players (in both groups)
    common = set(bat["player_id"]) & set(pit["player_id"])
    log.info("Two-way players: %d", len(common))

    # Players only in one group
    only_b = bat[~bat["player_id"].isin(common)]
    only_p = pit[~pit["player_id"].isin(common)]
    # Two-way: combine
    two_way = []
    for pid in common:
        b_row = bat[bat["player_id"] == pid].iloc[0]
        p_row = pit[pit["player_id"] == pid].iloc[0]
        # Use the batter_hand from b_row, pitcher_hand from p_row, position = "two_way"
        two_way.append({
            "player_id": int(pid),
            "season": int(b_row["season"]),
            "name": pd.NA,
            "team": "UNK",
            "batter_hand": b_row["batter_hand"],
            "pitcher_hand": p_row["pitcher_hand"],
            "primary_position": "two_way",
        })
    twodf = pd.DataFrame(two_way)
    only_b = only_b.rename(columns={"batter_hand_observed": "batter_hand"})
    only_p = only_p.rename(columns={"pitcher_hand_observed": "pitcher_hand"})

    b_out = only_b[["player_id", "season", "name", "team", "batter_hand", "pitcher_hand", "primary_position"]]
    p_out = only_p[["player_id", "season", "name", "team", "batter_hand", "pitcher_hand", "primary_position"]]

    out = pd.concat([b_out, p_out, twodf], ignore_index=True)
    # Dedupe
    out = out.drop_duplicates(subset=["player_id", "season"])
    # Type cleanup
    out["player_id"] = out["player_id"].astype("int64")
    out["season"] = out["season"].astype("int32")
    out["batter_hand"] = out["batter_hand"].astype("string")
    out["pitcher_hand"] = out["pitcher_hand"].astype("string")
    out["primary_position"] = out["primary_position"].astype("string")
    log.info("Roster table: %d player-seasons", len(out))
    return out


def make_splits(game_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    g = game_table.copy()
    g["season"] = pd.to_datetime(g["game_date"]).dt.year
    train_ids = g[g["season"].between(2015, 2022)][["game_pk"]].copy()
    val_ids = g[g["season"].between(2023, 2024)][["game_pk"]].copy()
    holdout_ids = g[g["season"] == 2025][["game_pk"]].copy()
    return {
        "train_ids.parquet": train_ids,
        "val_ids.parquet": val_ids,
        "holdout_ids.parquet": holdout_ids,
    }


def main():
    raw_path = os.path.join(RAW_DIR, "all_chunks.parquet")
    log.info("Reading %s", raw_path)
    pitches = pd.read_parquet(raw_path)
    log.info("Loaded %d pitch rows", len(pitches))

    # Build PA table
    pa = build_pa_table(pitches)
    pa.to_parquet(os.path.join(CURATED_DIR, "pa.parquet"), index=False)
    log.info("Wrote pa.parquet: %d rows", len(pa))

    # Build game table
    games = build_game_table(pitches)
    games.to_parquet(os.path.join(CURATED_DIR, "game.parquet"), index=False)
    log.info("Wrote game.parquet: %d rows", len(games))

    # Build roster
    roster = build_roster_table(pa)
    roster.to_parquet(os.path.join(CURATED_DIR, "roster.parquet"), index=False)
    log.info("Wrote roster.parquet: %d rows", len(roster))

    # Splits
    splits = make_splits(games)
    for fname, df in splits.items():
        df.to_parquet(os.path.join(SPLITS_DIR, fname), index=False)
        log.info("Wrote splits/%s: %d rows", fname, len(df))

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"pa.parquet: {len(pa):,} rows")
    print(f"game.parquet: {len(games):,} rows")
    print(f"roster.parquet: {len(roster):,} rows")
    for fname, df in splits.items():
        print(f"splits/{fname}: {len(df):,} rows")


if __name__ == "__main__":
    main()
