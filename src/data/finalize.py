"""
Build the final curated parquet store from per-season pitch-level parquets.
Memory-efficient: processes one season at a time and writes incrementally.

Outputs:
  /workspace/hr_model/data/curated/pa.parquet          (~7.5M rows, pitch-level)
  /workspace/hr_model/data/curated/game.parquet        (one row per game_pk)
  /workspace/hr_model/data/curated/roster.parquet      (one row per player-season)
  /workspace/hr_model/data/splits/{train,val,holdout}_ids.parquet
"""

import os
import gc
import datetime
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finalize")

RAW_DIR = "/workspace/hr_model/data/raw"
CURATED_DIR = "/workspace/hr_model/data/curated"
SPLITS_DIR = "/workspace/hr_model/data/splits"
os.makedirs(CURATED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


# ---------- park_id mapping ----------
PARK_TABLE = [
    (1, "Angel Stadium",              {"LAA", "ANA", "CAL"}),
    (2, "Chase Field",                {"ARI"}),
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
    (19, "Oakland Coliseum",          {"OAK"}),
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


def classify_pa_outcome(events):
    if not isinstance(events, str) or events == "" or events == "null":
        return None
    e = events.strip().lower()
    if e in ("home_run", "hr"):
        return "hr"
    if e in ("single", "double", "triple"):
        return "hit"
    if e in ("walk", "intent_walk"):
        return "walk"
    if e == "hit_by_pitch":
        return "hbp"
    if e in ("sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play", "sacrifice_bunt", "sacrifice_fly"):
        return "sacrifice"
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
    return "out"


# Schema for pa.parquet
PA_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("game_date", pa.date32()),
    ("batter_id", pa.int64()),
    ("pitcher_id", pa.int64()),
    ("park_id", pa.int32()),
    ("batter_hand", pa.string()),
    ("pitcher_hand", pa.string()),
    ("pa_outcome", pa.string()),
    ("events", pa.string()),
    ("inning", pa.int32()),
    ("inning_topbot", pa.string()),
    ("outs_when_up", pa.int32()),
    ("balls", pa.int32()),
    ("strikes", pa.int32()),
    ("pitch_number", pa.int32()),
    ("pitch_type", pa.string()),
])


def process_season_to_pa(year: int, writer: pq.ParquetWriter) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Process one season's parquet and append to pa.parquet writer.

    Returns (n_rows, games_df, players_df) where:
      games_df = unique games for this season
      players_df = unique player-roles (batter + pitcher) for this season
    """
    path = f"{RAW_DIR}/pa_{year}.parquet"
    log.info("Processing %d-season from %s", year, path)
    df = pd.read_parquet(path)
    n0 = len(df)

    # Convert game_date
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["batter_id"] = df["batter"].astype("int64")
    df["pitcher_id"] = df["pitcher"].astype("int64")
    df["game_pk"] = df["game_pk"].astype("int64")
    df["batter_hand"] = df["stand"].apply(normalize_hand)
    df["pitcher_hand"] = df["p_throws"].apply(normalize_hand)
    # PA outcome: derive from the last pitch of each (game, at_bat)
    pa_out = (
        df.dropna(subset=["events"])
        .query("events != '' and events != 'null'")
        .drop_duplicates(subset=["game_pk", "at_bat_number"], keep="last")[["game_pk", "at_bat_number", "events"]]
        .assign(pa_outcome=lambda d: d["events"].apply(classify_pa_outcome))
        [["game_pk", "at_bat_number", "pa_outcome"]]
    )
    df = df.merge(pa_out, on=["game_pk", "at_bat_number"], how="left")
    del pa_out
    gc.collect()

    df["events"] = df["events"].where(df["events"].astype(str).str.len() > 0, None)
    df["park_id"] = df["home_team"].apply(team_to_park).astype("int32")
    # inning, outs_when_up, balls, strikes
    for col in ("inning", "outs_when_up", "balls", "strikes", "pitch_number"):
        if col in df.columns:
            df[col] = df[col].astype("Int32")
    df["inning_topbot"] = df["inning_topbot"].astype("string")
    df["pitch_type"] = df["pitch_type"].astype("string")
    df["batter_hand"] = df["batter_hand"].astype("string")
    df["pitcher_hand"] = df["pitcher_hand"].astype("string")
    df["pa_outcome"] = df["pa_outcome"].astype("string")
    df["events"] = df["events"].astype("string")

    out_cols = [
        "game_pk", "game_date", "batter_id", "pitcher_id", "park_id",
        "batter_hand", "pitcher_hand", "pa_outcome", "events",
        "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
        "pitch_number", "pitch_type",
    ]
    pa_df = df[out_cols].copy()
    # Drop rows with missing required
    pa_df = pa_df.dropna(subset=["batter_hand", "pitcher_hand", "pa_outcome"])
    log.info("  %d: %d pitch rows -> %d after hand+outcome filter", year, n0, len(pa_df))

    # Convert to pyarrow Table with explicit schema
    table = pa.Table.from_pandas(pa_df, schema=PA_SCHEMA, preserve_index=False)
    writer.write_table(table)
    n_rows = len(pa_df)
    del pa_df, df, table
    gc.collect()

    # Build games df (read back from raw)
    games_df = pd.read_parquet(path, columns=["game_pk", "game_date", "home_team", "away_team"])
    games_df = games_df.drop_duplicates(subset=["game_pk"])
    games_df["game_date"] = pd.to_datetime(games_df["game_date"]).dt.date

    # Build players df (batter + pitcher, with handedness)
    players_df = pd.read_parquet(path, columns=["batter", "pitcher", "stand", "p_throws", "game_date"])
    players_df["season"] = pd.to_datetime(players_df["game_date"]).dt.year
    return n_rows, games_df, players_df


def main():
    pa_path = f"{CURATED_DIR}/pa.parquet"
    if os.path.exists(pa_path):
        os.remove(pa_path)
    log.info("Writing pa.parquet incrementally")
    writer = pq.ParquetWriter(pa_path, PA_SCHEMA, compression="snappy")

    all_games = []
    all_players = []
    total_rows = 0

    for y in range(2015, 2026):
        n, gd, pd_df = process_season_to_pa(y, writer)
        total_rows += n
        all_games.append(gd)
        all_players.append(pd_df)
        log.info("  cumulative pa rows: %d", total_rows)

    writer.close()
    log.info("Wrote %s: %d rows total (%.1f MB)", pa_path, total_rows, os.path.getsize(pa_path) / 1e6)

    # Build game.parquet from all_games
    log.info("Building game.parquet from per-season games")
    games = pd.concat(all_games, ignore_index=True)
    del all_games
    gc.collect()
    games = games.drop_duplicates(subset=["game_pk"])
    games["park_id"] = games["home_team"].apply(team_to_park).astype("int32")
    games["game_time_utc"] = pd.to_datetime(games["game_date"].astype(str) + "T17:00:00Z", errors="coerce")
    games["roof_closed"] = pd.NA
    games = games[["game_pk", "game_date", "home_team", "away_team", "park_id", "game_time_utc", "roof_closed"]]
    games.to_parquet(f"{CURATED_DIR}/game.parquet", index=False)
    log.info("Wrote game.parquet: %d rows", len(games))
    del games
    gc.collect()

    # Build roster.parquet from all_players
    log.info("Building roster.parquet from per-season players")
    players = pd.concat(all_players, ignore_index=True)
    del all_players
    gc.collect()
    log.info("  total player-appearances: %d", len(players))

    batters = (
        players[["batter", "season", "stand"]]
        .rename(columns={"batter": "player_id", "stand": "batter_hand"})
        .dropna(subset=["batter_hand"])
        .groupby(["player_id", "season"], as_index=False)
        .agg(batter_hand=("batter_hand", lambda s: s.mode().iloc[0] if len(s.mode()) else "R"))
    )
    batters["pitcher_hand"] = pd.NA
    batters["primary_position"] = "batter"
    batters["team"] = "UNK"
    batters["name"] = pd.NA

    pitchers = (
        players[["pitcher", "season", "p_throws"]]
        .rename(columns={"pitcher": "player_id", "p_throws": "pitcher_hand"})
        .dropna(subset=["pitcher_hand"])
        .groupby(["player_id", "season"], as_index=False)
        .agg(pitcher_hand=("pitcher_hand", lambda s: s.mode().iloc[0] if len(s.mode()) else "R"))
    )
    pitchers["batter_hand"] = pd.NA
    pitchers["primary_position"] = "pitcher"
    pitchers["team"] = "UNK"
    pitchers["name"] = pd.NA

    common = set(batters["player_id"]) & set(pitchers["player_id"])
    log.info("Two-way players: %d", len(common))
    only_b = batters[~batters["player_id"].isin(common)][["player_id", "season", "name", "team", "batter_hand", "pitcher_hand", "primary_position"]]
    only_p = pitchers[~pitchers["player_id"].isin(common)][["player_id", "season", "name", "team", "batter_hand", "pitcher_hand", "primary_position"]]
    two_way = []
    for pid in common:
        b_row = batters[batters["player_id"] == pid].iloc[0]
        p_row = pitchers[pitchers["player_id"] == pid].iloc[0]
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
    roster = pd.concat([only_b, only_p, twodf], ignore_index=True)
    roster = roster.drop_duplicates(subset=["player_id", "season"])
    roster["player_id"] = roster["player_id"].astype("int64")
    roster["season"] = roster["season"].astype("int32")
    roster["batter_hand"] = roster["batter_hand"].astype("string")
    roster["pitcher_hand"] = roster["pitcher_hand"].astype("string")
    roster["primary_position"] = roster["primary_position"].astype("string")
    log.info("Wrote roster.parquet: %d player-seasons", len(roster))
    roster.to_parquet(f"{CURATED_DIR}/roster.parquet", index=False)
    del players, batters, pitchers, roster
    gc.collect()

    # Build splits
    log.info("Building splits")
    games = pd.read_parquet(f"{CURATED_DIR}/game.parquet", columns=["game_pk", "game_date"])
    games["year"] = pd.to_datetime(games["game_date"]).dt.year
    train = games[games["year"].between(2015, 2022)][["game_pk"]]
    val = games[games["year"].between(2023, 2024)][["game_pk"]]
    hold = games[games["year"] == 2025][["game_pk"]]
    train.to_parquet(f"{SPLITS_DIR}/train_ids.parquet", index=False)
    val.to_parquet(f"{SPLITS_DIR}/val_ids.parquet", index=False)
    hold.to_parquet(f"{SPLITS_DIR}/holdout_ids.parquet", index=False)
    log.info("train: %d, val: %d, holdout: %d", len(train), len(val), len(hold))

    print("\n=== FINAL SUMMARY ===")
    for f in ["pa.parquet", "game.parquet", "roster.parquet"]:
        p = f"{CURATED_DIR}/{f}"
        print(f"{f}: {os.path.getsize(p)/1e6:.1f} MB")
    for f in ["train_ids.parquet", "val_ids.parquet", "holdout_ids.parquet"]:
        p = f"{SPLITS_DIR}/{f}"
        print(f"splits/{f}: {os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
