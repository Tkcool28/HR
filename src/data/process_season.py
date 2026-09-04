"""
Process a single season's chunks to a pitch-level parquet.

Memory-efficient: reads one CSV at a time, keeps only needed columns,
deduplicates, and writes a per-season parquet. Then deletes the CSVs.

Usage: python3 process_season.py <year>
"""

import os
import sys
import glob
import gc
import pandas as pd

RAW_DIR = "/workspace/hr_model/data/raw"

NEEDED = [
    "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher",
    "game_date", "home_team", "away_team", "stand", "p_throws", "events",
    "inning", "inning_topbot", "balls", "strikes", "outs_when_up", "pitch_type",
    "game_type", "game_year",
]


def process_season(year: int) -> str:
    files = sorted(glob.glob(f"{RAW_DIR}/savant_{year}_{year}*.csv"))
    if not files:
        return f"no files for {year}"
    print(f"{year}: {len(files)} chunks", flush=True)

    all_dfs = []
    total = 0
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, low_memory=False)
            df = df.dropna(how="all")
            if len(df) == 0:
                continue
            keep = [c for c in NEEDED if c in df.columns]
            df = df[keep]
            df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
            all_dfs.append(df)
            total += len(df)
            if i % 20 == 0:
                print(f"  {i+1}/{len(files)}: {total} rows", flush=True)
            del df
        except Exception as e:
            print(f"  failed: {f}: {e}", flush=True)

    if not all_dfs:
        return f"no data for {year}"

    big = pd.concat(all_dfs, ignore_index=True)
    n0 = len(big)
    big = big.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    n1 = len(big)
    print(f"  {year}: {n0} -> {n1} after final dedup, {big['game_pk'].nunique()} games", flush=True)
    out = f"{RAW_DIR}/pa_{year}.parquet"
    big.to_parquet(out, index=False)
    sz = os.path.getsize(out) / 1e6
    print(f"  saved {out} ({sz:.1f} MB)", flush=True)
    for f in files:
        os.remove(f)
    print(f"  deleted {len(files)} CSVs", flush=True)
    del big, all_dfs
    gc.collect()
    return out


if __name__ == "__main__":
    year = int(sys.argv[1])
    process_season(year)
