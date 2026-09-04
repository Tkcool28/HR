"""
Statcast BIP (Batted Ball In Play) data acquisition via Baseball Savant CSV endpoint.

Mirrors acquire.py but pulls with `type=bip` instead of `type=details`,
so the response includes the per-BIP Statcast fields:
  launch_speed, launch_angle, hit_distance_sc, woba_value,
  estimated_ba_using_speedangle, bat_speed, swing_length, bb_type, barrel.

The Savant bulk CSV endpoint has a hard 25,000-row cap per request. We chunk
each regular season into 3-4 day windows and pull each chunk in parallel.
BIPs are sparser than pitches (~600k over 10 seasons vs ~7M pitches), so
4-day chunks are very safe.

Output: raw CSVs saved to /workspace/hr_model/data/raw/bip_<season>_<start>_<end>.csv
and a combined parquet at /workspace/hr_model/data/raw/bip_all_raw.parquet
(one row per BIP pitch — still at pitch granularity from the API).

By default pulls 2015-2024 (10 seasons, no 2025 — see 2025 holdout rule).
"""

import os
import sys
import glob
import time
import datetime
import logging
import concurrent.futures
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("acquire_bip")

RAW_DIR = "/workspace/hr_model/data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Only BIP. type=bip. All other query params match v1.
SAVANT_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv"
    "?all=true&type=bip&game_type=R"
    "&season={season}"
    "&game_date_gt={start}&game_date_lt={end}"
    "&min_pa=0&min_results=0"
    "&group_by=name&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc"
)

CAP_LINE_THRESHOLD = 24_500  # below 25k to leave headroom


def fetch_chunk(season: int, start: str, end: str, max_retries: int = 4) -> dict:
    """Fetch one Savant BIP chunk. Returns dict with metadata + csv_text."""
    out_path = os.path.join(RAW_DIR, f"bip_{season}_{start}_{end}.csv")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        with open(out_path) as f:
            text = f.read()
        return {"season": season, "start": start, "end": end, "text": text, "cached": True, "path": out_path}

    url = SAVANT_URL.format(season=season, start=start, end=end)
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            r = requests.get(url, timeout=120)
            elapsed = time.time() - t0
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                time.sleep(2 + attempt * 2)
                continue
            text = r.text
            with open(out_path, "w") as f:
                f.write(text)
            return {
                "season": season,
                "start": start,
                "end": end,
                "text": text,
                "cached": False,
                "elapsed": elapsed,
                "path": out_path,
                "n_lines": len(text.split("\n")),
            }
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 + attempt * 2)
    log.warning("chunk %s %s..%s failed: %s", season, start, end, last_err)
    return {"season": season, "start": start, "end": end, "text": "", "error": last_err, "path": None}


def plan_chunks(season: int) -> list[tuple[str, str]]:
    """Yield (start, end) ISO date strings for the season's regular season.
    Use 4-day chunks for April/September/October, 3-day chunks for peak May-August.
    BIPs are sparser than pitches, so larger windows are fine and we can use
    4-day chunks uniformly for the shoulder months.
    """
    chunks = []
    # 4-day chunks: Apr 1 to Apr 30
    for d in _daterange(datetime.date(season, 4, 1), datetime.date(season, 5, 1), step=4):
        chunks.append(d)
    # 3-day chunks: May 1 to Aug 31
    for d in _daterange(datetime.date(season, 5, 1), datetime.date(season, 9, 1), step=3):
        chunks.append(d)
    # 4-day chunks: Sept 1 to Nov 1
    for d in _daterange(datetime.date(season, 9, 1), datetime.date(season, 11, 1), step=4):
        chunks.append(d)
    return chunks


def _daterange(start: datetime.date, end: datetime.date, step: int = 4) -> list[tuple[str, str]]:
    out = []
    d = start
    while d < end:
        e = min(d + datetime.timedelta(days=step), end)
        out.append((d.isoformat(), e.isoformat()))
        d = e
    return out


def re_chunk_if_capped(season: int, start: str, end: str, workers: int = 6) -> list[dict]:
    """If a chunk is at the cap, split it in half and recurse."""
    res = fetch_chunk(season, start, end)
    if not res.get("text") or res.get("cached"):
        return [res]
    n_lines = len(res["text"].split("\n"))
    if n_lines < CAP_LINE_THRESHOLD:
        return [res]
    # Split in half
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    if (e - s).days <= 1:
        log.warning("Cannot sub-chunk %s %s..%s (n_lines=%d)", season, start, end, n_lines)
        return [res]
    mid = s + (e - s) / 2
    mid_str = mid.isoformat()
    out = []
    out.extend(re_chunk_if_capped(season, start, mid_str, workers))
    out.extend(re_chunk_if_capped(season, mid_str, end, workers))
    return out


def acquire_season(season: int, workers: int = 5) -> list[dict]:
    log.info("=== Season %d ===", season)
    chunks = plan_chunks(season)
    log.info("Planned %d primary chunks", len(chunks))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(re_chunk_if_capped, season, s, e, workers): (s, e) for s, e in chunks}
        for f in concurrent.futures.as_completed(futures):
            try:
                res_list = f.result()
                results.extend(res_list)
                for r in res_list:
                    r.pop("text", None)
            except Exception as e:
                log.warning("chunk %s failed: %r", futures[f], e)
    log.info("Season %d: %d sub-chunks fetched", season, len(results))
    capped_count = 0
    for r in results:
        p = r.get("path")
        if p and os.path.exists(p):
            with open(p) as f:
                text = f.read()
            if len(text.split("\n")) >= CAP_LINE_THRESHOLD:
                capped_count += 1
    log.info("Season %d: %d still capped after re-chunking", season, capped_count)
    return results


def parse_all_chunks(results: list[dict], write_parquet: str | None = None) -> pd.DataFrame:
    """Read all saved CSVs (cached on disk) and return a single DataFrame.
    Streams chunks one at a time to control memory.
    """
    paths = sorted({r["path"] for r in results if r.get("path")})
    log.info("Reading %d chunk files", len(paths))
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
            df = df.dropna(how="all")
            if len(df) == 0:
                continue
            dfs.append(df)
            del df
        except Exception as e:
            log.warning("Failed to read %s: %r", p, e)
    if not dfs:
        return pd.DataFrame()
    big = pd.concat(dfs, ignore_index=True, sort=False)
    big.columns = [c.strip() for c in big.columns]
    n0 = len(big)
    # Dedup on (game_pk, at_bat_number, pitch_number) since the BIP endpoint
    # still returns one row per pitch, but in practice with type=bip each
    # at_bat has exactly one BIP row (the contact pitch).
    keep_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in big.columns]
    if keep_cols:
        big = big.drop_duplicates(subset=keep_cols)
    log.info("After dedupe: %d rows (from %d)", len(big), n0)
    if write_parquet:
        big.to_parquet(write_parquet, index=False)
        log.info("Wrote %s (%.1f MB)", write_parquet, os.path.getsize(write_parquet) / 1e6)
    return big


def main():
    # 2015-2024 ONLY (10 seasons). Do NOT include 2025.
    seasons = list(range(2015, 2025))
    if len(sys.argv) > 1:
        seasons = [int(x) for x in sys.argv[1:]]
    # Safety: refuse to pull 2025.
    if any(s == 2025 or s > 2024 for s in seasons):
        log.error("Refusing to pull 2025 — 2025 holdout rule is binding.")
        sys.exit(2)

    t_start = time.time()
    for s in seasons:
        acquire_season(s, workers=5)
        elapsed = time.time() - t_start
        log.info("Cumulative elapsed: %.1f s", elapsed)
    # Save combined parquet of raw BIP chunks
    paths = []
    for s in seasons:
        paths.extend(glob.glob(os.path.join(RAW_DIR, f"bip_{s}_*_*.csv")))
    if not paths:
        log.error("No CSVs found for seasons %s", seasons)
        sys.exit(1)
    results = [{"path": p} for p in paths]
    out = os.path.join(RAW_DIR, "bip_all_raw.parquet")
    df = parse_all_chunks(results, write_parquet=out)
    if df.empty:
        log.error("No data parsed!")
        sys.exit(1)
    log.info("Wrote %s with %d rows (%.1f MB)", out, len(df), os.path.getsize(out) / 1e6)
    log.info("Total acquisition time: %.1f s", time.time() - t_start)


if __name__ == "__main__":
    main()
