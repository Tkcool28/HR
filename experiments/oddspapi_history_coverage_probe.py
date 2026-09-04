"""Bounded OddsPapi historical MLB HR-prop coverage probe.

Purpose: discover real archive depth and schema before any bulk acquisition.
Uses a handful of fixed July dates across 2022-2026. Never prints the API key.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

BASE = "https://api.oddspapi.io/v4"
BASEBALL_SPORT_ID = 13
HR_MARKET_ID = "131663"
BOOKS = ["draftkings", "fanduel", "caesars"]
PROBE_DATES = {
    2022: "2022-07-01",
    2023: "2023-07-01",
    2024: "2024-07-01",
    2025: "2025-07-01",
    2026: "2026-07-01",  # positive-era control; docs guarantee history since Jan 2026
}


def get_json(session: requests.Session, path: str, api_key: str, **params):
    params = {"apiKey": api_key, **params}
    r = session.get(f"{BASE}/{path}", params=params, timeout=60)
    out = {"status": r.status_code, "etag": r.headers.get("ETag")}
    try:
        out["json"] = r.json()
    except Exception:
        out["json"] = None
        out["body_preview"] = r.text[:300]
    return out


def is_mlb_fixture(f: dict) -> bool:
    name = str(f.get("tournamentName", "")).lower()
    return "mlb" in name or "major league baseball" in name


def count_hr_players(hist: dict, book: str) -> tuple[int, int]:
    if not isinstance(hist, dict):
        return 0, 0
    bm = hist.get("bookmakers", {}).get(book, {})
    market = bm.get("markets", {}).get(HR_MARKET_ID, {})
    outcomes = market.get("outcomes", {}) if isinstance(market, dict) else {}
    player_count = 0
    snapshot_count = 0
    for outcome in outcomes.values():
        players = outcome.get("players", {}) if isinstance(outcome, dict) else {}
        for pid, snaps in players.items():
            if str(pid) == "0" or not isinstance(snaps, list):
                continue
            if snaps:
                player_count += 1
                snapshot_count += len(snaps)
    return player_count, snapshot_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    api_key = os.environ.get("ODDSPAPI_API_KEY")
    if not api_key:
        raise SystemExit("ODDSPAPI_API_KEY missing")

    s = requests.Session()
    result = {
        "probe": "OddsPapi MLB historical HR-prop archive depth",
        "sport_id": BASEBALL_SPORT_ID,
        "hr_market_id": HR_MARKET_ID,
        "bookmakers": BOOKS,
        "dates": {},
    }

    # Verify market catalog semantically without hard-relying on the blog ID.
    mk = get_json(s, "markets", api_key, language="en")
    if mk["status"] != 200 or not isinstance(mk.get("json"), list):
        raise RuntimeError(f"markets discovery failed status={mk['status']}")
    hr_catalog = [m for m in mk["json"] if str(m.get("marketId")) == HR_MARKET_ID]
    result["market_catalog_match"] = [
        {k: m.get(k) for k in ("marketId", "marketName", "playerProp", "sportId", "handicap", "period", "marketType")}
        for m in hr_catalog
    ]

    for year, day in PROBE_DATES.items():
        fx = get_json(
            s,
            "fixtures",
            api_key,
            sportId=BASEBALL_SPORT_ID,
            **{"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z", "statusId": 2},
        )
        fixtures = fx.get("json") if fx["status"] == 200 and isinstance(fx.get("json"), list) else []
        mlb = [f for f in fixtures if is_mlb_fixture(f)]
        chosen = mlb[0] if mlb else (fixtures[0] if fixtures else None)
        yr = {
            "date": day,
            "fixtures_status": fx["status"],
            "fixtures_total": len(fixtures),
            "mlb_named_total": len(mlb),
            "chosen_fixture": None,
            "books": {},
        }
        if chosen:
            yr["chosen_fixture"] = {
                k: chosen.get(k)
                for k in ("fixtureId", "tournamentName", "participant1Name", "participant2Name", "startTime", "statusId", "hasOdds")
            }
            fid = chosen.get("fixtureId")
            for book in BOOKS:
                # Historical endpoint is free but rate-limited to one request / 5 seconds.
                h = get_json(s, "historical-odds", api_key, fixtureId=fid, bookmakers=book)
                players, snaps = count_hr_players(h.get("json"), book)
                body = h.get("json")
                bookmakers_returned = sorted(body.get("bookmakers", {}).keys()) if isinstance(body, dict) else []
                yr["books"][book] = {
                    "status": h["status"],
                    "bookmakers_returned": bookmakers_returned,
                    "hr_market_present": players > 0,
                    "hr_player_count": players,
                    "hr_snapshot_count": snaps,
                    "etag_present": bool(h.get("etag")),
                }
                time.sleep(5.2)
        result["dates"][str(year)] = yr

    # No secret or raw payloads are persisted.
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
