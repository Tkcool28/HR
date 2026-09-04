"""Venue-aware, leakage-safe HR park factors for trusted v1.2.

Input BIP rows must already have actual MLB venue IDs in ``park_id`` and
must be restricted to 2015-2024 regular-season games.

For target season Y, only completed seasons Y-3..Y-1 are used. Rates are
Beta-style empirical-Bayes shrunk toward the league HR/BIP rate over those
same prior seasons. This prevents tiny/special-site samples from creating
extreme factors while allowing established parks to be data-dominant.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
BIP_PATH = ROOT/'data/raw/bip_all.parquet'
OUT = ROOT/'data/curated/park_factors_v12.parquet'
YEARS = range(2015, 2025)
PRIOR_YEARS = 3
PRIOR_BIP_OVERALL = 1000.0
PRIOR_BIP_HAND = 500.0
MIN_FACTOR = 50.0
MAX_FACTOR = 200.0


def log(msg: str) -> None:
    print(f"[park_trusted {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _factor(park_hr: float, park_bip: float, league_hr: float, league_bip: float,
            prior_bip: float) -> tuple[float, float]:
    if park_bip <= 0 or league_bip <= 0:
        return 100.0, 100.0
    league_rate = league_hr / league_bip
    if not np.isfinite(league_rate) or league_rate <= 0:
        return 100.0, 100.0
    raw = (park_hr / park_bip) / league_rate * 100.0
    smoothed_rate = (park_hr + prior_bip * league_rate) / (park_bip + prior_bip)
    shrunk = smoothed_rate / league_rate * 100.0
    shrunk = float(np.clip(shrunk, MIN_FACTOR, MAX_FACTOR))
    return float(raw), shrunk


def main() -> None:
    bip = pd.read_parquet(BIP_PATH)
    bip['game_date'] = pd.to_datetime(bip.game_date)
    bip['year'] = bip.game_date.dt.year.astype(int)
    if not bip.year.between(2015, 2024).all():
        raise RuntimeError('trusted park input escaped 2015-2024')
    if bip.park_id.isna().any():
        raise RuntimeError('trusted park input contains missing actual venue IDs')
    bip['park_id'] = bip.park_id.astype(int)
    bip['is_hr'] = bip.events.eq('home_run').astype('int8')
    bip['batter_hand'] = bip['stand'].astype(str).str.upper().str[:1]
    bip = bip[bip.batter_hand.isin(['L','R'])].copy()

    league = bip.groupby('year', as_index=False).agg(hr=('is_hr','sum'), bip=('is_hr','size'))
    league_h = bip.groupby(['year','batter_hand'], as_index=False).agg(hr=('is_hr','sum'), bip=('is_hr','size'))
    park = bip.groupby(['park_id','year'], as_index=False).agg(hr=('is_hr','sum'), bip=('is_hr','size'))
    park_h = bip.groupby(['park_id','year','batter_hand'], as_index=False).agg(hr=('is_hr','sum'), bip=('is_hr','size'))

    park_ids = sorted(bip.park_id.unique().tolist())
    rows = []
    for pid in park_ids:
        for year in YEARS:
            wanted = set(range(max(2015, year-PRIOR_YEARS), year))
            lp = league[league.year.isin(wanted)]
            pp = park[(park.park_id.eq(pid)) & park.year.isin(wanted)]
            src = sorted(pp.loc[pp.bip.gt(0),'year'].astype(int).unique().tolist())
            raw, overall = _factor(
                float(pp.hr.sum()), float(pp.bip.sum()),
                float(lp.hr.sum()), float(lp.bip.sum()), PRIOR_BIP_OVERALL)

            by_hand = {}
            raw_hand = {}
            prior_hand_bip = {}
            for hand in ['L','R']:
                lph = league_h[(league_h.batter_hand.eq(hand)) & league_h.year.isin(wanted)]
                pph = park_h[(park_h.park_id.eq(pid)) & park_h.batter_hand.eq(hand) & park_h.year.isin(wanted)]
                r, f = _factor(
                    float(pph.hr.sum()), float(pph.bip.sum()),
                    float(lph.hr.sum()), float(lph.bip.sum()), PRIOR_BIP_HAND)
                raw_hand[hand] = r; by_hand[hand] = f; prior_hand_bip[hand] = int(pph.bip.sum())

            rows.append({
                'park_id': int(pid), 'year': int(year),
                'park_hr_factor_3yr_prior': overall,
                'park_hr_factor_by_hand_L_prior': by_hand['L'],
                'park_hr_factor_by_hand_R_prior': by_hand['R'],
                'park_hr_factor_3yr_prior_raw': raw,
                'park_hr_factor_by_hand_L_prior_raw': raw_hand['L'],
                'park_hr_factor_by_hand_R_prior_raw': raw_hand['R'],
                'prior_bip': int(pp.bip.sum()),
                'prior_bip_L': prior_hand_bip['L'], 'prior_bip_R': prior_hand_bip['R'],
                'n_priors_used': len(src),
                'source_years_used': ','.join(str(y) for y in src),
                'shrinkage_prior_bip': PRIOR_BIP_OVERALL,
                'shrinkage_prior_bip_hand': PRIOR_BIP_HAND,
            })

    out = pd.DataFrame(rows)
    expected = len(park_ids) * len(list(YEARS))
    if len(out) != expected:
        raise RuntimeError(f'park grid size mismatch: {len(out)} != {expected}')
    cols = ['park_hr_factor_3yr_prior','park_hr_factor_by_hand_L_prior','park_hr_factor_by_hand_R_prior']
    if not out[cols].apply(lambda s: s.between(MIN_FACTOR, MAX_FACTOR)).all().all():
        raise RuntimeError('shrunk park factor outside safety range')
    for r in out.itertuples(index=False):
        src = [] if not r.source_years_used else [int(x) for x in r.source_years_used.split(',')]
        if not all(r.year-PRIOR_YEARS <= y < r.year for y in src):
            raise RuntimeError(f'future/current source year for venue {r.park_id}, target {r.year}: {src}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    log(f'wrote {len(out):,} venue-year rows for {len(park_ids)} actual venues')
    log(f"shrunk overall factor range {out.park_hr_factor_3yr_prior.min():.1f}-{out.park_hr_factor_3yr_prior.max():.1f}")
    log(f"raw overall factor range {out.park_hr_factor_3yr_prior_raw.min():.1f}-{out.park_hr_factor_3yr_prior_raw.max():.1f}")
    log(f"neutral/no-prior rows: {(out.prior_bip.eq(0)).sum():,}")


if __name__ == '__main__':
    main()
