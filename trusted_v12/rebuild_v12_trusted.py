"""Trusted 2015-2024 v1.2 rebuild driver.

Preserves the delivered v1.2 package. Reuses its sound temporal feature
functions but replaces the audited-bad universe/context layers:
- authoritative regular-season game universe from MLB Stats API gameType=R;
- actual per-game venue.id instead of static team->park mapping;
- first nine distinct batters per side instead of the inning-1 proxy;
- fail-closed split membership;
- numeric-only active feature contract.

2025 is never requested or read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path('/workspace/hr_model')
CUR = ROOT/'data/curated'
RAW = ROOT/'data/raw'
SPL = ROOT/'data/splits'
OUT = ROOT/'features/v1.2_trusted'
CTX = CUR/'game_context_v12_trusted.parquet'
YEARS = range(2015, 2025)
API = 'https://statsapi.mlb.com/api/v1/schedule'


def log(msg: str) -> None:
    print(f"[trusted_rebuild {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(path: Path) -> None:
    log(f"run {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def fetch_context() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        params = {'sportId': 1, 'season': year, 'gameType': 'R', 'hydrate': 'venue,team'}
        last = None
        for attempt in range(5):
            try:
                r = requests.get(API, params=params, timeout=60)
                r.raise_for_status(); payload = r.json(); break
            except Exception as exc:
                last = exc; time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"MLB schedule failed for {year}: {last}")
        yr_rows = []
        for d in payload.get('dates', []):
            for g in d.get('games', []):
                venue = g.get('venue') or {}; teams = g.get('teams') or {}
                home = (teams.get('home') or {}).get('team') or {}
                away = (teams.get('away') or {}).get('team') or {}
                yr_rows.append({
                    'game_pk': int(g['gamePk']),
                    'game_date': pd.Timestamp(g.get('officialDate') or d.get('date')),
                    'season': int(year), 'game_type': str(g.get('gameType','')),
                    'venue_id': int(venue['id']) if venue.get('id') is not None else pd.NA,
                    'venue_name': str(venue.get('name') or ''),
                    'home_team_name': str(home.get('name') or ''),
                    'away_team_name': str(away.get('name') or ''),
                })
        if not (800 <= len(yr_rows) <= 2600):
            raise RuntimeError(f"implausible regular schedule count {year}: {len(yr_rows)}")
        log(f"{year}: {len(yr_rows):,} regular schedule records")
        rows.extend(yr_rows)
    c = pd.DataFrame(rows)
    dup = c[c.game_pk.duplicated(keep=False)].copy()
    if not dup.empty:
        identity = ['season','game_type','home_team_name','away_team_name']
        conflicts = dup.groupby('game_pk')[identity].nunique(dropna=False).max(axis=1)
        if conflicts.gt(1).any():
            bad = conflicts[conflicts.gt(1)].index.tolist()[:20]
            raise RuntimeError(f"conflicting duplicate schedule identities: {bad}")
        log(f"collapsing {len(dup):,} duplicate schedule records across {dup.game_pk.nunique():,} gamePk values")
        # Postponed/rescheduled bookkeeping can expose the same gamePk on more
        # than one schedule date. Current schedule data is sorted by date and
        # we retain the latest record for the final played context.
        c = c.sort_values(['game_pk','game_date']).drop_duplicates('game_pk', keep='last')
    if set(c.game_type.unique()) != {'R'}: raise RuntimeError('non-R game in context')
    if c.venue_id.isna().any(): raise RuntimeError('missing venue_id')
    c['game_pk'] = c.game_pk.astype('int64'); c['venue_id'] = c.venue_id.astype('int64')
    c = c.sort_values(['game_date','game_pk']).reset_index(drop=True)
    c.to_parquet(CTX, index=False)
    return c


def postprocess_delivered_tables(ctx: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    ids = set(ctx.game_pk.astype(int)); venue = dict(zip(ctx.game_pk.astype(int), ctx.venue_id.astype(int)))

    run(ROOT/'src/data/build_pa_v12.py')
    pa = pd.read_parquet(CUR/'pa_v12.parquet')
    pa = pa[pa.game_pk.astype(int).isin(ids)].copy()
    pa['park_id'] = pa.game_pk.astype(int).map(venue).astype('int64')
    pa['game_date'] = pd.to_datetime(pa.game_date).dt.normalize()
    if pa.duplicated(['game_pk','at_bat_number']).any(): raise RuntimeError('PA grain duplicate')
    pa.to_parquet(CUR/'pa_v12_trusted.parquet', index=False)
    starters = pd.read_parquet(CUR/'game_starters_v12.parquet')
    starters = starters[starters.game_pk.astype(int).isin(ids)].copy()
    starters.to_parquet(CUR/'game_starters_v12_trusted.parquet', index=False)

    run(ROOT/'src/data/build_pitch_level_v12.py')
    pl = pd.read_parquet(CUR/'pitch_level_v12.parquet')
    pl = pl[pl.game_pk.astype(int).isin(ids)].copy()
    pl.to_parquet(CUR/'pitch_level_v12_trusted.parquet', index=False)

    bip = pd.read_parquet(RAW/'bip_all.parquet')
    before = len(bip)
    bip = bip[bip.game_pk.astype(int).isin(ids)].copy()
    bip['park_id'] = bip.game_pk.astype(int).map(venue).astype('int64')
    log(f"BIP regular-season filter: {len(bip):,}/{before:,}; removed {before-len(bip):,}")
    bip.to_parquet(RAW/'bip_all.parquet', index=False)
    run(ROOT/'src/data/process_park_factors_v12.py')
    pf = pd.read_parquet(CUR/'park_factors_v12.parquet')
    pf.to_parquet(CUR/'park_factors_v12_trusted.parquet', index=False)
    return pa, starters, pl, pf


def import_delivered_builder():
    path = ROOT/'features/v1.2/build_features.py'
    spec = importlib.util.spec_from_file_location('delivered_build_v12', path)
    mod = importlib.util.module_from_spec(spec); assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def target_rows(pa: pd.DataFrame, starters: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    p = pa.sort_values(['game_pk','at_bat_number']).copy()
    first = p.drop_duplicates(['game_pk','batter_id'], keep='first').copy()
    first['batter_side'] = first.inning_topbot.map({'Top':'away','Bot':'home'})
    first = first.dropna(subset=['batter_side']).sort_values(['game_pk','batter_side','at_bat_number'])
    first['lineup_slot'] = first.groupby(['game_pk','batter_side'], sort=False).cumcount() + 1
    counts = first.groupby(['game_pk','batter_side']).size().unstack(fill_value=0)
    for side in ['home','away']:
        if side not in counts: counts[side] = 0
    lineup_games = set(counts.index[(counts.home >= 9) & (counts.away >= 9)])
    sp = starters.dropna(subset=['starting_pitcher_home','starting_pitcher_away']).copy()
    valid = lineup_games & set(sp.game_pk.astype(int)) & set(ctx.game_pk.astype(int))
    first9 = first[first.game_pk.isin(valid) & first.lineup_slot.le(9)].copy()
    if not first9.groupby(['game_pk','batter_side']).size().eq(9).all():
        raise RuntimeError('not exactly nine reconstructed batters per side')

    hand = pa.dropna(subset=['batter_hand']).groupby(['game_pk','batter_id'], sort=False).batter_hand.first()
    idx = pd.MultiIndex.from_frame(first9[['game_pk','batter_id']])
    fill = pd.Series(hand.reindex(idx).to_numpy(), index=first9.index, dtype='string')
    first9['batter_hand'] = first9.batter_hand.fillna(fill)

    hr = pa.groupby(['game_pk','batter_id'], as_index=False, sort=False).is_hr.max().rename(columns={'is_hr':'hr_in_game'})
    rows = first9[['game_pk','batter_id','batter_side','lineup_slot','batter_hand']].merge(
        hr, on=['game_pk','batter_id'], how='left', validate='one_to_one')
    rows['hr_in_game'] = rows.hr_in_game.fillna(0).astype('int8')
    rows = rows.merge(sp, on='game_pk', how='inner', validate='many_to_one')
    rows['pitcher_id'] = np.where(rows.batter_side.eq('home'), rows.starting_pitcher_away, rows.starting_pitcher_home).astype('int64')
    rows = rows.drop(columns=['starting_pitcher_home','starting_pitcher_away'])
    ph = pa.dropna(subset=['pitcher_hand']).groupby('pitcher_id', as_index=False, sort=False).pitcher_hand.first()
    rows = rows.merge(ph, on='pitcher_id', how='left', validate='many_to_one')
    if rows[['batter_hand','pitcher_hand']].isna().any().any(): raise RuntimeError('missing target hand')
    rows = rows.merge(ctx, on='game_pk', how='inner', validate='many_to_one')
    rows['park_id'] = rows.venue_id.astype('int64')
    rows['home_team'] = rows.home_team_name; rows['away_team'] = rows.away_team_name
    rows['game_date'] = pd.to_datetime(rows.game_date).dt.normalize(); rows['year'] = rows.season.astype('int16')
    if not rows.groupby('game_pk').size().eq(18).all(): raise RuntimeError('not 18 targets/game')
    log(f"trusted targets: {len(rows):,} rows = {rows.game_pk.nunique():,} games × 18")
    return rows


def build_features(pa, starters, pl, pf, ctx) -> pd.DataFrame:
    bf = import_delivered_builder()
    pa = pa.copy(); pa['game_date'] = pd.to_datetime(pa.game_date)
    pl = pl.copy(); pl['game_date'] = pd.to_datetime(pl.game_date)
    targets = target_rows(pa, starters, ctx)
    train_ids = set(pd.read_parquet(SPL/'train_ids.parquet').game_pk.astype(int))
    val_ids = set(pd.read_parquet(SPL/'val_ids.parquet').game_pk.astype(int))
    intr = targets.game_pk.isin(train_ids); inv = targets.game_pk.isin(val_ids)
    bad = intr.eq(inv)
    if bad.any():
        raise RuntimeError(f"split membership must be exactly one; games={targets.loc[bad,'game_pk'].drop_duplicates().head(20).tolist()}")
    targets['split'] = np.where(intr, 'train', 'val')
    if not targets.year.between(2015,2024).all(): raise RuntimeError('year escaped')

    targets,c1 = bf.compute_batter_pa_features(targets, pa)
    targets,c2 = bf.compute_pitcher_pa_features(targets, pa)
    targets,c3 = bf.compute_park_features(targets, pf)
    targets,c4 = bf.compute_pitch_type_features(targets, pa)
    targets,c5 = bf.compute_pitcher_pitch_usage(targets, pl)
    targets,c6 = bf.compute_strength_on_top_pitch(targets, pa)
    targets['max_as_of_date'] = bf.compute_max_as_of_date(targets)
    targets['platoon_same_hand'] = targets.batter_hand.eq(targets.pitcher_hand).astype('int8')

    feature_cols = sorted(set(c1+c2+c3+c4+c5+c6+['platoon_same_hand']))
    feature_cols = [c for c in feature_cols if c in targets.columns and c != 'pitcher_top_pitch']
    nonnumeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(targets[c])]
    if nonnumeric: raise RuntimeError(f"nonnumeric active features: {nonnumeric}")

    asof = [c for c in targets.columns if c.endswith('_as_of')]
    ids = ['batter_id','pitcher_id','game_pk','game_date','park_id','batter_hand','pitcher_hand',
           'batter_side','lineup_slot','pitcher_top_pitch','split','year','home_team','away_team',
           'max_as_of_date','hr_in_game']
    out = targets[list(dict.fromkeys(ids+feature_cols+asof))].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT/'game_features.parquet', index=False)
    (OUT/'feature_list.json').write_text(json.dumps(feature_cols, indent=2))
    summary = {'n_rows':len(out),'n_games':int(out.game_pk.nunique()),'n_features':len(feature_cols),
               'splits':out.split.value_counts().to_dict(),'year_range':[int(out.year.min()),int(out.year.max())],
               'hr_rate_overall':float(out.hr_in_game.mean()),
               'feature_list_sha256':hashlib.sha256((OUT/'feature_list.json').read_bytes()).hexdigest()}
    (OUT/'_summary.json').write_text(json.dumps(summary, indent=2))
    log(f"features: {len(out):,} rows × {len(feature_cols)} numeric active features")
    return out


def validate(ctx, pa, pl, pf, out) -> None:
    ids = set(ctx.game_pk.astype(int))
    assert set(ctx.game_type.unique()) == {'R'}
    assert not (set(pa.game_pk.astype(int)) - ids)
    assert not (set(pl.game_pk.astype(int)) - ids)
    assert out.groupby('game_pk').size().eq(18).all()
    assert out.groupby(['game_pk','batter_side']).size().eq(9).all()
    assert out.lineup_slot.between(1,9).all()
    assert set(out.loc[out.year<=2022,'split']) == {'train'}
    assert set(out.loc[out.year>=2023,'split']) == {'val'}
    nonnat = out.dropna(subset=['max_as_of_date'])
    assert (pd.to_datetime(nonnat.max_as_of_date) < pd.to_datetime(nonnat.game_date)).all()
    def venues(team, year): return set(ctx[(ctx.home_team_name==team)&(ctx.season==year)].venue_id)
    assert venues('Atlanta Braves',2016) != venues('Atlanta Braves',2017)
    assert venues('Texas Rangers',2019) != venues('Texas Rangers',2020)
    assert ctx[(ctx.home_team_name=='Toronto Blue Jays')&(ctx.season==2021)].venue_id.nunique() >= 2
    for r in pf.itertuples(index=False):
        src = [] if not r.source_years_used else [int(x) for x in str(r.source_years_used).split(',') if x]
        assert all(r.year-3 <= y < r.year for y in src)
    log('trusted structural validation PASS')


def main() -> None:
    CUR.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    ctx = fetch_context()
    pa,starters,pl,pf = postprocess_delivered_tables(ctx)
    out = build_features(pa,starters,pl,pf,ctx)
    validate(ctx,pa,pl,pf,out)


if __name__ == '__main__':
    main()
