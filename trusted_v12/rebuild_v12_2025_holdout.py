"""Authorized trusted v1.2 rebuild through the sealed 2025 holdout.

Requires:
- historical 2015-2024 raw bundles already reconstructed;
- freshly acquired data/raw/pa_2025.parquet;
- freshly acquired data/raw/bip_2025_trusted.parquet;
- delivered build_pa_v12.py and build_pitch_level_v12.py patched ONLY to
  extend their source-year cap through 2025.

This is a data/feature range extension only. Model methodology is frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

import trusted_v12.rebuild_v12_trusted as base

ROOT=Path('/workspace/hr_model')
CUR=ROOT/'data/curated'; RAW=ROOT/'data/raw'
OUT=ROOT/'features/v1.2_2025_holdout'
CTX=CUR/'game_context_v12_holdout.parquet'


def build_context():
    base.YEARS=range(2015,2026)
    base.CTX=CTX
    return base.fetch_context()


def postprocess(ctx):
    ids=set(ctx.game_pk.astype(int)); venue=dict(zip(ctx.game_pk.astype(int),ctx.venue_id.astype(int)))
    base.run(ROOT/'src/data/build_pa_v12.py')
    pa=pd.read_parquet(CUR/'pa_v12.parquet')
    pa=pa[pa.game_pk.astype(int).isin(ids)].copy()
    pa['park_id']=pa.game_pk.astype(int).map(venue).astype('int64')
    pa['game_date']=pd.to_datetime(pa.game_date).dt.normalize()
    if pa.duplicated(['game_pk','at_bat_number']).any(): raise RuntimeError('PA grain duplicate')
    pa.to_parquet(CUR/'pa_v12_holdout.parquet',index=False)
    starters=pd.read_parquet(CUR/'game_starters_v12.parquet')
    starters=starters[starters.game_pk.astype(int).isin(ids)].copy()
    starters.to_parquet(CUR/'game_starters_v12_holdout.parquet',index=False)

    base.run(ROOT/'src/data/build_pitch_level_v12.py')
    pl=pd.read_parquet(CUR/'pitch_level_v12.parquet')
    pl=pl[pl.game_pk.astype(int).isin(ids)].copy()
    pl.to_parquet(CUR/'pitch_level_v12_holdout.parquet',index=False)

    # Park factors are frozen from observations through 2024 only.
    hist=pd.read_parquet(RAW/'bip_all.parquet')
    hist['game_date']=pd.to_datetime(hist.game_date).dt.normalize()
    hist=hist[hist.game_date.dt.year.between(2015,2024)].copy()
    hist=hist[hist.game_pk.astype(int).isin(ids)].copy()
    hist['park_id']=hist.game_pk.astype(int).map(venue).astype('int64')
    hist.to_parquet(RAW/'bip_all.parquet',index=False)
    base.run(Path(__file__).resolve().parent/'process_park_factors_2025_holdout.py')
    pf=pd.read_parquet(CUR/'park_factors_v12.parquet')
    if 2025 not in set(pf.year.astype(int)): raise RuntimeError('2025 park factor row missing')

    # Only after park priors are frozen may earlier-2025 BIPs become available
    # to rolling/season QoC features for later-2025 target dates.
    b25=pd.read_parquet(RAW/'bip_2025_trusted.parquet')
    b25['game_date']=pd.to_datetime(b25.game_date).dt.normalize()
    if not b25.game_date.dt.year.eq(2025).all(): raise RuntimeError('2025 BIP year violation')
    b25=b25[b25.game_pk.astype(int).isin(ids)].copy()
    b25['park_id']=b25.game_pk.astype(int).map(venue).astype('int64')
    combined=pd.concat([hist,b25],ignore_index=True,sort=False)
    if combined.duplicated(['game_pk','batter','pitcher','game_date']).all():
        raise RuntimeError('unexpected pathological BIP duplication')
    combined.to_parquet(RAW/'bip_all.parquet',index=False)
    return pa,starters,pl,pf


def build_core(pa,starters,pl,pf,ctx):
    bf=base.import_delivered_builder()
    pa=pa.copy(); pa['game_date']=pd.to_datetime(pa.game_date)
    pl=pl.copy(); pl['game_date']=pd.to_datetime(pl.game_date)
    targets=base.target_rows(pa,starters,ctx)
    targets['split']=np.select(
        [targets.year.le(2023),targets.year.eq(2024),targets.year.eq(2025)],
        ['fit','calibration','holdout'],default='INVALID')
    if (targets.split=='INVALID').any(): raise RuntimeError('invalid chronological split')
    if not targets.year.between(2015,2025).all(): raise RuntimeError('target year escaped 2015-2025')

    targets,c1=bf.compute_batter_pa_features(targets,pa)
    targets,c2=bf.compute_pitcher_pa_features(targets,pa)
    targets,c3=bf.compute_park_features(targets,pf)
    targets,c4=bf.compute_pitch_type_features(targets,pa)
    targets,c5=bf.compute_pitcher_pitch_usage(targets,pl)
    targets,c6=bf.compute_strength_on_top_pitch(targets,pa)
    targets['max_as_of_date']=bf.compute_max_as_of_date(targets)
    targets['platoon_same_hand']=targets.batter_hand.eq(targets.pitcher_hand).astype('int8')
    features=sorted(set(c1+c2+c3+c4+c5+c6+['platoon_same_hand']))
    features=[c for c in features if c in targets.columns and c!='pitcher_top_pitch']
    if len(features)!=53: raise RuntimeError(f'expected frozen 53 core features, got {len(features)}')
    bad=[c for c in features if not pd.api.types.is_numeric_dtype(targets[c])]
    if bad: raise RuntimeError(f'nonnumeric core features: {bad}')
    asof=[c for c in targets.columns if c.endswith('_as_of')]
    ids=['batter_id','pitcher_id','game_pk','game_date','park_id','batter_hand','pitcher_hand',
         'batter_side','lineup_slot','pitcher_top_pitch','split','year','home_team','away_team',
         'max_as_of_date','hr_in_game']
    out=targets[list(dict.fromkeys(ids+features+asof))].copy()
    if not out.groupby('game_pk').size().eq(18).all(): raise RuntimeError('not 18 targets/game')
    m=pd.to_datetime(out.max_as_of_date).notna()
    if not (pd.to_datetime(out.loc[m,'max_as_of_date']) < pd.to_datetime(out.loc[m,'game_date'])).all():
        raise RuntimeError('core temporal violation')
    if set(out.loc[out.year<=2023,'split'])!={'fit'}: raise RuntimeError('fit split violation')
    if set(out.loc[out.year==2024,'split'])!={'calibration'}: raise RuntimeError('cal split violation')
    if set(out.loc[out.year==2025,'split'])!={'holdout'}: raise RuntimeError('holdout split violation')
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_parquet(OUT/'game_features.parquet',index=False)
    (OUT/'feature_list.json').write_text(json.dumps(features,indent=2))
    summary={
        'n_rows':len(out),'n_games':int(out.game_pk.nunique()),'n_features':len(features),
        'year_range':[int(out.year.min()),int(out.year.max())],
        'splits':out.split.value_counts().to_dict(),
        'holdout_2025_rows':int(out.year.eq(2025).sum()),
        'holdout_2025_games':int(out.loc[out.year.eq(2025),'game_pk'].nunique()),
        'feature_list_sha256':hashlib.sha256((OUT/'feature_list.json').read_bytes()).hexdigest(),
    }
    (OUT/'_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    CUR.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    ctx=build_context(); pa,starters,pl,pf=postprocess(ctx); build_core(pa,starters,pl,pf,ctx)

if __name__=='__main__': main()
