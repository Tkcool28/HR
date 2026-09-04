"""2025 holdout park-factor extension for frozen trusted v1.2.

Uses ONLY regular-season BIP observations through 2024. Target-year 2025
factors therefore use 2022-2024, exactly matching the frozen prior-3-year
method. Venue IDs come from the authoritative 2015-2025 context so a new
2025 venue with no history receives the neutral 100 prior rather than being
silently absent.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/workspace/hr_model')
BIP=ROOT/'data/raw/bip_all.parquet'
CTX=ROOT/'data/curated/game_context_v12_holdout.parquet'
OUT=ROOT/'data/curated/park_factors_v12.parquet'
YEARS=range(2015,2026)
PRIOR_YEARS=3
PRIOR_BIP_OVERALL=1000.0
PRIOR_BIP_HAND=500.0
MIN_FACTOR=50.0
MAX_FACTOR=200.0


def _factor(park_hr,park_bip,league_hr,league_bip,prior_bip):
    if park_bip<=0 or league_bip<=0: return 100.0,100.0
    lr=league_hr/league_bip
    if not np.isfinite(lr) or lr<=0: return 100.0,100.0
    raw=(park_hr/park_bip)/lr*100.0
    sm=(park_hr+prior_bip*lr)/(park_bip+prior_bip)
    return float(raw),float(np.clip(sm/lr*100.0,MIN_FACTOR,MAX_FACTOR))


def main():
    bip=pd.read_parquet(BIP)
    bip['game_date']=pd.to_datetime(bip.game_date)
    bip['year']=bip.game_date.dt.year.astype(int)
    if not bip.year.between(2015,2024).all():
        raise RuntimeError('2025 park source must be historical <=2024 only')
    ctx=pd.read_parquet(CTX)
    if not ctx.season.between(2015,2025).all(): raise RuntimeError('context escaped 2015-2025')
    if bip.park_id.isna().any() or ctx.venue_id.isna().any(): raise RuntimeError('missing venue id')
    bip['park_id']=bip.park_id.astype(int)
    bip['is_hr']=bip.events.eq('home_run').astype('int8')
    bip['batter_hand']=bip['stand'].astype(str).str.upper().str[:1]
    bip=bip[bip.batter_hand.isin(['L','R'])].copy()
    league=bip.groupby('year',as_index=False).agg(hr=('is_hr','sum'),bip=('is_hr','size'))
    league_h=bip.groupby(['year','batter_hand'],as_index=False).agg(hr=('is_hr','sum'),bip=('is_hr','size'))
    park=bip.groupby(['park_id','year'],as_index=False).agg(hr=('is_hr','sum'),bip=('is_hr','size'))
    park_h=bip.groupby(['park_id','year','batter_hand'],as_index=False).agg(hr=('is_hr','sum'),bip=('is_hr','size'))
    park_ids=sorted(set(bip.park_id.astype(int))|set(ctx.venue_id.astype(int)))
    rows=[]
    for pid in park_ids:
        for year in YEARS:
            wanted=set(range(max(2015,year-PRIOR_YEARS),year))
            lp=league[league.year.isin(wanted)]
            pp=park[park.park_id.eq(pid)&park.year.isin(wanted)]
            src=sorted(pp.loc[pp.bip.gt(0),'year'].astype(int).unique().tolist())
            raw,overall=_factor(float(pp.hr.sum()),float(pp.bip.sum()),float(lp.hr.sum()),float(lp.bip.sum()),PRIOR_BIP_OVERALL)
            byh={}; rawh={}; nb={}
            for hand in ['L','R']:
                l=league_h[league_h.batter_hand.eq(hand)&league_h.year.isin(wanted)]
                p=park_h[park_h.park_id.eq(pid)&park_h.batter_hand.eq(hand)&park_h.year.isin(wanted)]
                r,f=_factor(float(p.hr.sum()),float(p.bip.sum()),float(l.hr.sum()),float(l.bip.sum()),PRIOR_BIP_HAND)
                rawh[hand]=r; byh[hand]=f; nb[hand]=int(p.bip.sum())
            rows.append({
                'park_id':pid,'year':year,
                'park_hr_factor_3yr_prior':overall,
                'park_hr_factor_by_hand_L_prior':byh['L'],
                'park_hr_factor_by_hand_R_prior':byh['R'],
                'park_hr_factor_3yr_prior_raw':raw,
                'park_hr_factor_by_hand_L_prior_raw':rawh['L'],
                'park_hr_factor_by_hand_R_prior_raw':rawh['R'],
                'prior_bip':int(pp.bip.sum()),'prior_bip_L':nb['L'],'prior_bip_R':nb['R'],
                'n_priors_used':len(src),'source_years_used':','.join(map(str,src)),
                'shrinkage_prior_bip':PRIOR_BIP_OVERALL,'shrinkage_prior_bip_hand':PRIOR_BIP_HAND,
            })
    out=pd.DataFrame(rows)
    if len(out)!=len(park_ids)*len(list(YEARS)): raise RuntimeError('park grid mismatch')
    for r in out.itertuples(index=False):
        src=[] if not r.source_years_used else [int(x) for x in r.source_years_used.split(',')]
        if not all(r.year-3<=y<r.year for y in src): raise RuntimeError('future/current park source')
    y25=out[out.year.eq(2025)]
    if y25.empty: raise RuntimeError('missing 2025 park grid')
    for s in y25.source_years_used:
        ys=[] if not s else [int(x) for x in s.split(',')]
        if not all(y in (2022,2023,2024) for y in ys): raise RuntimeError(f'2025 park source escaped 2022-24: {ys}')
    OUT.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(OUT,index=False)
    print(f'[park_holdout] wrote {len(out)} rows; parks={len(park_ids)}; 2025 rows={len(y25)}')
    print(f'[park_holdout] neutral 2025 parks={int(y25.prior_bip.eq(0).sum())}; source strictly <=2024')

if __name__=='__main__': main()
