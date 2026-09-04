"""Authorized one-shot 2025 Statcast acquisition for trusted v1.2.

This script is executed only after the 2025 freeze contract was committed.
It reproduces the original Baseball Savant source lineage but fixes scope at
the boundary by retaining only game_pk values from the authoritative MLB
Stats API regular-season schedule.

Outputs:
- data/raw/pa_2025.parquet               pitch-grain details feed
- data/raw/bip_2025_trusted.parquet      per-BIP QoC feed
- data/curated/game_context_2025_authorized.parquet
- data/raw/holdout_2025_acquisition.json
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path('/workspace/hr_model')
RAW = ROOT/'data/raw'
CUR = ROOT/'data/curated'
SCHEDULE_URL = 'https://statsapi.mlb.com/api/v1/schedule'
SAVANT = 'https://baseballsavant.mlb.com/statcast_search/csv'
YEAR = 2025
CAP = 24_500

BIP_EVENTS = {
    'single','double','triple','home_run','field_out','grounded_into_double_play',
    'force_out','fielders_choice','fielders_choice_out','sac_bunt','sac_fly',
    'sac_fly_double_play','sac_bunt_double_play','field_error','triple_play',
    'double_play','other_out',
}
PITCH_COLS = [
    'game_pk','at_bat_number','pitch_number','batter','pitcher','game_date',
    'home_team','away_team','stand','p_throws','events','inning','inning_topbot',
    'balls','strikes','outs_when_up','pitch_type',
]
BIP_COLS = [
    'game_pk','game_date','at_bat_number','pitch_number','batter','pitcher',
    'home_team','away_team','stand','p_throws','events','description','bb_type',
    'launch_speed','launch_angle','hit_distance_sc','woba_value',
    'estimated_ba_using_speedangle','estimated_woba_using_speedangle',
    'bat_speed','swing_length',
]


def _get(url, *, params=None, attempts=5, timeout=120):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            time.sleep(2 ** i)
    raise RuntimeError(f'GET failed after {attempts} attempts: {last}')


def fetch_context() -> pd.DataFrame:
    payload = _get(SCHEDULE_URL, params={
        'sportId':1,'season':YEAR,'gameType':'R','hydrate':'venue,team'
    }, timeout=60).json()
    rows=[]
    for d in payload.get('dates',[]):
        for g in d.get('games',[]):
            venue=g.get('venue') or {}; teams=g.get('teams') or {}
            home=(teams.get('home') or {}).get('team') or {}
            away=(teams.get('away') or {}).get('team') or {}
            rows.append({
                'game_pk':int(g['gamePk']),
                'game_date':pd.Timestamp(g.get('officialDate') or d.get('date')).normalize(),
                'season':YEAR,'game_type':str(g.get('gameType','')),
                'venue_id':int(venue['id']) if venue.get('id') is not None else pd.NA,
                'venue_name':str(venue.get('name') or ''),
                'home_team_name':str(home.get('name') or ''),
                'away_team_name':str(away.get('name') or ''),
            })
    c=pd.DataFrame(rows)
    dup=c[c.game_pk.duplicated(keep=False)]
    if not dup.empty:
        ident=['season','game_type','home_team_name','away_team_name']
        conflicts=dup.groupby('game_pk')[ident].nunique(dropna=False).max(axis=1)
        if conflicts.gt(1).any():
            raise RuntimeError(f'conflicting duplicate schedule identities: {conflicts[conflicts.gt(1)].index.tolist()[:20]}')
        c=c.sort_values(['game_pk','game_date']).drop_duplicates('game_pk',keep='last')
    if c.empty or set(c.game_type.unique()) != {'R'} or c.venue_id.isna().any():
        raise RuntimeError('invalid authoritative 2025 regular-season context')
    if not (2300 <= len(c) <= 2550):
        raise RuntimeError(f'implausible 2025 schedule size: {len(c)}')
    c['venue_id']=c.venue_id.astype('int64')
    c=c.sort_values(['game_date','game_pk']).reset_index(drop=True)
    CUR.mkdir(parents=True,exist_ok=True)
    c.to_parquet(CUR/'game_context_2025_authorized.parquet',index=False)
    return c


def _windows(first: pd.Timestamp, last: pd.Timestamp, days=3):
    # Savant's historical query uses game_date_gt/game_date_lt. Give every
    # primary window a one-day halo and dedupe afterward so boundary dates
    # cannot be lost because of endpoint inclusivity semantics.
    d=first.date(); end=(last+pd.Timedelta(days=1)).date()
    while d < end:
        e=min(d+dt.timedelta(days=days),end)
        yield (d-dt.timedelta(days=1), e+dt.timedelta(days=1))
        d=e


def _fetch_savant(kind: str, season: int, start: dt.date, end: dt.date):
    params={
        'all':'true','type':kind,'game_type':'R','season':season,
        'game_date_gt':start.isoformat(),'game_date_lt':end.isoformat(),
        'min_pa':0,'min_results':0,'group_by':'name','sort_col':'pitches',
        'player_event_sort':'api_p_release_speed','sort_order':'desc',
    }
    text=_get(SAVANT,params=params).text
    n=max(0,len(text.splitlines())-1)
    if n >= CAP:
        if (end-start).days <= 2:
            raise RuntimeError(f'Savant {kind} chunk still capped at minimum window {start}..{end}: {n}')
        mid=start+(end-start)//2
        a=_fetch_savant(kind,season,start,mid)
        b=_fetch_savant(kind,season,mid,end)
        return a+b
    return [(start.isoformat(),end.isoformat(),text,n)]


def acquire_kind(kind: str, ctx: pd.DataFrame):
    first=ctx.game_date.min(); last=ctx.game_date.max()
    wins=list(_windows(first,last))
    chunks=[]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs=[ex.submit(_fetch_savant,kind,YEAR,s,e) for s,e in wins]
        for f in cf.as_completed(futs):
            chunks.extend(f.result())
    if not chunks:
        raise RuntimeError(f'no Savant chunks for {kind}')
    frames=[]
    for s,e,text,n in chunks:
        if not text.strip():
            continue
        z=pd.read_csv(io.StringIO(text),low_memory=False).dropna(how='all')
        if len(z): frames.append(z)
    if not frames:
        raise RuntimeError(f'no parsed Savant rows for {kind}')
    df=pd.concat(frames,ignore_index=True,sort=False)
    df.columns=[str(c).strip() for c in df.columns]
    ids=set(ctx.game_pk.astype(int))
    df['game_pk']=pd.to_numeric(df.game_pk,errors='coerce')
    df=df[df.game_pk.notna()].copy(); df['game_pk']=df.game_pk.astype('int64')
    before=len(df)
    df=df[df.game_pk.isin(ids)].copy()
    return df, {'n_chunks':len(chunks),'n_rows_before_schedule_filter':before,'n_rows_regular_schedule':len(df)}


def main():
    RAW.mkdir(parents=True,exist_ok=True); CUR.mkdir(parents=True,exist_ok=True)
    ctx=fetch_context(); venue=dict(zip(ctx.game_pk.astype(int),ctx.venue_id.astype(int)))

    details,dm=acquire_kind('details',ctx)
    missing=[c for c in PITCH_COLS if c not in details.columns]
    if missing: raise RuntimeError(f'details feed missing required columns: {missing}')
    pitch=details[PITCH_COLS].copy()
    pitch=pitch.dropna(subset=['game_pk','at_bat_number','pitch_number'])
    pitch=pitch.drop_duplicates(['game_pk','at_bat_number','pitch_number'])
    pitch['game_date']=pd.to_datetime(pitch.game_date).dt.normalize()
    if not pitch.game_date.dt.year.eq(YEAR).all(): raise RuntimeError('details escaped 2025')
    n_games=int(pitch.game_pk.nunique())
    if n_games < 2400: raise RuntimeError(f'too few 2025 regular games in details feed: {n_games}')
    pitch.to_parquet(RAW/'pa_2025.parquet',index=False)

    br, bm=acquire_kind('bip',ctx)
    missing=[c for c in BIP_COLS if c not in br.columns]
    if missing: raise RuntimeError(f'BIP feed missing required columns: {missing}')
    b=br[BIP_COLS].copy()
    b=b[b.description.astype('string').eq('hit_into_play')].copy()
    b=b.drop_duplicates(['game_pk','at_bat_number'],keep='first')
    b['events']=b.events.astype('string').str.strip().str.lower()
    b=b[b.events.isin(BIP_EVENTS) & b.launch_speed.notna()].copy()
    b['game_date']=pd.to_datetime(b.game_date).dt.normalize()
    b['park_id']=b.game_pk.map(venue).astype('int64')
    if not b.game_date.dt.year.eq(YEAR).all(): raise RuntimeError('BIP escaped 2025')
    if len(b) < 110_000: raise RuntimeError(f'implausibly small 2025 BIP set: {len(b)}')
    b.to_parquet(RAW/'bip_2025_trusted.parquet',index=False)

    meta={
        'authorized_year':YEAR,
        'schedule_unique_game_pks':int(ctx.game_pk.nunique()),
        'schedule_min_date':str(ctx.game_date.min().date()),
        'schedule_max_date':str(ctx.game_date.max().date()),
        'details':{**dm,'dedup_regular_pitch_rows':int(len(pitch)),'regular_game_pks':n_games},
        'bip':{**bm,'processed_per_bip_rows':int(len(b)),'regular_game_pks':int(b.game_pk.nunique())},
        'scope_filter':'authoritative MLB gameType=R game_pk intersection',
        'historical_bip_contract':'description=hit_into_play; dedupe game_pk+at_bat_number; 17-event whitelist; launch_speed non-null',
    }
    (RAW/'holdout_2025_acquisition.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    main()
