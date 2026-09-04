from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd


def summarize(df: pd.DataFrame, year: int):
    d=df.copy()
    if 'year' in d.columns:
        d=d[d['year'].astype(int)==year]
    elif 'game_date' in d.columns:
        d=d[pd.to_datetime(d['game_date']).dt.year==year]
    rank_col='model_rank' if 'model_rank' in d.columns else None
    if rank_col is None and 'raw_rank' in d.columns: rank_col='raw_rank'
    if rank_col is None:
        raise RuntimeError('No model rank column found')
    out={'year':year,'rows':len(d),'slates':int(pd.to_datetime(d['game_date']).dt.date.nunique())}
    for n in (1,2,4,8):
        x=d[d[rank_col].astype(float)<=n]
        out[f'top{n}_picks']=len(x)
        out[f'top{n}_unique_games']=int(x['game_pk'].nunique())
    if 'model_top5' in d.columns:
        model_top5_mask=d['model_top5'].astype(bool)
    else:
        sizes=d.groupby('game_date').size().map(lambda n:max(1,int(math.ceil(n*.05))))
        model_top5_mask=d.apply(lambda r: float(r[rank_col])<=sizes.loc[r['game_date']], axis=1)
    x=d[model_top5_mask]
    out['top5pct_picks']=len(x)
    out['top5pct_unique_games']=int(x['game_pk'].nunique())
    if 'obvious_top5' in d.columns:
        u=d[model_top5_mask | d['obvious_top5'].astype(bool)]
        out['union_model_obvious_top5_picks']=len(u)
        out['union_model_obvious_top5_unique_games']=int(u['game_pk'].nunique())
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--historical',required=True)
    ap.add_argument('--holdout',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    h=pd.read_parquet(a.historical)
    q=pd.read_parquet(a.holdout)
    rows=[]
    for y in (2023,2024): rows.append(summarize(h,y))
    rows.append(summarize(q,2025))
    result={'years':rows,'combined':{}}
    for key in ['top1_unique_games','top2_unique_games','top4_unique_games','top8_unique_games','top5pct_unique_games','union_model_obvious_top5_unique_games']:
        vals=[r.get(key) for r in rows]
        if all(v is not None for v in vals):
            total=int(sum(vals))
            result['combined'][key]=total
            result['combined'][key.replace('_unique_games','_credits_one_snapshot')]=total*10
            result['combined'][key.replace('_unique_games','_credits_two_snapshots')]=total*20
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
