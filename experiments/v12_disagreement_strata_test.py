"""Multiplicity-aware follow-up for full73-vs-obvious-power disagreements.

The rank bands (5-8, 9-16, 17+) were declared before the first edge-localization
run. Because the 17+ band looked best after observing all three, this script
re-tests all three together rather than promoting that band alone.

For each obvious-power rank band, compare hitters selected into full73's daily
top four against hitters *not* selected by full73 in the same rank band.

Inference:
- 10,000 slate-date bootstrap replicates for paired rate-difference CIs.
- 10,000 exact-within-date randomization replicates under the null that,
  conditional on date and obvious-rank band, HR outcomes are unrelated to
  full73 selection. The observed number selected per date is preserved.
- One-sided randomization p-values test selected > control.
- Holm correction is applied across the three predeclared rank bands.

This is development evidence only (2023-2024). 2025 is rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260904
STRATA = {
    'obvious_rank_5_8': lambda x: x.between(5, 8),
    'obvious_rank_9_16': lambda x: x.between(9, 16),
    'obvious_rank_17_plus': lambda x: x.ge(17),
}


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    need = {
        'game_pk','batter_id','game_date','year','hr_in_game',
        'model_top4','obvious_top4','obvious_rank'
    }
    missing = need - set(frame.columns)
    if missing:
        raise RuntimeError(f'missing columns: {sorted(missing)}')
    f = frame.copy()
    f['game_date'] = pd.to_datetime(f.game_date).dt.normalize()
    if not f.year.isin([2023, 2024]).all():
        raise RuntimeError('disagreement test escaped 2023-2024 development years')
    if f.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate batter-game rows')
    if (f.model_top4 & f.obvious_rank.le(4) & ~f.obvious_top4).any():
        raise RuntimeError('rank/obvious_top4 contract inconsistent')
    return f


def _daily_arrays(frame: pd.DataFrame, band: pd.Series):
    z = frame.loc[band, ['game_date','hr_in_game','model_top4']].copy()
    days = np.array(sorted(frame.game_date.drop_duplicates().to_numpy()))
    pos = {d:i for i,d in enumerate(days)}
    ss = np.zeros(len(days), dtype=np.float64)
    ns = np.zeros(len(days), dtype=np.float64)
    sc = np.zeros(len(days), dtype=np.float64)
    nc = np.zeros(len(days), dtype=np.float64)
    for day, g in z.groupby('game_date', sort=False):
        i = pos[np.datetime64(pd.Timestamp(day).to_datetime64())]
        sel = g.model_top4.to_numpy(dtype=bool)
        y = g.hr_in_game.to_numpy(dtype=np.float64)
        ss[i] = y[sel].sum(); ns[i] = sel.sum()
        sc[i] = y[~sel].sum(); nc[i] = (~sel).sum()
    return days, ss, ns, sc, nc


def _safe_rate(s: np.ndarray, n: np.ndarray) -> float:
    den = float(n.sum())
    if den <= 0:
        raise RuntimeError('empty comparison arm')
    return float(s.sum()/den)


def bootstrap_dates(frame: pd.DataFrame, band: pd.Series, reps: int, seed: int) -> dict:
    days, ss, ns, sc, nc = _daily_arrays(frame, band)
    if len(days) < 100:
        raise RuntimeError(f'too few slate dates: {len(days)}')
    observed_sel = _safe_rate(ss,ns)
    observed_ctl = _safe_rate(sc,nc)
    observed_delta = observed_sel-observed_ctl
    rng = np.random.default_rng(seed)
    rs = np.empty(reps, dtype=np.float64)
    rc = np.empty(reps, dtype=np.float64)
    chunk=1000
    pos=0
    while pos<reps:
        k=min(chunk,reps-pos)
        idx=rng.integers(0,len(days),size=(k,len(days)),endpoint=False)
        ds=ns[idx].sum(axis=1); dc=nc[idx].sum(axis=1)
        rs[pos:pos+k]=ss[idx].sum(axis=1)/ds
        rc[pos:pos+k]=sc[idx].sum(axis=1)/dc
        pos+=k
    delta=rs-rc
    return {
        'n_slate_dates': int(len(days)),
        'n_replicates': int(reps),
        'selected': {
            'n': int(ns.sum()), 'hr': int(ss.sum()), 'rate': observed_sel,
            'ci95_low': float(np.quantile(rs,.025)),
            'ci95_high': float(np.quantile(rs,.975)),
        },
        'control': {
            'n': int(nc.sum()), 'hr': int(sc.sum()), 'rate': observed_ctl,
            'ci95_low': float(np.quantile(rc,.025)),
            'ci95_high': float(np.quantile(rc,.975)),
        },
        'delta_selected_minus_control': {
            'observed': observed_delta,
            'mean': float(delta.mean()),
            'median': float(np.median(delta)),
            'ci95_low': float(np.quantile(delta,.025)),
            'ci95_high': float(np.quantile(delta,.975)),
            'prob_delta_gt_0': float(np.mean(delta>0)),
        },
    }


def randomization_test(frame: pd.DataFrame, band: pd.Series, reps: int, seed: int) -> dict:
    """Exact within-date null via hypergeometric selected-HR counts.

    Conditional on a date's rank-band total N, total HR count H, and observed
    full73-selected count K, random relabeling gives X~Hypergeometric(H,N-H,K)
    selected HRs. Vectorizing these draws is equivalent to explicitly shuffling
    batter labels but much faster.
    """
    z=frame.loc[band,['game_date','hr_in_game','model_top4']].copy()
    rows=[]
    obs_sel=obs_ns=obs_ctl=obs_nc=0
    for _,g in z.groupby('game_date',sort=True):
        y=g.hr_in_game.to_numpy(dtype=np.int8)
        sel=g.model_top4.to_numpy(dtype=bool)
        n=int(len(g)); h=int(y.sum()); k=int(sel.sum())
        if k==n and n>0:
            raise RuntimeError('rank band has date with no control rows')
        rows.append((n,h,k))
        obs_sel += int(y[sel].sum()); obs_ns += k
        obs_ctl += int(y[~sel].sum()); obs_nc += n-k
    if obs_ns<=0 or obs_nc<=0:
        raise RuntimeError('empty randomization comparison arm')
    observed=obs_sel/obs_ns-obs_ctl/obs_nc

    arr=np.asarray(rows,dtype=np.int64)
    n=arr[:,0]; h=arr[:,1]; k=arr[:,2]
    rng=np.random.default_rng(seed)
    null=np.empty(reps,dtype=np.float64)
    chunk=1000
    pos=0
    total_h=int(h.sum()); total_k=int(k.sum()); total_n=int(n.sum())
    while pos<reps:
        q=min(chunk,reps-pos)
        # Broadcasting per-date hypergeometric parameters over q replicates.
        x=rng.hypergeometric(
            ngood=h[None,:],
            nbad=(n-h)[None,:],
            nsample=k[None,:],
            size=(q,len(n)),
        )
        sel_h=x.sum(axis=1)
        ctl_h=total_h-sel_h
        null[pos:pos+q]=sel_h/total_k-ctl_h/(total_n-total_k)
        pos+=q
    # Add-one correction avoids zero Monte Carlo p-values.
    p_one=(1+int(np.sum(null>=observed)))/(reps+1)
    return {
        'n_replicates': int(reps),
        'observed_delta': float(observed),
        'null_mean': float(null.mean()),
        'null_ci95_low': float(np.quantile(null,.025)),
        'null_ci95_high': float(np.quantile(null,.975)),
        'p_one_sided_selected_gt_control': float(p_one),
    }


def holm_adjust(pvals: dict[str,float]) -> dict[str,float]:
    ordered=sorted(pvals.items(),key=lambda kv:kv[1])
    m=len(ordered)
    adjusted={}
    running=0.0
    for i,(name,p) in enumerate(ordered):
        raw=min(1.0,(m-i)*p)
        running=max(running,raw)
        adjusted[name]=min(1.0,running)
    return adjusted


def fingerprints(frame: pd.DataFrame, band: pd.Series) -> dict:
    z=frame.loc[band]
    score_cols=[c for c in ('park_score','recent_batter_form_score','pitcher_vulnerability_score','pitch_matchup_score') if c in z.columns]
    out={}
    for c in score_cols:
        out[c.removesuffix('_score')]={
            'selected_mean': float(z.loc[z.model_top4,c].mean()),
            'control_mean': float(z.loc[~z.model_top4,c].mean()),
            'selected_minus_control': float(z.loc[z.model_top4,c].mean()-z.loc[~z.model_top4,c].mean()),
        }
    return out


def run_scope(frame: pd.DataFrame, reps: int, perm_reps: int, seed: int, run_permutation: bool) -> dict:
    results={}
    raw_p={}
    for i,(name,fn) in enumerate(STRATA.items()):
        band=fn(frame.obvious_rank)
        b=bootstrap_dates(frame,band,reps,seed+i)
        item={'date_cluster_bootstrap':b,'context_fingerprints':fingerprints(frame,band)}
        if run_permutation:
            p=randomization_test(frame,band,perm_reps,seed+100+i)
            item['within_date_randomization']=p
            raw_p[name]=p['p_one_sided_selected_gt_control']
        results[name]=item
    if run_permutation:
        adjusted=holm_adjust(raw_p)
        for name in results:
            results[name]['within_date_randomization']['holm_adjusted_p_across_3_strata']=float(adjusted[name])
    return results


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--reps',type=int,default=10000)
    ap.add_argument('--perm-reps',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=SEED)
    args=ap.parse_args()
    if args.reps<1000 or args.perm_reps<1000:
        raise RuntimeError('requires at least 1000 bootstrap and permutation replicates')
    f=validate(pd.read_parquet(args.input))
    payload={
        'design':{
            'purpose':'follow-up all three predeclared obvious-power disagreement strata together',
            'strata':['5-8','9-16','17+'],
            'selected':'full73 daily top4 within stratum',
            'control':'not full73 top4 within same obvious-power rank stratum',
            'primary_uncertainty':'paired slate-date bootstrap',
            'null_test':'exact within-date hypergeometric label randomization preserving selected count',
            'multiplicity':'Holm correction across three strata',
            'alternative':'selected HR rate > control HR rate',
            'development_years':[2023,2024],
            'bootstrap_replicates':int(args.reps),
            'randomization_replicates':int(args.perm_reps),
            'sealed_final_holdout':'2025',
            'holdout_2025_read':False,
        },
        'combined_2023_2024':run_scope(f,args.reps,args.perm_reps,args.seed,True),
        'by_year':{
            '2023':run_scope(f[f.year.eq(2023)],args.reps,args.perm_reps,args.seed+1000,False),
            '2024':run_scope(f[f.year.eq(2024)],args.reps,args.perm_reps,args.seed+2000,False),
        },
    }
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)
    print('[disagreement-strata] 2025 NOT READ',flush=True)


if __name__=='__main__':
    main()
