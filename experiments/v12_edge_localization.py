"""Localize where the frozen full73 HR model earns actionable tail edge.

This does NOT claim to reconstruct historical public betting consensus or HR
prop prices.  Instead it builds an intentionally simple "obvious power" proxy
from long-horizon batter-only power/contact-quality features.  It then asks:

- how much does the full73 top-4 overlap the obvious-power top-4?
- how often do full73-only (differentiated) picks homer?
- how do those picks compare with obvious-power-only picks on the same dates?
- which contextual feature groups are elevated in the differentiated picks?

The point is to determine whether the model's practical value comes only from
ranking familiar sluggers, or whether park/pitcher/recent-form/matchup context
finds useful candidates the batter-only proxy would not select.

All inference is 2023-2024 development evidence.  2025 is rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
FEAT = ROOT/'features/v1.2_trusted/game_features.parquet'
FLIST = ROOT/'features/v1.2_trusted/feature_list.json'
SEED = 20260904


def _top_n(g: pd.DataFrame, score: str, n: int = 4) -> set[tuple[int,int]]:
    z = g.sort_values([score, 'game_pk', 'batter_id'], ascending=[False, True, True]).head(n)
    return set(zip(z.game_pk.astype(int), z.batter_id.astype(int)))


def _rate(frame: pd.DataFrame, mask: pd.Series) -> dict:
    x = frame.loc[mask]
    return {
        'n': int(len(x)),
        'n_dates': int(x.game_date.nunique()) if len(x) else 0,
        'hr': int(x.hr_in_game.sum()) if len(x) else 0,
        'hr_rate': float(x.hr_in_game.mean()) if len(x) else None,
    }


def _bootstrap_segment_rates(frame: pd.DataFrame, masks: dict[str,pd.Series], reps: int, seed: int) -> dict:
    dates = np.array(sorted(frame.game_date.drop_duplicates().to_numpy()))
    if len(dates) < 100:
        raise RuntimeError(f'too few dates for edge bootstrap: {len(dates)}')
    date_pos = {d:i for i,d in enumerate(dates)}
    stats: dict[str, tuple[np.ndarray,np.ndarray]] = {}
    for name, mask in masks.items():
        x = frame.loc[mask, ['game_date','hr_in_game']]
        agg = x.groupby('game_date').hr_in_game.agg(['sum','count'])
        s = np.zeros(len(dates), dtype=np.float64)
        n = np.zeros(len(dates), dtype=np.float64)
        for d, row in agg.iterrows():
            i = date_pos[np.datetime64(d)]
            s[i] = float(row['sum']); n[i] = float(row['count'])
        stats[name] = (s,n)

    rng = np.random.default_rng(seed)
    draws: dict[str,np.ndarray] = {k: np.empty(reps, dtype=np.float64) for k in stats}
    chunk = 1000
    pos = 0
    while pos < reps:
        k = min(chunk, reps-pos)
        idx = rng.integers(0, len(dates), size=(k,len(dates)), endpoint=False)
        for name,(s,n) in stats.items():
            den = n[idx].sum(axis=1)
            num = s[idx].sum(axis=1)
            draws[name][pos:pos+k] = np.divide(num, den, out=np.full(k,np.nan), where=den>0)
        pos += k

    def dist(x: np.ndarray) -> dict:
        x = x[np.isfinite(x)]
        return {
            'mean': float(np.mean(x)),
            'median': float(np.median(x)),
            'ci95_low': float(np.quantile(x,0.025)),
            'ci95_high': float(np.quantile(x,0.975)),
        }

    out = {name: dist(x) for name,x in draws.items()}
    for a,b,label in [
        ('model_top4','obvious_top4','model_top4_minus_obvious_top4'),
        ('model_only','obvious_only','model_only_minus_obvious_only'),
    ]:
        delta = draws[a]-draws[b]
        finite = delta[np.isfinite(delta)]
        out[label] = {
            **dist(finite),
            'prob_delta_gt_0': float(np.mean(finite>0)),
            'prob_delta_ge_0': float(np.mean(finite>=0)),
        }
    return out


def _present_obvious(active: list[str]) -> list[str]:
    preferred = [
        'batter_hr_per_pa_season',
        'batter_hr_per_pa_career',
        'batter_barrel_rate_season',
        'batter_barrel_rate_career',
        'batter_xwoba_on_contact_season',
        'batter_xwoba_on_contact_career',
        'batter_avg_ev_season',
    ]
    cols = [c for c in preferred if c in active]
    # Delivered-core naming is checked dynamically, but do not silently broaden
    # this into recent/context features.  The proxy must remain long-horizon
    # and batter-only.
    if len(cols) < 5:
        fallback = [
            c for c in active
            if c.startswith('batter_')
            and (c.endswith('_season') or c.endswith('_career'))
            and any(tok in c for tok in ('hr_per_pa','barrel_rate','xwoba_on_contact','avg_ev'))
        ]
        cols = sorted(set(cols+fallback))
    if len(cols) < 5:
        raise RuntimeError(f'insufficient obvious-power proxy features: {cols}')
    return cols


def _group_features(active: list[str]) -> dict[str,list[str]]:
    groups = {
        'park': sorted([c for c in active if 'park_hr_factor' in c]),
        'recent_batter_form': sorted([
            c for c in active
            if c.startswith('batter_')
            and ('14d' in c or '30d' in c)
            and not c.startswith('batter_hr_per_pa_vs_')
            and any(tok in c for tok in (
                'hr_per_pa','barrel_rate','xwoba_on_contact','avg_ev','ev90',
                'hard_hit','fb_pct','sweet_spot','iso_xbp'
            ))
        ]),
        'pitcher_vulnerability': sorted([
            c for c in active
            if c.startswith('pitcher_')
            and not c.startswith('pitcher_usage_')
            and any(tok in c for tok in (
                'hr_per_pa','barrel_rate_allowed','xwoba_on_contact_allowed',
                'hard_hit_pct_allowed','avg_ev_allowed','iso_xbp_allowed'
            ))
        ]),
        'pitch_matchup': sorted([
            c for c in active
            if c.startswith('batter_hr_per_pa_vs_')
            or c == 'batter_strength_on_pitcher_top_pitch'
        ]),
    }
    return {k:v for k,v in groups.items() if v}


def _composite_percentile(frame: pd.DataFrame, cols: list[str], train_medians: pd.Series) -> pd.Series:
    pieces = []
    for c in cols:
        v = pd.to_numeric(frame[c], errors='coerce').fillna(float(train_medians[c]))
        pieces.append(v.groupby(frame.game_date).rank(method='average', pct=True))
    return pd.concat(pieces, axis=1).mean(axis=1)


def _scope(frame: pd.DataFrame, reps: int, seed: int) -> dict:
    masks = {
        'model_top4': frame.model_top4,
        'obvious_top4': frame.obvious_top4,
        'shared_top4': frame.model_top4 & frame.obvious_top4,
        'model_only': frame.model_top4 & ~frame.obvious_top4,
        'obvious_only': frame.obvious_top4 & ~frame.model_top4,
    }
    result = {'segments': {k:_rate(frame,m) for k,m in masks.items()}}
    model_only = frame.loc[masks['model_only']]
    result['model_only_obvious_rank'] = {
        'median': float(model_only.obvious_rank.median()) if len(model_only) else None,
        'p75': float(model_only.obvious_rank.quantile(.75)) if len(model_only) else None,
        'rank_5_8': _rate(frame, masks['model_only'] & frame.obvious_rank.between(5,8)),
        'rank_9_16': _rate(frame, masks['model_only'] & frame.obvious_rank.between(9,16)),
        'rank_17_plus': _rate(frame, masks['model_only'] & frame.obvious_rank.ge(17)),
    }
    result['bootstrap_10000_date_cluster'] = _bootstrap_segment_rates(frame,masks,reps,seed)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-parquet', required=True)
    ap.add_argument('--baseline-out', required=True)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    if args.reps < 1000:
        raise RuntimeError('edge-localization bootstrap requires >=1000 reps')

    active = json.loads(FLIST.read_text())
    obvious = _present_obvious(active)
    groups = _group_features(active)
    needed = list(dict.fromkeys(
        ['game_pk','batter_id','game_date','year','hr_in_game','lineup_slot']
        + obvious + [c for cols in groups.values() for c in cols]
    ))
    features = pd.read_parquet(FEAT, columns=needed)
    features['game_date'] = pd.to_datetime(features.game_date).dt.normalize()
    if not features.year.between(2015,2024).all():
        raise RuntimeError('edge-localization features escaped 2015-2024')

    train = features[features.year<=2022]
    medians = train[sorted(set(obvious+[c for g in groups.values() for c in g]))].median(numeric_only=True)

    pred = pd.read_parquet(args.predictions)
    pred['game_date'] = pd.to_datetime(pred.game_date).dt.normalize()
    if not pred.year.isin([2023,2024]).all():
        raise RuntimeError('edge-localization predictions escaped 2023-2024')
    if '2025' in ' '.join(map(str,pred.year.unique())):
        raise RuntimeError('2025 escaped into edge-localization')

    dev = features[features.year.isin([2023,2024])].merge(
        pred[['game_pk','batter_id','p_raw']], on=['game_pk','batter_id'], how='inner', validate='one_to_one'
    )
    if len(dev) != len(pred):
        raise RuntimeError(f'prediction/feature row mismatch: {len(pred)} vs {len(dev)}')

    dev['obvious_power_score'] = _composite_percentile(dev, obvious, medians)
    for name,cols in groups.items():
        dev[f'{name}_score'] = _composite_percentile(dev, cols, medians)

    dev['model_rank'] = dev.groupby('game_date').p_raw.rank(method='first', ascending=False)
    # deterministic baseline ties: sort then cumcount
    dev = dev.sort_values(['game_date','obvious_power_score','game_pk','batter_id'], ascending=[True,False,True,True]).copy()
    dev['obvious_rank'] = dev.groupby('game_date').cumcount()+1

    model_keys = set()
    obvious_keys = set()
    for _,g in dev.groupby('game_date',sort=True):
        model_keys |= _top_n(g,'p_raw',4)
        obvious_keys |= _top_n(g,'obvious_power_score',4)
    keys = list(zip(dev.game_pk.astype(int),dev.batter_id.astype(int)))
    dev['model_top4'] = [k in model_keys for k in keys]
    dev['obvious_top4'] = [k in obvious_keys for k in keys]
    dev['segment'] = np.select(
        [dev.model_top4 & dev.obvious_top4, dev.model_top4 & ~dev.obvious_top4, ~dev.model_top4 & dev.obvious_top4],
        ['shared_top4','model_only','obvious_only'],
        default='neither'
    )

    # Feature-group fingerprints: compare differentiated picks with shared picks
    # and the entire development slate. Scores are within-day percentiles.
    fingerprints = {}
    for name in groups:
        c=f'{name}_score'
        fingerprints[name] = {
            'all_mean': float(dev[c].mean()),
            'shared_top4_mean': float(dev.loc[dev.segment.eq('shared_top4'),c].mean()),
            'model_only_mean': float(dev.loc[dev.segment.eq('model_only'),c].mean()),
            'obvious_only_mean': float(dev.loc[dev.segment.eq('obvious_only'),c].mean()),
        }

    payload = {
        'design': {
            'champion': 'full73_aggressive',
            'comparison': 'long-horizon batter-only obvious-power proxy; NOT historical public consensus or market odds',
            'obvious_power_features': obvious,
            'context_feature_groups': groups,
            'selector': 'top4 within game_date',
            'ranking_score': 'raw XGBoost probability',
            'development_years': [2023,2024],
            'bootstrap_replicates': int(args.reps),
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'combined_2023_2024': _scope(dev,args.reps,args.seed),
        'by_year': {
            '2023': _scope(dev[dev.year.eq(2023)],args.reps,args.seed+10),
            '2024': _scope(dev[dev.year.eq(2024)],args.reps,args.seed+20),
        },
        'context_fingerprints': fingerprints,
        'overlap': {
            'model_top4_n': int(dev.model_top4.sum()),
            'shared_n': int((dev.model_top4 & dev.obvious_top4).sum()),
            'shared_fraction_of_model_top4': float((dev.model_top4 & dev.obvious_top4).sum()/dev.model_top4.sum()),
        },
    }

    outj=Path(args.out_json); outj.parent.mkdir(parents=True,exist_ok=True)
    outj.write_text(json.dumps(payload,indent=2))
    Path(args.out_parquet).parent.mkdir(parents=True,exist_ok=True)
    dev.to_parquet(args.out_parquet,index=False)

    baseline = pred[['game_pk','batter_id','game_date','year','hr_in_game']].merge(
        dev[['game_pk','batter_id','obvious_power_score']], on=['game_pk','batter_id'], how='one_to_one' if False else 'inner', validate='one_to_one'
    )
    if len(baseline)!=len(pred):
        raise RuntimeError('baseline export row mismatch')
    baseline['p_raw']=baseline.obvious_power_score.astype('float32')
    baseline['p_cal']=baseline.p_raw
    baseline['model']='obvious_power_proxy'
    baseline=baseline.drop(columns='obvious_power_score')
    Path(args.baseline_out).parent.mkdir(parents=True,exist_ok=True)
    baseline.to_parquet(args.baseline_out,index=False)

    print(json.dumps(payload,indent=2),flush=True)
    print('[edge-localization] 2025 NOT READ',flush=True)


if __name__ == '__main__':
    main()
