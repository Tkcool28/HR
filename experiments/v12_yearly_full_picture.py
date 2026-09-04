"""Post-holdout 2019-2025 full-picture audit for frozen v1.2 Full73.

Retrospective reporting only. No model fitting choices are changed here.
Historical 2019-2024 rows come from the already-built expanding walk-forward
scored artifact. The 2025 row comes from the immutable sealed one-shot artifact.

Comparator consistency note:
The historical proxy implementation contains two preferred names
(`batter_hr_per_pa_season/career`) that are not present in the delivered active
matrix. Its actual runtime proxy therefore consisted of the five present
long-horizon QoC features below. The sealed 2025 evaluator used those exact same
five features. This script asserts the artifact-recorded historical definition
rather than inferring it from nominal source-code preferences.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OBVIOUS5 = [
    'batter_barrel_rate_season',
    'batter_barrel_rate_career',
    'batter_xwoba_on_contact_season',
    'batter_xwoba_on_contact_career',
    'batter_avg_ev_season',
]
ROLES = {
    2019: 'walk-forward tuning fold; outcome participated in hyperparameter selection',
    2020: 'walk-forward tuning fold; shortened season; outcome participated in hyperparameter selection',
    2021: 'walk-forward tuning fold; outcome participated in hyperparameter selection',
    2022: 'walk-forward calibration/freshness year; not pristine',
    2023: 'walk-forward retrospective view of development-assessment season',
    2024: 'walk-forward retrospective view of development-assessment season; later used to calibrate 2025',
    2025: 'sealed one-shot holdout; frozen fit 2015-2023, calibration 2024',
}


def top5_mask(d: pd.DataFrame, rank_col: str) -> pd.Series:
    sizes = d.groupby('game_date').size()
    cut = d.game_date.map(lambda x: max(1, math.ceil(.05 * int(sizes.loc[x]))))
    return d[rank_col].le(cut)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return ctr-half, ctr+half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--historical-scored', required=True)
    ap.add_argument('--historical-json', required=True)
    ap.add_argument('--holdout-predictions', required=True)
    ap.add_argument('--holdout-result', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-year-csv', required=True)
    ap.add_argument('--out-rank-csv', required=True)
    ap.add_argument('--out-consistency-json', required=True)
    args = ap.parse_args()

    hist = pd.read_parquet(args.historical_scored)
    hist['game_date'] = pd.to_datetime(hist.game_date).dt.normalize()
    if set(hist.year.astype(int)) != set(range(2019, 2025)):
        raise RuntimeError('historical scored scope must be exactly 2019-2024')
    needed = {
        'game_pk','batter_id','game_date','year','hr_in_game','p_raw','model_rank',
        'obvious_rank','obvious_rank_percentile','full73_top5','obvious_top5',
    }
    if not needed.issubset(hist.columns):
        raise RuntimeError(f'historical scored columns missing: {sorted(needed-set(hist.columns))}')
    if hist.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate historical batter-game')

    hm = json.loads(Path(args.historical_json).read_text())
    actual_hist_proxy = hm['design']['obvious_proxy']
    if actual_hist_proxy != OBVIOUS5:
        raise RuntimeError(f'historical artifact proxy differs from frozen five: {actual_hist_proxy}')

    hp = pd.read_parquet(args.holdout_predictions)
    hp['game_date'] = pd.to_datetime(hp.game_date).dt.normalize()
    if set(hp.year.astype(int)) != {2025}:
        raise RuntimeError('holdout prediction scope must be exactly 2025')
    if hp.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate 2025 batter-game')
    for c in ['hr_in_game','p_raw','model_rank','obvious_rank']:
        if c not in hp.columns:
            raise RuntimeError(f'2025 frozen predictions missing {c}')

    holdout = json.loads(Path(args.holdout_result).read_text())
    primary = holdout['primary']
    hp['full73_top5'] = top5_mask(hp, 'model_rank')
    hp['obvious_top5'] = top5_mask(hp, 'obvious_rank')
    slate_size = hp.groupby('game_date').game_pk.transform('size').astype(int)
    hp['obvious_rank_percentile'] = (hp.obvious_rank.astype(float)-1) / np.maximum(slate_size-1, 1)

    # Prove that the artifact-saved 2025 comparator reproduces the locked as-run result.
    m5 = hp.full73_top5
    o5 = hp.obvious_top5
    mrate = float(hp.loc[m5,'hr_in_game'].mean())
    orate = float(hp.loc[o5,'hr_in_game'].mean())
    delta = (mrate-orate)*100
    if abs(mrate-primary['full73_top5']['hr_rate']) > 1e-12:
        raise RuntimeError('2025 Full73 top5 does not reproduce frozen primary')
    if abs(orate-primary['obvious_top5']['hr_rate']) > 1e-12:
        raise RuntimeError('2025 obvious top5 does not reproduce frozen primary')
    if abs(delta-primary['observed_full73_minus_obvious_pp']) > 1e-12:
        raise RuntimeError('2025 comparator delta does not reproduce frozen primary')

    consistency = {
        'verdict': 'CONSISTENT_NO_COMPARATOR_CORRECTION_REQUIRED',
        'historical_artifact_proxy': actual_hist_proxy,
        'holdout_evaluator_proxy': OBVIOUS5,
        'same_definition': True,
        'why_nominal_source_looked_like_seven': (
            'historical helper preferred batter_hr_per_pa_season/career, but those names are absent '
            'from the delivered active matrix; five existing preferred QoC fields met the helper minimum, '
            'so runtime output was exactly the five features recorded in the historical artifact'
        ),
        '2025_reproduced_as_run': {
            'full73_top5_hr_rate': mrate,
            'obvious_top5_hr_rate': orate,
            'full73_minus_obvious_pp': delta,
        },
        'model_results_changed': False,
        'comparator_results_changed': False,
    }
    Path(args.out_consistency_json).write_text(json.dumps(consistency, indent=2))

    # Append the sealed holdout row to the historical walk-forward scored surface.
    h25 = hp[['game_pk','batter_id','game_date','year','hr_in_game','p_raw','model_rank','obvious_rank','obvious_rank_percentile','full73_top5','obvious_top5']].copy()
    combined = pd.concat([
        hist[['game_pk','batter_id','game_date','year','hr_in_game','p_raw','model_rank','obvious_rank','obvious_rank_percentile','full73_top5','obvious_top5']],
        h25,
    ], ignore_index=True)
    combined['full73_only'] = combined.full73_top5 & ~combined.obvious_top5

    rows = []
    for y in range(2019, 2026):
        g = combined[combined.year.eq(y)].copy()
        mt = g.full73_top5
        ot = g.obvious_top5
        mo = g.full73_only
        if y <= 2024:
            mm = hm['walk_forward_fit_metrics'][str(y)]
            auc = float(mm['raw_auc'])
            brier = float(mm['raw_brier'])
            fit = f'2015-{y-1} expanding walk-forward fit; frozen Full73 params/194 rounds'
        else:
            mm = holdout['secondary_model_metrics']
            auc = float(mm['raw_auc'])
            brier = float(mm['raw_brier'])
            fit = '2015-2023 frozen base fit; 2024 calibration; raw score ranks 2025'
        rows.append({
            'year': y,
            'role': ROLES[y],
            'fit_semantics': fit,
            'slate_dates': int(g.game_date.nunique()),
            'target_rows': int(len(g)),
            'base_hr_rate': float(g.hr_in_game.mean()),
            'raw_auc': auc,
            'raw_brier': brier,
            'full73_top5_n': int(mt.sum()),
            'full73_top5_hr_rate': float(g.loc[mt,'hr_in_game'].mean()),
            'obvious5_top5_hr_rate': float(g.loc[ot,'hr_in_game'].mean()),
            'full73_minus_obvious5_pp': float((g.loc[mt,'hr_in_game'].mean()-g.loc[ot,'hr_in_game'].mean())*100),
            'top1_per_day_hr_rate': float(g.loc[g.model_rank.le(1),'hr_in_game'].mean()),
            'top2_per_day_hr_rate': float(g.loc[g.model_rank.le(2),'hr_in_game'].mean()),
            'top4_per_day_hr_rate': float(g.loc[g.model_rank.le(4),'hr_in_game'].mean()),
            'top8_per_day_hr_rate': float(g.loc[g.model_rank.le(8),'hr_in_game'].mean()),
            'top5_overlap_fraction': float((mt&ot).sum()/mt.sum()),
            'full73_only_n': int(mo.sum()),
            'full73_only_hr_rate': float(g.loc[mo,'hr_in_game'].mean()) if mo.any() else None,
            'full73_only_obvious_depth_mean': float(g.loc[mo,'obvious_rank_percentile'].mean()) if mo.any() else None,
        })
    year_df = pd.DataFrame(rows)
    year_df.to_csv(args.out_year_csv, index=False)

    rank_rows = []
    scopes = [str(y) for y in range(2019,2026)] + ['POOLED_2019_2025']
    for scope in scopes:
        g = combined if scope == 'POOLED_2019_2025' else combined[combined.year.eq(int(scope))]
        for r in range(1,9):
            x = g[g.model_rank.eq(r)]
            k = int(x.hr_in_game.sum()); n = int(len(x)); lo, hi = wilson(k,n)
            rank_rows.append({
                'scope': scope, 'exact_rank': r, 'n': n, 'hr': k,
                'hr_rate': float(k/n) if n else None,
                'wilson95_low': lo, 'wilson95_high': hi,
            })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(args.out_rank_csv, index=False)

    pooled = rank_df[rank_df.scope.eq('POOLED_2019_2025')].sort_values('exact_rank')
    pooled_rates = pooled.hr_rate.to_numpy(float)
    rank_spearman = float(pd.Series(range(1,9)).corr(pd.Series(pooled_rates), method='spearman'))
    adjacent_inversions = int(np.sum(np.diff(pooled_rates) > 0))

    result = {
        'design': {
            'historical_years': [2019,2020,2021,2022,2023,2024],
            'holdout_year': 2025,
            'historical_semantics': 'expanding walk-forward; frozen Full73 params/194 rounds; retrospective and not equally independent',
            'holdout_semantics': 'sealed one-shot; frozen fit 2015-2023; calibration 2024',
            'obvious_proxy': OBVIOUS5,
            'model_ranking': 'raw XGBoost score',
            'no_reoptimization': True,
        },
        'comparator_consistency': consistency,
        'yearly': rows,
        'pooled_exact_rank_monotonicity': {
            'spearman_exact_rank_vs_hr_rate': rank_spearman,
            'adjacent_inversions_among_ranks_1_to_8': adjacent_inversions,
            'exact_rank_rows': pooled.to_dict(orient='records'),
        },
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
