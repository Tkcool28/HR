"""Post-holdout 2019-2025 full-picture audit for frozen v1.2 Full73.

This is retrospective reporting only. It does not retune or change the model.
It also repairs one comparator implementation defect discovered after the sealed
2025 run: the freeze contract referenced the existing obvious-power definition,
which contains seven long-horizon batter-only features. The as-run 2025
evaluator accidentally used only five. This script preserves the as-run result,
recomputes only the comparator under the pre-existing seven-feature definition,
and then builds the requested year-by-year perspective table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260904
TOP_FRAC = 0.05
OBVIOUS7 = [
    'batter_hr_per_pa_season',
    'batter_hr_per_pa_career',
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


def ordered(g: pd.DataFrame, score: str) -> pd.DataFrame:
    return g.sort_values([score, 'game_pk', 'batter_id'], ascending=[False, True, True])


def add_rank(d: pd.DataFrame, score: str) -> pd.Series:
    out = pd.Series(index=d.index, dtype='int32')
    for _, g in d.groupby('game_date', sort=True):
        z = ordered(g, score)
        out.loc[z.index] = np.arange(1, len(z) + 1, dtype=np.int32)
    return out.astype('int32')


def top5_mask(d: pd.DataFrame, rank_col: str) -> pd.Series:
    sizes = d.groupby('game_date').size()
    cut = d.game_date.map(lambda x: max(1, math.ceil(TOP_FRAC * int(sizes.loc[x]))))
    return d[rank_col].le(cut)


def obvious_score(d: pd.DataFrame, medians: pd.Series) -> pd.Series:
    parts = []
    for c in OBVIOUS7:
        v = pd.to_numeric(d[c], errors='coerce').fillna(float(medians[c]))
        parts.append(v.groupby(d.game_date).rank(method='average', pct=True))
    return pd.concat(parts, axis=1).mean(axis=1)


def rate(d: pd.DataFrame, m: pd.Series) -> dict:
    x = d.loc[m]
    return {
        'n': int(len(x)),
        'hr': int(x.hr_in_game.sum()),
        'hr_rate': float(x.hr_in_game.mean()) if len(x) else None,
    }


def paired_bootstrap(d: pd.DataFrame, ref: pd.Series, model: pd.Series, reps: int = 10000) -> dict:
    dates = np.array(sorted(d.game_date.drop_duplicates().to_numpy()))
    pos = {x: i for i, x in enumerate(dates)}

    def agg(mask: pd.Series):
        a = d.loc[mask, ['game_date', 'hr_in_game']].groupby('game_date').hr_in_game.agg(['sum', 'count'])
        s = np.zeros(len(dates)); n = np.zeros(len(dates))
        for day, row in a.iterrows():
            key = np.datetime64(pd.Timestamp(day).to_datetime64())
            i = pos[key]; s[i] = row['sum']; n[i] = row['count']
        return s, n

    sr, nr = agg(ref); sm, nm = agg(model)
    rng = np.random.default_rng(SEED)
    delta = np.empty(reps)
    at = 0
    while at < reps:
        k = min(1000, reps - at)
        ix = rng.integers(0, len(dates), size=(k, len(dates)))
        rr = sr[ix].sum(1) / nr[ix].sum(1)
        rm = sm[ix].sum(1) / nm[ix].sum(1)
        delta[at:at+k] = rm - rr
        at += k
    return {
        'delta_mean_pp': float(delta.mean() * 100),
        'delta_median_pp': float(np.median(delta) * 100),
        'ci95_low_pp': float(np.quantile(delta, .025) * 100),
        'ci95_high_pp': float(np.quantile(delta, .975) * 100),
        'prob_delta_gt_0': float((delta > 0).mean()),
        'replicates': reps,
        'cluster': 'slate_date',
    }


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return ctr-half, ctr+half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--historical-scored', required=True)
    ap.add_argument('--historical-json', required=True)
    ap.add_argument('--historical-features', required=True)
    ap.add_argument('--holdout-features', required=True)
    ap.add_argument('--holdout-predictions', required=True)
    ap.add_argument('--holdout-result', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-year-csv', required=True)
    ap.add_argument('--out-rank-csv', required=True)
    ap.add_argument('--out-correction-json', required=True)
    args = ap.parse_args()

    hist = pd.read_parquet(args.historical_scored)
    hist['game_date'] = pd.to_datetime(hist.game_date).dt.normalize()
    if set(hist.year.astype(int)) != set(range(2019, 2025)):
        raise RuntimeError('historical scored scope must be exactly 2019-2024')
    need_hist = {'game_pk','batter_id','game_date','year','hr_in_game','p_raw','model_rank','obvious_rank','full73_top5','obvious_top5'}
    if not need_hist.issubset(hist.columns):
        raise RuntimeError(f'historical scored columns missing: {sorted(need_hist-set(hist.columns))}')

    hm = json.loads(Path(args.historical_json).read_text())
    hfeat = pd.read_parquet(args.historical_features, columns=['year'] + OBVIOUS7)
    if not hfeat.year.between(2015, 2024).all() or 2025 in set(hfeat.year.astype(int)):
        raise RuntimeError('historical feature scope violation')
    med = hfeat[OBVIOUS7].median(numeric_only=True)
    if med.isna().any():
        raise RuntimeError(f'2025 obvious7 all-NaN medians: {med[med.isna()].index.tolist()}')

    hf = pd.read_parquet(args.holdout_features)
    hp = pd.read_parquet(args.holdout_predictions)
    hf['game_date'] = pd.to_datetime(hf.game_date).dt.normalize()
    hp['game_date'] = pd.to_datetime(hp.game_date).dt.normalize()
    if set(hf.year.astype(int)) != {2025} or set(hp.year.astype(int)) != {2025}:
        raise RuntimeError('holdout scope must be exactly 2025')
    if hf.duplicated(['game_pk','batter_id']).any() or hp.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate holdout batter-game')
    missing = [c for c in OBVIOUS7 if c not in hf.columns]
    if missing:
        raise RuntimeError(f'holdout feature matrix missing obvious7: {missing}')

    pcols = ['game_pk','batter_id','p_raw','model_rank','hr_in_game']
    h25 = hf[['game_pk','batter_id','game_date','year','hr_in_game'] + OBVIOUS7].merge(
        hp[pcols], on=['game_pk','batter_id'], suffixes=('', '_saved'), validate='one_to_one'
    )
    if len(h25) != len(hp):
        raise RuntimeError('2025 holdout feature/prediction row mismatch')
    if not h25.hr_in_game.eq(h25.hr_in_game_saved).all():
        raise RuntimeError('2025 outcome mismatch between frozen artifacts')
    h25['model_rank_recomputed'] = add_rank(h25, 'p_raw')
    if not h25.model_rank_recomputed.eq(h25.model_rank).all():
        raise RuntimeError('frozen 2025 model rank changed during comparator correction')
    h25['obvious_power_score'] = obvious_score(h25, med)
    h25['obvious_rank'] = add_rank(h25, 'obvious_power_score')
    h25['full73_top5'] = top5_mask(h25, 'model_rank')
    h25['obvious_top5'] = top5_mask(h25, 'obvious_rank')

    holdout = json.loads(Path(args.holdout_result).read_text())
    asrun = holdout['primary']
    m5 = h25.full73_top5
    o5 = h25.obvious_top5
    corrected = {
        'defect': '2025 as-run evaluator used 5 obvious-power features; freeze contract referenced pre-existing 7-feature definition',
        'correction_scope': 'comparator only; Full73 predictions/ranks/outcomes are unchanged',
        'frozen_existing_obvious_features': OBVIOUS7,
        'historical_median_scope_for_2025': '2015-2024 only',
        'as_run_5_feature_comparator': {
            'obvious_top5_hr_rate': asrun['obvious_top5']['hr_rate'],
            'full73_minus_obvious_pp': asrun['observed_full73_minus_obvious_pp'],
            'bootstrap': asrun['paired_bootstrap'],
        },
        'corrected_7_feature_comparator': {
            'full73_top5': rate(h25, m5),
            'obvious_top5': rate(h25, o5),
            'full73_minus_obvious_pp': float((h25.loc[m5,'hr_in_game'].mean()-h25.loc[o5,'hr_in_game'].mean())*100),
            'paired_bootstrap': paired_bootstrap(h25, o5, m5),
            'overlap_n': int((m5&o5).sum()),
            'overlap_fraction': float((m5&o5).sum()/m5.sum()),
        },
    }
    Path(args.out_correction_json).write_text(json.dumps(corrected, indent=2))

    # Normalize 2025 columns to historical scored surface and append.
    h25['obvious_rank_percentile'] = (h25.obvious_rank - 1) / np.maximum(h25.groupby('game_date').game_pk.transform('size') - 1, 1)
    h25['full73_only'] = h25.full73_top5 & ~h25.obvious_top5
    h25['shared_top5'] = h25.full73_top5 & h25.obvious_top5
    hist['full73_only'] = hist.full73_top5 & ~hist.obvious_top5
    hist['shared_top5'] = hist.full73_top5 & hist.obvious_top5
    combined = pd.concat([
        hist[['game_pk','batter_id','game_date','year','hr_in_game','p_raw','model_rank','obvious_rank','obvious_rank_percentile','full73_top5','obvious_top5','full73_only','shared_top5']],
        h25[['game_pk','batter_id','game_date','year','hr_in_game','p_raw','model_rank','obvious_rank','obvious_rank_percentile','full73_top5','obvious_top5','full73_only','shared_top5']],
    ], ignore_index=True)

    rows = []
    for y in range(2019, 2026):
        g = combined[combined.year.eq(y)].copy()
        mt = g.full73_top5; ot = g.obvious_top5; mo = g.full73_only
        if y <= 2024:
            mm = hm['walk_forward_fit_metrics'][str(y)]
            auc = float(mm['raw_auc']); brier = float(mm['raw_brier'])
            fit = f"2015-{y-1} expanding walk-forward fit; fixed Full73 params/194 rounds"
        else:
            mm = holdout['secondary_model_metrics']
            auc = float(mm['raw_auc']); brier = float(mm['raw_brier'])
            fit = '2015-2023 frozen base fit; 2024 calibration reserved; raw score ranks 2025'
        row = {
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
            'obvious7_top5_hr_rate': float(g.loc[ot,'hr_in_game'].mean()),
            'full73_minus_obvious7_pp': float((g.loc[mt,'hr_in_game'].mean()-g.loc[ot,'hr_in_game'].mean())*100),
            'top1_per_day_hr_rate': float(g.loc[g.model_rank.le(1),'hr_in_game'].mean()),
            'top2_per_day_hr_rate': float(g.loc[g.model_rank.le(2),'hr_in_game'].mean()),
            'top4_per_day_hr_rate': float(g.loc[g.model_rank.le(4),'hr_in_game'].mean()),
            'top8_per_day_hr_rate': float(g.loc[g.model_rank.le(8),'hr_in_game'].mean()),
            'top5_overlap_fraction': float((mt&ot).sum()/mt.sum()),
            'full73_only_n': int(mo.sum()),
            'full73_only_hr_rate': float(g.loc[mo,'hr_in_game'].mean()) if mo.any() else None,
            'full73_only_obvious_depth_mean': float(g.loc[mo,'obvious_rank_percentile'].mean()) if mo.any() else None,
        }
        rows.append(row)
    year_df = pd.DataFrame(rows)
    year_df.to_csv(args.out_year_csv, index=False)

    rank_rows = []
    for y in list(range(2019, 2026)) + ['POOLED_2019_2025']:
        g = combined if isinstance(y, str) else combined[combined.year.eq(y)]
        for r in range(1, 9):
            x = g[g.model_rank.eq(r)]
            k = int(x.hr_in_game.sum()); n = int(len(x)); lo, hi = wilson(k, n)
            rank_rows.append({
                'scope': str(y), 'exact_rank': r, 'n': n, 'hr': k,
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
            'historical_semantics': 'expanding walk-forward; fixed Full73 params/194 rounds; retrospective, not equally independent',
            'holdout_semantics': 'sealed one-shot frozen fit 2015-2023; 2024 calibration',
            'obvious_proxy': OBVIOUS7,
            'model_ranking': 'raw XGBoost score',
            'no_reoptimization': True,
        },
        'comparator_correction': corrected,
        'yearly': rows,
        'pooled_exact_rank_monotonicity': {
            'spearman_rank_vs_hr_rate': rank_spearman,
            'adjacent_inversions_among_ranks_1_to_8': adjacent_inversions,
            'exact_rank_rows': pooled.to_dict(orient='records'),
        },
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
