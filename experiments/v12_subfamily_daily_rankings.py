"""Daily-slate rankings for the fine-grained v1.2 subfamily ablations.

Ranks raw XGBoost scores independently per game date on the 2023-2024
DEVELOPMENT set. 2025 is never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
PRED = ROOT/'models/v1.2_subablations/development_predictions.parquet'
FEAT = ROOT/'features/v1.2_trusted/game_features.parquet'
OUT = ROOT/'models/v1.2_subablations'
FRACTIONS = (0.10, 0.05, 0.02, 0.01)
TOP_NS = (1, 2, 4, 8)


def order(g: pd.DataFrame) -> np.ndarray:
    return np.lexsort((
        g.batter_id.to_numpy(dtype=np.int64),
        g.game_pk.to_numpy(dtype=np.int64),
        -g.p_raw.to_numpy(dtype=np.float64),
    ))


def daily_select(frame: pd.DataFrame, frac=None, top_n=None) -> pd.DataFrame:
    if (frac is None) == (top_n is None):
        raise ValueError('set exactly one selector')
    parts = []
    for _, g in frame.groupby('game_date', sort=True):
        o = order(g)
        n = max(1, int(np.ceil(len(g)*frac))) if frac is not None else min(int(top_n), len(g))
        parts.append(g.iloc[o[:n]])
    return pd.concat(parts, ignore_index=True)


def stats(sel: pd.DataFrame) -> dict:
    obs = float(sel.hr_in_game.mean())
    pred = float(sel.p_cal.mean())
    return {
        'n': int(len(sel)),
        'n_dates': int(sel.game_date.nunique()),
        'observed_hr_rate': obs,
        'mean_calibrated_probability': pred,
        'calibration_gap_observed_minus_predicted': obs-pred,
    }


def keyset(df: pd.DataFrame):
    return set(zip(df.game_date.astype(str), df.game_pk.astype(int), df.batter_id.astype(int)))


def main() -> None:
    pred = pd.read_parquet(PRED)
    ctx = pd.read_parquet(FEAT, columns=['game_pk','batter_id','game_date']).drop_duplicates(['game_pk','batter_id'])
    pred = pred.merge(ctx, on=['game_pk','batter_id'], how='left', validate='many_to_one')
    pred['game_date'] = pd.to_datetime(pred.game_date).dt.normalize()
    if pred.game_date.isna().any() or not pred.year.between(2023, 2024).all():
        raise RuntimeError('invalid development ranking context')

    models = pred.model.drop_duplicates().tolist()
    if 'full_73' not in models:
        raise RuntimeError('missing full reference')

    results = {}
    sels = {}
    for model in models:
        f = pred[pred.model.eq(model)].copy()
        one = {}
        store = {}
        for frac in FRACTIONS:
            k = f'daily_top{int(frac*100)}pct'
            s = daily_select(f, frac=frac)
            one[k] = stats(s); store[k] = s
        for n in TOP_NS:
            k = f'daily_top{n}'
            s = daily_select(f, top_n=n)
            one[k] = stats(s); store[k] = s
        results[model] = {'overall': one}
        sels[model] = store

    for model in models:
        ov = {}
        if model != 'full_73':
            for k, a_df in sels['full_73'].items():
                a, b = keyset(a_df), keyset(sels[model][k])
                inter = len(a & b)
                ov[k] = {'retention_vs_full': inter/len(a), 'jaccard_vs_full': inter/len(a | b)}
        results[model]['overlap_vs_full'] = ov

    full = results['full_73']['overall']
    rows = []
    keys = [f'daily_top{int(f*100)}pct' for f in FRACTIONS] + [f'daily_top{n}' for n in TOP_NS]
    for model in models:
        row = {'model': model}
        for k in keys:
            h = results[model]['overall'][k]['observed_hr_rate']
            row[f'{k}_hr_rate'] = h
            row[f'{k}_delta_vs_full'] = h-full[k]['observed_hr_rate']
            row[f'{k}_mean_cal_probability'] = results[model]['overall'][k]['mean_calibrated_probability']
            if model != 'full_73':
                row[f'{k}_retention_vs_full'] = results[model]['overlap_vs_full'][k]['retention_vs_full']
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'daily_subablation_summary.csv', index=False)
    (OUT/'daily_subablation_results.json').write_text(json.dumps({
        'design': {
            'rank_within': 'game_date',
            'ranking_score': 'raw_xgboost_probability',
            'development_years': [2023,2024],
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'results': results,
    }, indent=2))

    idx = table.set_index('model')
    order_models = ['full_73'] + sorted(
        [m for m in models if m != 'full_73'],
        key=lambda m: idx.loc[m,'daily_top4_hr_rate'],
        reverse=True,
    )
    lines = [
        '# Fine-grained daily-slate feature ablations','',
        '**2025 was not read or evaluated.**','',
        '| Model | top5% | top2% | Top1/day | Top2/day | Top4/day | Top8/day | Top4 retention |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for model in order_models:
        r = idx.loc[model]
        ret = '-' if model == 'full_73' else f"{100*r['daily_top4_retention_vs_full']:.1f}%"
        lines.append(
            f"| {model} | {100*r['daily_top5pct_hr_rate']:.2f}% | {100*r['daily_top2pct_hr_rate']:.2f}% | "
            f"{100*r['daily_top1_hr_rate']:.2f}% | {100*r['daily_top2_hr_rate']:.2f}% | "
            f"{100*r['daily_top4_hr_rate']:.2f}% | {100*r['daily_top8_hr_rate']:.2f}% | {ret} |"
        )
    (OUT/'daily_subablation_report.md').write_text('\n'.join(lines)+'\n')
    print((OUT/'daily_subablation_report.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
