"""Build the frozen obvious-power disagreement surface for 2022.

This reproduces the already-defined long-horizon batter-only obvious-power
proxy and full73 raw-score daily top-four ranking, but on 2022 only. Training
medians for proxy preprocessing use 2015-2021. The rank bands/cutoffs are not
changed. This is a freshness check, not a pristine final holdout.

2025 is not read.
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


def present_obvious(active: list[str]) -> list[str]:
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
    if len(cols) < 5:
        fallback = [
            c for c in active
            if c.startswith('batter_')
            and (c.endswith('_season') or c.endswith('_career'))
            and any(tok in c for tok in ('hr_per_pa','barrel_rate','xwoba_on_contact','avg_ev'))
        ]
        cols = sorted(set(cols + fallback))
    if len(cols) < 5:
        raise RuntimeError(f'insufficient obvious-power proxy features: {cols}')
    return cols


def composite_percentile(frame: pd.DataFrame, cols: list[str], train_medians: pd.Series) -> pd.Series:
    pieces = []
    for c in cols:
        v = pd.to_numeric(frame[c], errors='coerce').fillna(float(train_medians[c]))
        pieces.append(v.groupby(frame.game_date).rank(method='average', pct=True))
    return pd.concat(pieces, axis=1).mean(axis=1)


def assign_daily_rank(frame: pd.DataFrame, score: str) -> pd.Series:
    ranks = pd.Series(index=frame.index, dtype='int32')
    for _, g in frame.groupby('game_date', sort=True):
        ordered = g.sort_values([score, 'game_pk', 'batter_id'], ascending=[False, True, True])
        ranks.loc[ordered.index] = np.arange(1, len(ordered)+1, dtype=np.int32)
    return ranks.astype('int32')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    active = json.loads(FLIST.read_text())
    obvious = present_obvious(active)
    needed = list(dict.fromkeys(
        ['game_pk','batter_id','game_date','year','hr_in_game'] + obvious
    ))
    features = pd.read_parquet(FEAT, columns=needed)
    features['game_date'] = pd.to_datetime(features.game_date).dt.normalize()
    if not features.year.between(2015, 2024).all() or 2025 in set(features.year):
        raise RuntimeError('2022 edge surface escaped trusted 2015-2024 matrix')

    medians = features.loc[features.year <= 2021, obvious].median(numeric_only=True)
    if medians.isna().any():
        raise RuntimeError(f'all-NaN pre-2022 proxy medians: {medians[medians.isna()].index.tolist()}')

    pred = pd.read_parquet(args.predictions)
    pred['game_date'] = pd.to_datetime(pred.game_date).dt.normalize()
    if set(map(int, pred.year.unique())) != {2022}:
        raise RuntimeError(f'2022 predictions contain wrong years: {sorted(pred.year.unique())}')

    score = features[features.year.eq(2022)].merge(
        pred[['game_pk','batter_id','p_raw']],
        on=['game_pk','batter_id'], how='inner', validate='one_to_one'
    )
    if len(score) != len(pred):
        raise RuntimeError(f'2022 prediction/feature mismatch: {len(pred)} vs {len(score)}')

    score['obvious_power_score'] = composite_percentile(score, obvious, medians)
    score['model_rank'] = assign_daily_rank(score, 'p_raw')
    score['obvious_rank'] = assign_daily_rank(score, 'obvious_power_score')
    score['model_top4'] = score.model_rank.le(4)
    score['obvious_top4'] = score.obvious_rank.le(4)

    out = score[[
        'game_pk','batter_id','game_date','year','hr_in_game','p_raw',
        'obvious_power_score','model_rank','obvious_rank','model_top4','obvious_top4'
    ]].copy()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    summary = {
        'year': 2022,
        'n_rows': int(len(out)),
        'n_slate_dates': int(out.game_date.nunique()),
        'obvious_power_features': obvious,
        'proxy_preprocessing_years': '2015-2021',
        'model_ranking': 'raw XGBoost; base train 2015-2021',
        'rank_bands_frozen_elsewhere': ['5-8','9-16','17+'],
        'holdout_2025_read': False,
    }
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('[2022-edge-surface] 2025 NOT READ', flush=True)


if __name__ == '__main__':
    main()
