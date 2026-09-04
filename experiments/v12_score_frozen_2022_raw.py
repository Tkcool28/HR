"""Score the frozen full73 architecture on 2022 using only pre-2022 training.

This is a freshness/replication check for the already-defined disagreement rule,
not a pristine holdout. The XGBoost base model is trained on 2015-2021 only and
2022 is scored with raw probabilities. Isotonic calibration is deliberately not
fit or evaluated here because 2022 served as the calibration year in the main
architecture.

2025 is not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/workspace/hr_model')
FEAT = ROOT/'features/v1.2_trusted/game_features.parquet'


def matrix(df: pd.DataFrame, cols: list[str], means: np.ndarray | None = None):
    X = df[cols].to_numpy(dtype=np.float32, copy=True)
    if means is None:
        means = np.nanmean(X, axis=0).astype(np.float32)
    if np.isnan(means).any():
        raise RuntimeError('all-NaN feature mean in 2022 freshness scorer')
    rr, cc = np.where(np.isnan(X))
    if len(rr):
        X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError('non-finite 2022 score matrix')
    return X, means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature-list', required=True)
    ap.add_argument('--contract', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cols = json.loads(Path(args.feature_list).read_text())
    contract = json.loads(Path(args.contract).read_text())
    params = dict(contract['best_params'])
    rounds = int(contract['best_round'])
    if rounds <= 0:
        raise RuntimeError('best_round must be positive')

    needed = list(dict.fromkeys(cols + ['game_pk','batter_id','game_date','year','hr_in_game']))
    df = pd.read_parquet(FEAT, columns=needed)
    if not df.year.between(2015, 2024).all() or 2025 in set(df.year):
        raise RuntimeError('2022 freshness scorer escaped trusted 2015-2024 matrix')
    if df.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate batter-game rows')

    tr = df[df.year <= 2021].copy()
    score = df[df.year == 2022].copy()
    if min(len(tr), len(score)) <= 0:
        raise RuntimeError('empty 2015-2021 training or 2022 scoring partition')

    Xtr, means = matrix(tr, cols)
    Xscore, _ = matrix(score, cols, means)
    ytr = tr.hr_in_game.to_numpy(dtype=np.int8)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'seed': 42,
        'nthread': -1,
        **params,
    }
    bst = xgb.train(
        xgb_params,
        xgb.DMatrix(Xtr, label=ytr, feature_names=cols),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    p_raw = bst.predict(xgb.DMatrix(Xscore, feature_names=cols))

    out = score[['game_pk','batter_id','game_date','year','hr_in_game']].copy()
    out['game_date'] = pd.to_datetime(out.game_date).dt.normalize()
    out['p_raw'] = p_raw.astype(np.float32)
    out['model'] = 'full73_aggressive_pre2022_raw'
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    summary = {
        'model': 'full73_aggressive_pre2022_raw',
        'train_years': '2015-2021',
        'score_year': 2022,
        'n_features': len(cols),
        'best_round': rounds,
        'n_rows': int(len(out)),
        'calibration_used': False,
        'holdout_2025_read': False,
    }
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('[2022-freshness-score] 2025 NOT READ', flush=True)


if __name__ == '__main__':
    main()
