"""Re-score a frozen v1.2 feature contract for tail-bootstrap evidence.

This is not a tuner. It takes an already-frozen feature list, XGBoost
hyperparameters, and boosting-round count; refits the 2015-2021 evaluation
base model, fits isotonic calibration on 2022, and exports raw/calibrated
predictions for the 2023-2024 development set.

It exists so paired bootstrap comparisons can be reproduced cheaply without
rerunning Optuna.

2025 is not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path('/workspace/hr_model')
FEAT = ROOT/'features/v1.2_trusted/game_features.parquet'


def matrix(df: pd.DataFrame, cols: list[str], means: np.ndarray | None = None):
    X = df[cols].to_numpy(dtype=np.float32, copy=True)
    if means is None:
        means = np.nanmean(X, axis=0).astype(np.float32)
    if np.isnan(means).any():
        raise RuntimeError('all-NaN feature mean in frozen scorer')
    rr, cc = np.where(np.isnan(X))
    if len(rr):
        X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError('non-finite frozen score matrix')
    return X, means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature-list', required=True)
    ap.add_argument('--contract', required=True,
                    help='JSON with best_params and best_round')
    ap.add_argument('--label', required=True)
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
    if not df.year.between(2015, 2024).all():
        raise RuntimeError('frozen scorer escaped 2015-2024')
    if df.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate batter-game rows')

    tr = df[df.year <= 2021].copy()
    cal = df[df.year == 2022].copy()
    dev = df[df.year.isin([2023, 2024])].copy()
    if min(len(tr), len(cal), len(dev)) <= 0:
        raise RuntimeError('empty chronological partition')

    Xtr, means = matrix(tr, cols)
    Xcal, _ = matrix(cal, cols, means)
    Xdev, _ = matrix(dev, cols, means)
    ytr = tr.hr_in_game.to_numpy(dtype=np.int8)
    ycal = cal.hr_in_game.to_numpy(dtype=np.int8)
    ydev = dev.hr_in_game.to_numpy(dtype=np.int8)

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
    p_cal_raw = bst.predict(xgb.DMatrix(Xcal, feature_names=cols))
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p_cal_raw, ycal)
    p_raw = bst.predict(xgb.DMatrix(Xdev, feature_names=cols))
    p_cal = iso.predict(p_raw)

    out = dev[['game_pk','batter_id','game_date','year','hr_in_game']].copy()
    out['game_date'] = pd.to_datetime(out.game_date).dt.normalize()
    out['p_raw'] = p_raw.astype(np.float32)
    out['p_cal'] = np.asarray(p_cal, dtype=np.float32)
    out['model'] = args.label
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    metrics = {
        'model': args.label,
        'n_features': len(cols),
        'best_round': rounds,
        'raw_brier': float(brier_score_loss(ydev, p_raw)),
        'raw_auc': float(roc_auc_score(ydev, p_raw)),
        'cal_brier': float(brier_score_loss(ydev, p_cal)),
        'cal_auc': float(roc_auc_score(ydev, p_cal)),
        'holdout_2025_read': False,
    }
    path.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'[frozen-tail-score] wrote {len(out):,} rows: {path}')
    print('[frozen-tail-score] 2025 NOT READ')


if __name__ == '__main__':
    main()
