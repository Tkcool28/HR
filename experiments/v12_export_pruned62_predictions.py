"""Export 2023-2024 development predictions for the aggressively tuned pruned62 challenger.

The prediction table is the frozen evidence surface for later paired bootstrap
analysis. Selection is always by raw XGBoost score within game date; isotonic
probability is retained only for calibration diagnostics.

2025 is not read.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path('/workspace/hr_model')
FEAT = ROOT/'features/v1.2_trusted/game_features.parquet'
MODEL_DIR = ROOT/'models/v1.2_pruned62'
OUT = MODEL_DIR/'development_predictions_2023_2024.parquet'
SUMMARY = MODEL_DIR/'daily_ranking_summary.json'


def deterministic_order(g: pd.DataFrame) -> np.ndarray:
    return np.lexsort((
        g.batter_id.to_numpy(dtype=np.int64),
        g.game_pk.to_numpy(dtype=np.int64),
        -g.p_raw.to_numpy(dtype=np.float64),
    ))


def select_daily(frame: pd.DataFrame, *, frac: float | None = None, top_n: int | None = None) -> pd.DataFrame:
    if (frac is None) == (top_n is None):
        raise ValueError('set exactly one of frac/top_n')
    parts = []
    for _, g in frame.groupby('game_date', sort=True):
        order = deterministic_order(g)
        n = max(1, int(np.ceil(len(g) * frac))) if frac is not None else min(int(top_n), len(g))
        parts.append(g.iloc[order[:n]])
    return pd.concat(parts, ignore_index=True)


def metrics(sel: pd.DataFrame) -> dict:
    return {
        'n': int(len(sel)),
        'n_dates': int(sel.game_date.nunique()),
        'observed_hr_rate': float(sel.hr_in_game.mean()),
        'mean_raw_probability': float(sel.p_raw.mean()),
        'mean_calibrated_probability': float(sel.p_cal.mean()),
    }


def main() -> None:
    eval_bundle = joblib.load(MODEL_DIR/'xgb_evaluation.joblib')
    iso_bundle = joblib.load(MODEL_DIR/'isotonic_evaluation.joblib')
    feature_cols = list(eval_bundle['feature_cols'])
    if len(feature_cols) != 62:
        raise RuntimeError(f'expected 62 features, got {len(feature_cols)}')

    id_cols = ['game_pk','game_date','year','batter_id','hr_in_game']
    df = pd.read_parquet(FEAT, columns=list(dict.fromkeys(id_cols + feature_cols)))
    df = df[df.year.isin([2023, 2024])].copy()
    if df.empty or not df.year.between(2023, 2024).all():
        raise RuntimeError('invalid 2023-2024 development slice')
    if df.duplicated(['game_pk','batter_id']).any():
        raise RuntimeError('duplicate batter-game rows')

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    means = np.asarray(eval_bundle['col_means'], dtype=np.float32)
    rr, cc = np.where(np.isnan(X))
    if len(rr):
        X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError('non-finite model matrix after frozen imputation')

    p_raw = eval_bundle['model'].predict(xgb.DMatrix(X, feature_names=feature_cols))
    p_cal = iso_bundle['model'].predict(p_raw)
    out = df[id_cols].copy()
    out['game_date'] = pd.to_datetime(out.game_date).dt.normalize()
    out['p_raw'] = p_raw.astype(np.float32)
    out['p_cal'] = np.asarray(p_cal, dtype=np.float32)
    out.to_parquet(OUT, index=False)

    overall = {
        'daily_top5pct': metrics(select_daily(out, frac=0.05)),
        'daily_top4': metrics(select_daily(out, top_n=4)),
        'daily_top2': metrics(select_daily(out, top_n=2)),
        'daily_top1': metrics(select_daily(out, top_n=1)),
    }
    by_year = {}
    for year in (2023, 2024):
        fy = out[out.year.eq(year)]
        by_year[str(year)] = {
            'daily_top5pct': metrics(select_daily(fy, frac=0.05)),
            'daily_top4': metrics(select_daily(fy, top_n=4)),
        }

    payload = {
        'design': {
            'model': 'pruned62_aggressively_tuned',
            'feature_count': 62,
            'rank_within': 'game_date',
            'ranking_score': 'raw_xgboost_probability',
            'calibration_used_for_selection': False,
            'development_years': [2023, 2024],
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'overall': overall,
        'by_year': by_year,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print(f'[pruned62-export] wrote {len(out):,} development predictions to {OUT}')
    print('[pruned62-export] 2025 NOT READ')


if __name__ == '__main__':
    main()
