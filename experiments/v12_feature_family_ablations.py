"""Controlled feature-family ablations for the trusted v1.2 HR model.

This is a development experiment on 2015-2024 only. 2025 is never read.

The aggressive 73-feature model is the frozen reference. Every ablation:
- removes exactly one declared feature family;
- keeps the aggressive Optuna hyperparameters and 194-round budget frozen;
- trains the base model on 2015-2021;
- fits isotonic calibration on 2022 only;
- evaluates on the already-developmental 2023-2024 assessment set.

Ranking diagnostics use RAW XGBoost scores, not isotonic probabilities, so
isotonic tie blocks cannot change who enters a top bucket. Calibration is
reported separately on the raw-score-selected buckets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path('/workspace/hr_model')
FEAT_DIR = ROOT/'features/v1.2_trusted'
FEAT = FEAT_DIR/'game_features.parquet'
FLIST = FEAT_DIR/'feature_list.json'
SUMMARY = FEAT_DIR/'_summary.json'
AGGRESSIVE = ROOT/'models/v1.2_trusted/metrics.json'
OUT = ROOT/'models/v1.2_ablations'
SEED = 42
FRACTIONS = (0.10, 0.05, 0.02, 0.01)


def xgb_params(best: dict) -> dict:
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'seed': SEED,
        'nthread': -1,
        **best,
    }


def to_matrix(df: pd.DataFrame, cols: list[str], means: np.ndarray | None = None):
    X = df[cols].to_numpy(dtype=np.float32, copy=True)
    y = df.hr_in_game.to_numpy(dtype=np.int8, copy=True)
    if means is None:
        means = np.nanmean(X, axis=0).astype(np.float32)
    if np.isnan(means).any():
        bad = [cols[i] for i in np.where(np.isnan(means))[0]]
        raise RuntimeError(f'all-NaN features: {bad}')
    rr, cc = np.where(np.isnan(X))
    if len(rr):
        X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError('non-finite feature matrix after imputation')
    return X, y, means


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        'brier': float(brier_score_loss(y, p)),
        'auc': float(roc_auc_score(y, p)),
        'ap': float(average_precision_score(y, p)),
        'logloss': float(log_loss(y, p, labels=[0, 1])),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    out = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            out += (m.sum() / len(y)) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(out)


def rank_order(frame: pd.DataFrame, p_raw: np.ndarray) -> np.ndarray:
    # Primary: descending raw score. Secondary keys make exact-score ties
    # deterministic without consulting outcomes or calibrated probabilities.
    return np.lexsort((
        frame.batter_id.to_numpy(dtype=np.int64),
        frame.game_pk.to_numpy(dtype=np.int64),
        -np.asarray(p_raw, dtype=np.float64),
    ))


def bucket_stats(frame: pd.DataFrame, y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray):
    order = rank_order(frame, p_raw)
    out = {}
    selected = {}
    for frac in FRACTIONS:
        n = max(1, int(np.ceil(len(order) * frac)))
        idx = order[:n]
        key = f'top{int(frac*100)}pct'
        obs = float(y[idx].mean())
        mean_cal = float(np.asarray(p_cal)[idx].mean())
        out[key] = {
            'n': int(n),
            'observed_hr_rate': obs,
            'mean_raw_score': float(np.asarray(p_raw)[idx].mean()),
            'mean_calibrated_probability': mean_cal,
            'calibration_gap_observed_minus_predicted': obs - mean_cal,
        }
        selected[key] = set(zip(
            frame.game_pk.to_numpy(dtype=np.int64)[idx].tolist(),
            frame.batter_id.to_numpy(dtype=np.int64)[idx].tolist(),
        ))
    return out, selected


def evaluate_model(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    best: dict,
    rounds: int,
):
    train = df[df.year <= 2021].copy()
    cal = df[df.year == 2022].copy()
    dev = df[df.year.isin([2023, 2024])].copy()

    Xtr, ytr, means = to_matrix(train, feature_cols)
    Xcal, ycal, _ = to_matrix(cal, feature_cols, means)
    Xdev, ydev, _ = to_matrix(dev, feature_cols, means)

    bst = xgb.train(
        xgb_params(best),
        xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols),
        num_boost_round=rounds,
        verbose_eval=False,
    )
    p_cal_raw = bst.predict(xgb.DMatrix(Xcal, feature_names=feature_cols))
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p_cal_raw, ycal)
    p_raw = bst.predict(xgb.DMatrix(Xdev, feature_names=feature_cols))
    p_cal = iso.predict(p_raw)

    overall_buckets, selected = bucket_stats(dev, ydev, p_raw, p_cal)
    by_year = {}
    for year in (2023, 2024):
        m = dev.year.to_numpy() == year
        year_frame = dev.loc[m].reset_index(drop=True)
        b, _ = bucket_stats(year_frame, ydev[m], p_raw[m], p_cal[m])
        by_year[str(year)] = b

    result = {
        'model': name,
        'n_features': int(len(feature_cols)),
        'raw': metrics(ydev, p_raw),
        'calibrated': metrics(ydev, p_cal),
        'ece': ece(ydev, p_cal),
        'raw_rank_buckets': overall_buckets,
        'raw_rank_buckets_by_year': by_year,
    }
    pred = pd.DataFrame({
        'model': name,
        'game_pk': dev.game_pk.to_numpy(dtype=np.int64),
        'batter_id': dev.batter_id.to_numpy(dtype=np.int64),
        'year': dev.year.to_numpy(dtype=np.int16),
        'hr_in_game': ydev,
        'p_raw': p_raw.astype(np.float32),
        'p_cal': np.asarray(p_cal, dtype=np.float32),
    })
    return result, selected, pred


def family_map(active: list[str], summary: dict) -> dict[str, list[str]]:
    qoc = list(summary.get('qoc_feature_names', []))
    if len(qoc) != 20 or not set(qoc).issubset(active):
        raise RuntimeError(f'expected declared 20-feature QoC block, got {len(qoc)}')

    groups = {
        'all_qoc': qoc,
        'barrel': [c for c in qoc if 'barrel_rate' in c],
        'xwoba': [c for c in qoc if 'xwoba_on_contact' in c],
        'pitcher_qoc': [c for c in qoc if c.startswith('pitcher_')],
        'batter_qoc': [c for c in qoc if c.startswith('batter_')],
        'recent_qoc_30d': [c for c in qoc if c.endswith('_30d')],
        'long_horizon_qoc': [c for c in qoc if not c.endswith('_30d')],
        'pitch_type_matchup': [
            c for c in active
            if ('_vs_' in c)
            or c.startswith('pitcher_usage_')
            or c in {'batter_strength_on_pitcher_top_pitch', 'top_pitch_total_pitches_30d'}
        ],
        'park': [c for c in active if c.startswith('park_')],
    }
    for name, cols in groups.items():
        if not cols:
            raise RuntimeError(f'ablation family {name!r} resolved to zero features')
        if len(cols) != len(set(cols)):
            raise RuntimeError(f'duplicate feature inside family {name}')
        missing = sorted(set(cols) - set(active))
        if missing:
            raise RuntimeError(f'family {name} includes inactive features: {missing}')
    return groups


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    active = json.loads(FLIST.read_text())
    summary = json.loads(SUMMARY.read_text())
    aggressive = json.loads(AGGRESSIVE.read_text())
    if aggressive['design']['holdout_2025_read'] is not False:
        raise RuntimeError('sealed holdout contract violated')
    if aggressive['n_features'] != len(active) or len(active) != 73:
        raise RuntimeError('aggressive reference and active feature list disagree')

    best = dict(aggressive['best_params'])
    rounds = int(aggressive['best_round'])
    needed = list(dict.fromkeys(active + ['hr_in_game','game_pk','batter_id','year']))
    df = pd.read_parquet(FEAT, columns=needed)
    if not df.year.between(2015, 2024).all():
        raise RuntimeError('ablation matrix escaped 2015-2024')

    groups = family_map(active, summary)
    (OUT/'feature_families.json').write_text(json.dumps(groups, indent=2))

    experiments = [('full_73', [])] + [(f'no_{name}', cols) for name, cols in groups.items()]
    results = {}
    selections = {}
    preds = []

    for name, removed in experiments:
        cols = [c for c in active if c not in set(removed)]
        if len(cols) != len(active) - len(set(removed)):
            raise RuntimeError(f'feature-count mismatch for {name}')
        print(f'[ablation] {name}: {len(cols)} features; removed {len(removed)}', flush=True)
        r, s, p = evaluate_model(name, df, cols, best, rounds)
        r['removed_features'] = removed
        results[name] = r
        selections[name] = s
        preds.append(p)
        print(
            f"[ablation] {name}: Brier={r['calibrated']['brier']:.6f} "
            f"AUC={r['calibrated']['auc']:.4f} ECE={r['ece']:.5f} "
            f"raw-top5={r['raw_rank_buckets']['top5pct']['observed_hr_rate']:.4f}",
            flush=True,
        )

    # Reproducing the full 73-feature model with the frozen params/rounds must
    # agree with the aggressive run on proper probability metrics.
    full = results['full_73']
    for ours, ref, label in [
        (full['raw']['brier'], aggressive['xgb_raw_test']['brier'], 'raw brier'),
        (full['raw']['auc'], aggressive['xgb_raw_test']['auc'], 'raw auc'),
        (full['calibrated']['brier'], aggressive['xgb_calibrated_test']['brier'], 'cal brier'),
        (full['calibrated']['auc'], aggressive['xgb_calibrated_test']['auc'], 'cal auc'),
        (full['ece'], aggressive['test_ece'], 'ece'),
    ]:
        if abs(float(ours) - float(ref)) > 1e-7:
            raise RuntimeError(f'full-model reproduction mismatch {label}: {ours} vs {ref}')

    # Ranking overlap is measured against the raw-score full-model buckets.
    for name, r in results.items():
        overlap = {}
        if name != 'full_73':
            for key in [f'top{int(f*100)}pct' for f in FRACTIONS]:
                a = selections['full_73'][key]
                b = selections[name][key]
                inter = len(a & b)
                union = len(a | b)
                overlap[key] = {
                    'intersection': inter,
                    'retention_vs_full': inter / len(a),
                    'jaccard_vs_full': inter / union,
                }
        r['ranking_overlap_vs_full'] = overlap

    full_top = {k: v['observed_hr_rate'] for k, v in full['raw_rank_buckets'].items()}
    rows = []
    for name, r in results.items():
        row = {
            'model': name,
            'n_features': r['n_features'],
            'n_removed': len(r['removed_features']),
            'cal_brier': r['calibrated']['brier'],
            'delta_cal_brier_vs_full': r['calibrated']['brier'] - full['calibrated']['brier'],
            'cal_auc': r['calibrated']['auc'],
            'delta_cal_auc_vs_full': r['calibrated']['auc'] - full['calibrated']['auc'],
            'cal_ap': r['calibrated']['ap'],
            'ece': r['ece'],
        }
        for key, stat in r['raw_rank_buckets'].items():
            row[f'{key}_hr_rate'] = stat['observed_hr_rate']
            row[f'{key}_delta_vs_full'] = stat['observed_hr_rate'] - full_top[key]
            if name != 'full_73':
                row[f'{key}_retention_vs_full'] = r['ranking_overlap_vs_full'][key]['retention_vs_full']
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT/'ablation_summary.csv', index=False)
    pd.concat(preds, ignore_index=True).to_parquet(OUT/'development_predictions.parquet', index=False)
    (OUT/'ablation_results.json').write_text(json.dumps({
        'design': {
            'reference': 'full_73 aggressive model',
            'params_frozen_from_aggressive_optuna': True,
            'rounds_frozen': rounds,
            'train': '2015-2021',
            'calibration': '2022',
            'development_assessment': '2023-2024',
            'ranking_score': 'raw_xgboost_probability',
            'ranking_tiebreak': 'game_pk_then_batter_id',
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'families': groups,
        'results': results,
    }, indent=2))

    # Human-readable compact report, ordered by top-5 damage when removed.
    ranked = table.copy()
    ranked['top5_damage_when_removed'] = -ranked['top5pct_delta_vs_full']
    ranked = ranked.sort_values(['model' if False else 'top5_damage_when_removed'], ascending=False)
    lines = [
        '# Trusted v1.2 feature-family ablations', '',
        '**2025 was not read or evaluated.**', '',
        'All ablations use the frozen aggressive Optuna hyperparameters and boosting rounds.',
        'Top buckets are selected by raw XGBoost score; isotonic calibration is evaluated separately.', '',
        '| Model | Features | Cal Brier | Cal AUC | ECE | Raw top10 | Raw top5 | Raw top2 | Raw top1 | Top5 retention |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    by_name = table.set_index('model')
    for name in ['full_73'] + [x for x in ranked.model.tolist() if x != 'full_73']:
        r = by_name.loc[name]
        ret = '-' if name == 'full_73' else f"{100*r['top5pct_retention_vs_full']:.1f}%"
        lines.append(
            f"| {name} | {int(r['n_features'])} | {r['cal_brier']:.6f} | {r['cal_auc']:.4f} | "
            f"{r['ece']:.5f} | {100*r['top10pct_hr_rate']:.2f}% | {100*r['top5pct_hr_rate']:.2f}% | "
            f"{100*r['top2pct_hr_rate']:.2f}% | {100*r['top1pct_hr_rate']:.2f}% | {ret} |"
        )
    lines += ['', '## Removed feature families', '']
    for name, cols in groups.items():
        lines.append(f"- **{name}** ({len(cols)}): " + ', '.join(cols))
    (OUT/'ablation_report.md').write_text('\n'.join(lines) + '\n')

    print('\n' + (OUT/'ablation_report.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
