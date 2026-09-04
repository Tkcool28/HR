"""Predeclared yearly contextual-migration analysis for frozen v1.2 full73.

Scores 2019-2024 walk-forward with the same frozen 73-feature XGBoost contract,
compares daily top-5% selections with the frozen long-horizon batter-only
obvious-power proxy, and measures whether differentiated full73 selections
migrate deeper or shallower in obvious-power rank over time.

2025 is rejected fail-closed and is never scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path('/workspace/hr_model')
FEAT = ROOT / 'features/v1.2_trusted/game_features.parquet'
FLIST = ROOT / 'features/v1.2_trusted/feature_list.json'
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
TOP_FRAC = 0.05
SEED = 20260904
KEYS = ['game_pk', 'batter_id']


def _matrix(df: pd.DataFrame, cols: list[str], means: np.ndarray | None = None):
    X = df[cols].to_numpy(dtype=np.float32, copy=True)
    if means is None:
        means = np.nanmean(X, axis=0).astype(np.float32)
    if np.isnan(means).any():
        raise RuntimeError('all-NaN feature mean in yearly migration scorer')
    rr, cc = np.where(np.isnan(X))
    if len(rr):
        X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError('non-finite yearly migration matrix')
    return X, means


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
    if len(cols) < 5:
        fallback = [
            c for c in active
            if c.startswith('batter_')
            and (c.endswith('_season') or c.endswith('_career'))
            and any(tok in c for tok in ('hr_per_pa', 'barrel_rate', 'xwoba_on_contact', 'avg_ev'))
        ]
        cols = sorted(set(cols + fallback))
    if len(cols) < 5:
        raise RuntimeError(f'insufficient obvious-power proxy features: {cols}')
    return cols


def _group_features(active: list[str]) -> dict[str, list[str]]:
    groups = {
        'park': sorted([c for c in active if 'park_hr_factor' in c]),
        'recent_batter_form': sorted([
            c for c in active
            if c.startswith('batter_')
            and ('14d' in c or '30d' in c)
            and not c.startswith('batter_hr_per_pa_vs_')
            and any(tok in c for tok in (
                'hr_per_pa', 'barrel_rate', 'xwoba_on_contact', 'avg_ev', 'ev90',
                'hard_hit', 'fb_pct', 'sweet_spot', 'iso_xbp'
            ))
        ]),
        'pitcher_vulnerability': sorted([
            c for c in active
            if c.startswith('pitcher_')
            and not c.startswith('pitcher_usage_')
            and any(tok in c for tok in (
                'hr_per_pa', 'barrel_rate_allowed', 'xwoba_on_contact_allowed',
                'hard_hit_pct_allowed', 'avg_ev_allowed', 'iso_xbp_allowed'
            ))
        ]),
        'pitch_matchup': sorted([
            c for c in active
            if c.startswith('batter_hr_per_pa_vs_')
            or c == 'batter_strength_on_pitcher_top_pitch'
        ]),
    }
    return {k: v for k, v in groups.items() if v}


def _composite_percentile(frame: pd.DataFrame, cols: list[str], medians: pd.Series) -> pd.Series:
    pieces = []
    for c in cols:
        v = pd.to_numeric(frame[c], errors='coerce').fillna(float(medians[c]))
        pieces.append(v.groupby(frame.game_date).rank(method='average', pct=True))
    return pd.concat(pieces, axis=1).mean(axis=1)


def _assign_daily_rank(frame: pd.DataFrame, score: str) -> pd.Series:
    ranks = pd.Series(index=frame.index, dtype='int32')
    for _, g in frame.groupby('game_date', sort=True):
        ordered = g.sort_values([score, 'game_pk', 'batter_id'], ascending=[False, True, True])
        ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1, dtype=np.int32)
    return ranks.astype('int32')


def _top5_flag(frame: pd.DataFrame, rank_col: str) -> pd.Series:
    n_by_date = frame.groupby('game_date').size().map(lambda n: max(1, int(np.ceil(n * TOP_FRAC))))
    cut = frame.game_date.map(n_by_date).astype(int)
    return frame[rank_col].le(cut)


def _spearman_year(values: list[float], years: list[int]) -> float:
    a = pd.Series(years, dtype=float).rank(method='average').to_numpy()
    b = pd.Series(values, dtype=float).rank(method='average').to_numpy()
    if np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _slope(years: np.ndarray, vals: np.ndarray) -> float:
    x = years.astype(float)
    xc = x - x.mean()
    return float(np.dot(xc, vals) / np.dot(xc, xc))


def _adjacent_signs(vals: list[float]) -> list[str]:
    out = []
    for a, b in zip(vals[:-1], vals[1:]):
        d = b - a
        out.append('+' if d > 0 else '-' if d < 0 else '0')
    return out


def _bootstrap_trend(scored: pd.DataFrame, reps: int, seed: int, years: list[int]) -> dict:
    rng = np.random.default_rng(seed)
    depth_by_year = []
    overlap_by_year = []
    year_arr = np.array(years, dtype=float)

    for year in years:
        y = scored[scored.year.eq(year)].copy()
        dates = np.array(sorted(y.game_date.drop_duplicates().to_numpy()))
        if len(dates) < 40:
            raise RuntimeError(f'too few slate dates for trend bootstrap {year}: {len(dates)}')
        rows = []
        for d, g in y.groupby('game_date', sort=True):
            mo = g.full73_only
            mt = g.full73_top5
            rows.append({
                'game_date': d,
                'depth_sum': float(g.loc[mo, 'obvious_rank_percentile'].sum()),
                'depth_n': int(mo.sum()),
                'shared_n': int((g.full73_top5 & g.obvious_top5).sum()),
                'full_n': int(mt.sum()),
            })
        z = pd.DataFrame(rows)
        idx = rng.integers(0, len(z), size=(reps, len(z)), endpoint=False)
        depth_den = z.depth_n.to_numpy(float)[idx].sum(1)
        depth_num = z.depth_sum.to_numpy(float)[idx].sum(1)
        depth = np.divide(depth_num, depth_den, out=np.full(reps, np.nan), where=depth_den > 0)
        overlap = z.shared_n.to_numpy(float)[idx].sum(1) / z.full_n.to_numpy(float)[idx].sum(1)
        if not np.isfinite(depth).all():
            raise RuntimeError(f'non-finite bootstrap depth for {year}')
        depth_by_year.append(depth)
        overlap_by_year.append(overlap)

    depth_mat = np.vstack(depth_by_year).T
    overlap_mat = np.vstack(overlap_by_year).T
    xc = year_arr - year_arr.mean()
    den = np.dot(xc, xc)
    depth_slopes = depth_mat @ xc / den
    overlap_slopes = overlap_mat @ xc / den

    def dist(x: np.ndarray) -> dict:
        return {
            'mean': float(np.mean(x)),
            'median': float(np.median(x)),
            'ci95_low': float(np.quantile(x, 0.025)),
            'ci95_high': float(np.quantile(x, 0.975)),
            'prob_gt_0': float(np.mean(x > 0)),
        }

    return {
        'years': years,
        'depth_slope_bootstrap': dist(depth_slopes),
        'overlap_slope_bootstrap': dist(overlap_slopes),
    }


def _year_summary(g: pd.DataFrame) -> dict:
    mo = g[g.full73_only]
    succ = mo[mo.hr_in_game.eq(1)]
    shared = int((g.full73_top5 & g.obvious_top5).sum())
    n_full = int(g.full73_top5.sum())
    return {
        'n_slate_dates': int(g.game_date.nunique()),
        'n_rows': int(len(g)),
        'n_full73_top5': n_full,
        'n_obvious_top5': int(g.obvious_top5.sum()),
        'n_shared_top5': shared,
        'top5_overlap_fraction': float(shared / n_full),
        'n_full73_only': int(len(mo)),
        'full73_only_hr': int(mo.hr_in_game.sum()),
        'full73_only_hr_rate': float(mo.hr_in_game.mean()),
        'full73_only_obvious_rank_mean': float(mo.obvious_rank.mean()),
        'full73_only_obvious_rank_median': float(mo.obvious_rank.median()),
        'full73_only_depth_mean': float(mo.obvious_rank_percentile.mean()),
        'full73_only_depth_median': float(mo.obvious_rank_percentile.median()),
        'full73_only_depth_p75': float(mo.obvious_rank_percentile.quantile(.75)),
        'successful_full73_only_n': int(len(succ)),
        'successful_full73_only_depth_mean': None if len(succ) == 0 else float(succ.obvious_rank_percentile.mean()),
        'successful_full73_only_depth_median': None if len(succ) == 0 else float(succ.obvious_rank_percentile.median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-year-csv', required=True)
    ap.add_argument('--out-context-csv', required=True)
    ap.add_argument('--out-parquet', required=True)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()
    if args.reps < 1000:
        raise RuntimeError('yearly migration bootstrap requires >=1000 reps')

    active = json.loads(FLIST.read_text())
    if len(active) != 73:
        raise RuntimeError(f'expected frozen 73 features, got {len(active)}')
    contract = json.loads(Path(args.contract).read_text())
    params = dict(contract['best_params'])
    rounds = int(contract['best_round'])
    if rounds != 194:
        raise RuntimeError(f'frozen full73 round contract changed: {rounds}')

    obvious = _present_obvious(active)
    groups = _group_features(active)
    context_cols = sorted(set(c for vv in groups.values() for c in vv))
    needed = list(dict.fromkeys(active + ['game_pk', 'batter_id', 'game_date', 'year', 'hr_in_game'] + obvious + context_cols))
    df = pd.read_parquet(FEAT, columns=needed)
    df['game_date'] = pd.to_datetime(df.game_date).dt.normalize()
    if not df.year.between(2015, 2024).all() or 2025 in set(df.year.astype(int)):
        raise RuntimeError('2025 present or trusted year scope violated')
    if df.duplicated(KEYS).any():
        raise RuntimeError('duplicate batter-game keys')

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'seed': 42,
        'nthread': -1,
        **params,
    }

    scored_years = []
    fit_metrics = {}
    for year in YEARS:
        train = df[df.year.lt(year)].copy()
        test = df[df.year.eq(year)].copy()
        if train.year.max() != year - 1 or len(test) == 0:
            raise RuntimeError(f'bad walk-forward partition for {year}')

        Xtr, means = _matrix(train, active)
        Xte, _ = _matrix(test, active, means)
        ytr = train.hr_in_game.to_numpy(np.int8)
        yte = test.hr_in_game.to_numpy(np.int8)
        bst = xgb.train(
            xgb_params,
            xgb.DMatrix(Xtr, label=ytr, feature_names=active),
            num_boost_round=rounds,
            verbose_eval=False,
        )
        p = bst.predict(xgb.DMatrix(Xte, feature_names=active)).astype(np.float32)

        training_medians = train[sorted(set(obvious + context_cols))].median(numeric_only=True)
        if training_medians.isna().any():
            bad = training_medians[training_medians.isna()].index.tolist()
            raise RuntimeError(f'all-NaN walk-forward medians {year}: {bad}')

        z = test[['game_pk', 'batter_id', 'game_date', 'year', 'hr_in_game'] + sorted(set(obvious + context_cols))].copy()
        z['p_raw'] = p
        z['obvious_power_score'] = _composite_percentile(z, obvious, training_medians)
        for name, cols in groups.items():
            z[f'{name}_score'] = _composite_percentile(z, cols, training_medians)

        z['model_rank'] = _assign_daily_rank(z, 'p_raw')
        z['obvious_rank'] = _assign_daily_rank(z, 'obvious_power_score')
        z['slate_size'] = z.groupby('game_date').game_pk.transform('size').astype(int)
        denom = np.maximum(z.slate_size.to_numpy(float) - 1.0, 1.0)
        z['obvious_rank_percentile'] = (z.obvious_rank.to_numpy(float) - 1.0) / denom
        z['full73_top5'] = _top5_flag(z, 'model_rank')
        z['obvious_top5'] = _top5_flag(z, 'obvious_rank')
        z['full73_only'] = z.full73_top5 & ~z.obvious_top5
        z['shared_top5'] = z.full73_top5 & z.obvious_top5
        scored_years.append(z)

        fit_metrics[str(year)] = {
            'train_year_min': int(train.year.min()),
            'train_year_max': int(train.year.max()),
            'n_train': int(len(train)),
            'n_test': int(len(test)),
            'raw_brier': float(brier_score_loss(yte, p)),
            'raw_auc': float(roc_auc_score(yte, p)),
        }
        print(f'[migration] scored {year}: train<= {year-1}, n={len(test):,}', flush=True)

    scored = pd.concat(scored_years, ignore_index=True)
    if set(scored.year.astype(int)) != set(YEARS):
        raise RuntimeError('scored years mismatch')

    summaries = {str(y): _year_summary(scored[scored.year.eq(y)]) for y in YEARS}
    depth_vals = [summaries[str(y)]['full73_only_depth_mean'] for y in YEARS]
    overlap_vals = [summaries[str(y)]['top5_overlap_fraction'] for y in YEARS]
    year_arr = np.array(YEARS, dtype=float)

    primary = {
        'depth_mean_linear_slope_per_year': _slope(year_arr, np.array(depth_vals)),
        'overlap_linear_slope_per_year': _slope(year_arr, np.array(overlap_vals)),
        'depth_mean_spearman_year': _spearman_year(depth_vals, YEARS),
        'overlap_spearman_year': _spearman_year(overlap_vals, YEARS),
        'depth_adjacent_change_signs': _adjacent_signs(depth_vals),
        'overlap_adjacent_change_signs': _adjacent_signs(overlap_vals),
    }
    primary['bootstrap_all_years'] = _bootstrap_trend(scored, args.reps, args.seed, YEARS)

    ex2020 = [y for y in YEARS if y != 2020]
    d2 = [summaries[str(y)]['full73_only_depth_mean'] for y in ex2020]
    o2 = [summaries[str(y)]['top5_overlap_fraction'] for y in ex2020]
    primary['sensitivity_excluding_2020'] = {
        'years': ex2020,
        'depth_mean_linear_slope_per_year': _slope(np.array(ex2020, float), np.array(d2)),
        'overlap_linear_slope_per_year': _slope(np.array(ex2020, float), np.array(o2)),
        'depth_mean_spearman_year': _spearman_year(d2, ex2020),
        'overlap_spearman_year': _spearman_year(o2, ex2020),
        'depth_adjacent_change_signs': _adjacent_signs(d2),
        'overlap_adjacent_change_signs': _adjacent_signs(o2),
        'bootstrap': _bootstrap_trend(scored, args.reps, args.seed + 100, ex2020),
    }

    context_rows = []
    context_trends = {}
    mo = scored[scored.full73_only].copy()
    for name in groups:
        col = f'{name}_score'
        yearly = []
        for y in YEARS:
            g = mo[mo.year.eq(y)]
            mean_score = float(g[col].mean())
            corr = float(g[[col, 'obvious_rank_percentile']].corr(method='spearman').iloc[0, 1]) if len(g) >= 20 else np.nan
            yearly.append(mean_score)
            context_rows.append({
                'year': y,
                'context_group': name,
                'n_full73_only': int(len(g)),
                'mean_context_percentile': mean_score,
                'spearman_context_vs_obvious_depth': corr,
            })
        context_trends[name] = {
            'mean_context_score_linear_slope_per_year': _slope(year_arr, np.array(yearly)),
            'overall_spearman_context_vs_obvious_depth': float(mo[[col, 'obvious_rank_percentile']].corr(method='spearman').iloc[0, 1]),
        }

    payload = {
        'design': {
            'years': YEARS,
            'selector': 'daily raw-score top 5%',
            'full73_contract': 'same 73 features / frozen champion params / 194 rounds; expanding train through year-1',
            'obvious_proxy': obvious,
            'primary_depth': '(obvious_rank-1)/(slate_size-1) among full73-only top5 selections',
            'bootstrap_replicates': int(args.reps),
            'bootstrap_seed': int(args.seed),
            'primary_includes_2020': True,
            'predeclared_ex2020_sensitivity': True,
            'fixed_rank_buckets_analyzed': False,
            'calendar_month_outcomes_emitted': False,
            'sealed_final_holdout': 2025,
            'holdout_2025_read': False,
        },
        'walk_forward_fit_metrics': fit_metrics,
        'yearly': summaries,
        'trend': primary,
        'context_mechanism_trends': context_trends,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    pd.DataFrame([{'year': y, **summaries[str(y)]} for y in YEARS]).to_csv(args.out_year_csv, index=False)
    pd.DataFrame(context_rows).to_csv(args.out_context_csv, index=False)
    scored.to_parquet(args.out_parquet, index=False)

    print(json.dumps({
        'design': payload['design'],
        'yearly': payload['yearly'],
        'trend': payload['trend'],
        'context_mechanism_trends': payload['context_mechanism_trends'],
    }, indent=2), flush=True)
    print('[yearly-context-migration] 2025 NOT READ; no threshold buckets or monthly outcome tables emitted', flush=True)


if __name__ == '__main__':
    main()
