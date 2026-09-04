"""Daily-slate ranking diagnostics for trusted v1.2 ablation predictions.

Uses only 2023-2024 development predictions already produced by the frozen
feature-family ablation experiment. 2025 is never read.

Unlike a pooled two-year percentile, this ranks candidates independently on
each MLB game date, which better matches the live product question: who are
the strongest HR candidates on today's slate?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
PRED = ROOT/'models/v1.2_ablations/development_predictions.parquet'
FEAT = ROOT/'features/v1.2_trusted/game_features.parquet'
OUT = ROOT/'models/v1.2_ablations'
FRACTIONS = (0.10, 0.05, 0.02, 0.01)
TOP_NS = (1, 2, 4, 8)


def deterministic_order(g: pd.DataFrame) -> np.ndarray:
    return np.lexsort((
        g.batter_id.to_numpy(dtype=np.int64),
        g.game_pk.to_numpy(dtype=np.int64),
        -g.p_raw.to_numpy(dtype=np.float64),
    ))


def summarize_selected(sel: pd.DataFrame) -> dict:
    obs = float(sel.hr_in_game.mean()) if len(sel) else float('nan')
    pred = float(sel.p_cal.mean()) if len(sel) else float('nan')
    return {
        'n': int(len(sel)),
        'observed_hr_rate': obs,
        'mean_calibrated_probability': pred,
        'calibration_gap_observed_minus_predicted': obs - pred,
        'n_dates': int(sel.game_date.nunique()),
    }


def daily_select(frame: pd.DataFrame, *, frac: float | None = None, top_n: int | None = None) -> pd.DataFrame:
    if (frac is None) == (top_n is None):
        raise ValueError('set exactly one of frac/top_n')
    parts = []
    for _, g in frame.groupby('game_date', sort=True):
        order = deterministic_order(g)
        if frac is not None:
            n = max(1, int(np.ceil(len(g) * frac)))
        else:
            n = min(int(top_n), len(g))
        parts.append(g.iloc[order[:n]])
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def keyset(df: pd.DataFrame) -> set[tuple[str, int, int]]:
    return set(zip(
        df.game_date.astype(str).tolist(),
        df.game_pk.astype(int).tolist(),
        df.batter_id.astype(int).tolist(),
    ))


def main() -> None:
    pred = pd.read_parquet(PRED)
    ctx = pd.read_parquet(FEAT, columns=['game_pk','batter_id','game_date'])
    ctx = ctx.drop_duplicates(['game_pk','batter_id'])
    pred = pred.merge(ctx, on=['game_pk','batter_id'], how='left', validate='many_to_one')
    pred['game_date'] = pd.to_datetime(pred.game_date).dt.normalize()
    if pred.game_date.isna().any():
        raise RuntimeError('missing game_date after prediction/context join')
    if not pred.year.between(2023, 2024).all():
        raise RuntimeError('daily ranking diagnostics escaped 2023-2024 development set')

    models = pred.model.drop_duplicates().tolist()
    if 'full_73' not in models:
        raise RuntimeError('missing full_73 reference predictions')

    selections: dict[str, dict[str, pd.DataFrame]] = {}
    results = {}
    for model in models:
        f = pred[pred.model.eq(model)].copy()
        d = {}
        s = {}
        for frac in FRACTIONS:
            key = f'daily_top{int(frac*100)}pct'
            sel = daily_select(f, frac=frac)
            d[key] = summarize_selected(sel)
            s[key] = sel
        for n in TOP_NS:
            key = f'daily_top{n}'
            sel = daily_select(f, top_n=n)
            d[key] = summarize_selected(sel)
            s[key] = sel

        # Year-specific top-N and percentile results, still selected within day.
        by_year = {}
        for year in (2023, 2024):
            fy = f[f.year.eq(year)]
            yd = {}
            for frac in FRACTIONS:
                key = f'daily_top{int(frac*100)}pct'
                yd[key] = summarize_selected(daily_select(fy, frac=frac))
            for n in TOP_NS:
                key = f'daily_top{n}'
                yd[key] = summarize_selected(daily_select(fy, top_n=n))
            by_year[str(year)] = yd

        results[model] = {'overall': d, 'by_year': by_year}
        selections[model] = s

    # Set overlap with the full model measures how much each family changes the
    # names users would actually see on a daily shortlist.
    for model in models:
        overlap = {}
        if model != 'full_73':
            for key, full_sel in selections['full_73'].items():
                a = keyset(full_sel)
                b = keyset(selections[model][key])
                inter = len(a & b)
                overlap[key] = {
                    'retention_vs_full': inter / len(a),
                    'jaccard_vs_full': inter / len(a | b),
                }
        results[model]['overlap_vs_full'] = overlap

    # Compact comparison table emphasizing the practical four-name slate.
    full = results['full_73']['overall']
    rows = []
    for model in models:
        r = results[model]
        row = {'model': model}
        for key in [f'daily_top{int(f*100)}pct' for f in FRACTIONS] + [f'daily_top{n}' for n in TOP_NS]:
            hr = r['overall'][key]['observed_hr_rate']
            row[f'{key}_hr_rate'] = hr
            row[f'{key}_delta_vs_full'] = hr - full[key]['observed_hr_rate']
            row[f'{key}_mean_cal_probability'] = r['overall'][key]['mean_calibrated_probability']
            if model != 'full_73':
                row[f'{key}_retention_vs_full'] = r['overlap_vs_full'][key]['retention_vs_full']
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'daily_ranking_summary.csv', index=False)

    payload = {
        'design': {
            'development_years': [2023, 2024],
            'rank_within': 'game_date',
            'ranking_score': 'raw_xgboost_probability',
            'calibration_used_for_selection': False,
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'results': results,
    }
    (OUT/'daily_ranking_results.json').write_text(json.dumps(payload, indent=2))

    # Human-readable report.
    idx = table.set_index('model')
    order = ['full_73'] + sorted(
        [m for m in models if m != 'full_73'],
        key=lambda m: idx.loc[m, 'daily_top4_hr_rate'],
        reverse=True,
    )
    lines = [
        '# Daily-slate ranking diagnostics', '',
        '**2025 was not read or evaluated.**', '',
        'Candidates are ranked independently on each game date using raw XGBoost scores.',
        'This avoids pooled-year cutoff composition effects and isotonic tie artifacts.', '',
        '| Model | Daily top10% | top5% | top2% | top1% | Top 1/day | Top 2/day | Top 4/day | Top 8/day | Top4 retention |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for model in order:
        r = idx.loc[model]
        ret = '-' if model == 'full_73' else f"{100*r['daily_top4_retention_vs_full']:.1f}%"
        lines.append(
            f"| {model} | {100*r['daily_top10pct_hr_rate']:.2f}% | {100*r['daily_top5pct_hr_rate']:.2f}% | "
            f"{100*r['daily_top2pct_hr_rate']:.2f}% | {100*r['daily_top1pct_hr_rate']:.2f}% | "
            f"{100*r['daily_top1_hr_rate']:.2f}% | {100*r['daily_top2_hr_rate']:.2f}% | "
            f"{100*r['daily_top4_hr_rate']:.2f}% | {100*r['daily_top8_hr_rate']:.2f}% | {ret} |"
        )
    lines += ['', '## Full-model year stability', '']
    for year in ('2023','2024'):
        y = results['full_73']['by_year'][year]
        lines.append(
            f"- **{year}:** top5% {100*y['daily_top5pct']['observed_hr_rate']:.2f}%, "
            f"top2% {100*y['daily_top2pct']['observed_hr_rate']:.2f}%, "
            f"top4/day {100*y['daily_top4']['observed_hr_rate']:.2f}%"
        )
    (OUT/'daily_ranking_report.md').write_text('\n'.join(lines) + '\n')
    print((OUT/'daily_ranking_report.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
