"""Nearest-rank same-slate matched test for full73 disagreement picks.

This script implements the inference design precommitted in
`docs/v12_matched_pair_precommit_2026-09-04.md`.

For each of the three previously declared obvious-power rank bands (5-8,
9-16, 17+), every full73 daily-top4 selection is paired, without replacement,
with the nearest available non-full73 hitter from the same date and same rank
band. Matching uses absolute obvious-rank distance with deterministic ties.

Primary inference is a 10,000-replicate slate-date clustered bootstrap of the
paired HR-rate lift. An exact one-sided McNemar/binomial test is also reported,
with Holm adjustment across the three bands.

Operational survival was precommitted before this result:
- observed paired lift >= +5.0 percentage points; AND
- 95% paired date-bootstrap CI lower bound > 0.
Strong-signal grade additionally requires CI lower bound >= +2.0 pp.

This is development evidence only (2023-2024). 2025 is rejected.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260904
STRATA = {
    'obvious_rank_5_8': lambda x: x.between(5, 8),
    'obvious_rank_9_16': lambda x: x.between(9, 16),
    'obvious_rank_17_plus': lambda x: x.ge(17),
}
MAGNITUDE_FLOOR = 0.05
STRONG_CI_FLOOR = 0.02


def validate(frame: pd.DataFrame, allowed_years: set[int] | None = None) -> pd.DataFrame:
    need = {
        'game_pk', 'batter_id', 'game_date', 'year', 'hr_in_game',
        'model_top4', 'obvious_rank',
    }
    missing = need - set(frame.columns)
    if missing:
        raise RuntimeError(f'missing columns: {sorted(missing)}')
    f = frame.copy()
    f['game_date'] = pd.to_datetime(f.game_date).dt.normalize()
    if allowed_years is None:
        allowed_years = {2023, 2024}
    years = set(map(int, f.year.unique()))
    if not years.issubset(allowed_years):
        raise RuntimeError(f'matched test escaped allowed years {sorted(allowed_years)}: {sorted(years)}')
    if 2025 in years:
        raise RuntimeError('2025 is sealed')
    if f.duplicated(['game_pk', 'batter_id']).any():
        raise RuntimeError('duplicate batter-game rows')
    if f.obvious_rank.isna().any() or (f.obvious_rank < 1).any():
        raise RuntimeError('invalid obvious-power ranks')
    return f


def _stable_rows(g: pd.DataFrame) -> pd.DataFrame:
    return g.sort_values(['obvious_rank', 'game_pk', 'batter_id'], ascending=[True, True, True])


def match_band(frame: pd.DataFrame, band: pd.Series, band_name: str) -> tuple[pd.DataFrame, dict]:
    """Greedy nearest-rank matching within date/band, without replacement.

    Selected rows are processed in ascending obvious rank. For each selected
    row, the available control minimizing absolute rank distance is chosen;
    ties prefer lower obvious rank, then game_pk, then batter_id. The matching
    rule is deterministic and outcome-blind.
    """
    z = frame.loc[band].copy()
    pairs: list[dict] = []
    total_selected = int(z.model_top4.sum())
    unmatched = 0

    for day, g in z.groupby('game_date', sort=True):
        sel = _stable_rows(g[g.model_top4])
        ctl = _stable_rows(g[~g.model_top4]).copy()
        available = list(ctl.index)

        for sidx, s in sel.iterrows():
            if not available:
                unmatched += 1
                continue
            candidates = ctl.loc[available].copy()
            candidates['_gap'] = (candidates.obvious_rank.astype(float) - float(s.obvious_rank)).abs()
            candidates = candidates.sort_values(
                ['_gap', 'obvious_rank', 'game_pk', 'batter_id'],
                ascending=[True, True, True, True],
            )
            cidx = candidates.index[0]
            c = ctl.loc[cidx]
            available.remove(cidx)
            pairs.append({
                'band': band_name,
                'game_date': pd.Timestamp(day),
                'year': int(s.year),
                'selected_game_pk': int(s.game_pk),
                'selected_batter_id': int(s.batter_id),
                'selected_obvious_rank': int(s.obvious_rank),
                'selected_hr': int(s.hr_in_game),
                'control_game_pk': int(c.game_pk),
                'control_batter_id': int(c.batter_id),
                'control_obvious_rank': int(c.obvious_rank),
                'control_hr': int(c.hr_in_game),
                'rank_gap': int(abs(int(s.obvious_rank) - int(c.obvious_rank))),
            })

    p = pd.DataFrame(pairs)
    if p.empty:
        raise RuntimeError(f'no matched pairs for {band_name}')
    if p.duplicated(['band', 'game_date', 'control_game_pk', 'control_batter_id']).any():
        raise RuntimeError(f'control reused within {band_name}')
    if p.duplicated(['band', 'selected_game_pk', 'selected_batter_id']).any():
        raise RuntimeError(f'selected row duplicated within {band_name}')
    if (p.year != pd.to_datetime(p.game_date).dt.year).any():
        raise RuntimeError('year/date mismatch in pairs')

    diagnostics = {
        'selected_total': total_selected,
        'matched_pairs': int(len(p)),
        'unmatched_selected': int(unmatched),
        'match_fraction': float(len(p) / total_selected) if total_selected else 0.0,
        'rank_gap_mean': float(p.rank_gap.mean()),
        'rank_gap_median': float(p.rank_gap.median()),
        'rank_gap_p90': float(p.rank_gap.quantile(.90)),
        'rank_gap_max': int(p.rank_gap.max()),
        'exact_rank_match_fraction': float((p.rank_gap == 0).mean()),
    }
    return p, diagnostics


def bootstrap_pairs(pairs: pd.DataFrame, reps: int, seed: int) -> dict:
    if reps < 1000:
        raise RuntimeError('requires at least 1000 bootstrap replicates')
    days = np.array(sorted(pairs.game_date.drop_duplicates().to_numpy()))
    if len(days) < 50:
        raise RuntimeError(f'too few matched slate dates: {len(days)}')
    pos = {d: i for i, d in enumerate(days)}
    ss = np.zeros(len(days), dtype=np.float64)
    sc = np.zeros(len(days), dtype=np.float64)
    n = np.zeros(len(days), dtype=np.float64)
    for day, g in pairs.groupby('game_date', sort=False):
        i = pos[np.datetime64(pd.Timestamp(day).to_datetime64())]
        ss[i] = g.selected_hr.sum()
        sc[i] = g.control_hr.sum()
        n[i] = len(g)

    obs_s = float(pairs.selected_hr.mean())
    obs_c = float(pairs.control_hr.mean())
    obs_d = obs_s - obs_c

    rng = np.random.default_rng(seed)
    rs = np.empty(reps, dtype=np.float64)
    rc = np.empty(reps, dtype=np.float64)
    chunk = 1000
    at = 0
    while at < reps:
        k = min(chunk, reps - at)
        idx = rng.integers(0, len(days), size=(k, len(days)), endpoint=False)
        den = n[idx].sum(axis=1)
        rs[at:at+k] = ss[idx].sum(axis=1) / den
        rc[at:at+k] = sc[idx].sum(axis=1) / den
        at += k
    delta = rs - rc
    ci_low = float(np.quantile(delta, .025))
    ci_high = float(np.quantile(delta, .975))
    survives = bool(obs_d >= MAGNITUDE_FLOOR and ci_low > 0.0)
    strong = bool(survives and ci_low >= STRONG_CI_FLOOR)

    return {
        'n_slate_dates': int(len(days)),
        'n_pairs': int(len(pairs)),
        'n_replicates': int(reps),
        'selected': {
            'hr': int(pairs.selected_hr.sum()),
            'rate': obs_s,
            'ci95_low': float(np.quantile(rs, .025)),
            'ci95_high': float(np.quantile(rs, .975)),
        },
        'matched_control': {
            'hr': int(pairs.control_hr.sum()),
            'rate': obs_c,
            'ci95_low': float(np.quantile(rc, .025)),
            'ci95_high': float(np.quantile(rc, .975)),
        },
        'paired_lift': {
            'observed': obs_d,
            'mean': float(delta.mean()),
            'median': float(np.median(delta)),
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'prob_delta_gt_0': float(np.mean(delta > 0)),
        },
        'precommitted_decision': {
            'magnitude_floor': MAGNITUDE_FLOOR,
            'strong_ci_floor': STRONG_CI_FLOOR,
            'ci_clears_zero': bool(ci_low > 0.0),
            'magnitude_floor_met': bool(obs_d >= MAGNITUDE_FLOOR),
            'survives_operational_consideration': survives,
            'strong_signal_grade': strong,
        },
    }


def exact_mcnemar_one_sided(pairs: pd.DataFrame) -> dict:
    b = int(((pairs.selected_hr == 1) & (pairs.control_hr == 0)).sum())
    c = int(((pairs.selected_hr == 0) & (pairs.control_hr == 1)).sum())
    n = b + c
    if n == 0:
        p = 1.0
    else:
        # Under paired null, selected wins among discordant pairs are Binomial(n, .5).
        numerator = sum(math.comb(n, k) for k in range(b, n + 1))
        p = float(numerator / (2 ** n))
    return {
        'selected_only_hr_pairs': b,
        'control_only_hr_pairs': c,
        'discordant_pairs': n,
        'p_one_sided_selected_gt_control': p,
    }


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        raw = min(1.0, (m - i) * p)
        running = max(running, raw)
        adjusted[name] = min(1.0, running)
    return adjusted


def run_scope(frame: pd.DataFrame, reps: int, seed: int, include_holm: bool) -> tuple[dict, pd.DataFrame]:
    results: dict = {}
    all_pairs: list[pd.DataFrame] = []
    raw_p: dict[str, float] = {}
    for i, (name, fn) in enumerate(STRATA.items()):
        pairs, diag = match_band(frame, fn(frame.obvious_rank), name)
        boot = bootstrap_pairs(pairs, reps, seed + i)
        exact = exact_mcnemar_one_sided(pairs)
        results[name] = {
            'matching': diag,
            'date_cluster_bootstrap': boot,
            'exact_matched_binary_test': exact,
        }
        raw_p[name] = exact['p_one_sided_selected_gt_control']
        all_pairs.append(pairs)
    if include_holm:
        adj = holm_adjust(raw_p)
        for name in results:
            results[name]['exact_matched_binary_test']['holm_adjusted_p_across_3_strata'] = float(adj[name])
    return results, pd.concat(all_pairs, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-pairs', required=True)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    f = validate(pd.read_parquet(args.input), {2023, 2024})
    combined, pairs = run_scope(f, args.reps, args.seed, True)
    by_year = {}
    for year, offset in [(2023, 1000), (2024, 2000)]:
        r, _ = run_scope(f[f.year.eq(year)], args.reps, args.seed + offset, False)
        by_year[str(year)] = r

    payload = {
        'design': {
            'purpose': 'same-slate nearest-obvious-rank matched follow-up',
            'strata': ['5-8', '9-16', '17+'],
            'matching': 'same date + same rank band + non-full73 control; nearest obvious rank without replacement; deterministic outcome-blind ties',
            'primary_uncertainty': '10,000-replicate paired slate-date bootstrap',
            'supporting_test': 'exact one-sided matched-binary McNemar/binomial; Holm across three bands',
            'precommitted_observed_magnitude_floor': MAGNITUDE_FLOOR,
            'precommitted_strong_ci_floor': STRONG_CI_FLOOR,
            'development_years': [2023, 2024],
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'combined_2023_2024': combined,
        'by_year': by_year,
    }

    outj = Path(args.out_json)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(payload, indent=2))
    outp = Path(args.out_pairs)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(outp, index=False)

    print(json.dumps(payload, indent=2), flush=True)
    print('[nearest-rank-matched] 2025 NOT READ', flush=True)


if __name__ == '__main__':
    main()
