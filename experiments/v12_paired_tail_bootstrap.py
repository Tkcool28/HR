"""Paired bootstrap for actionable HR-model tail selectors.

Primary inference unit: game date / daily slate.

Why date-clustered bootstrap?
-----------------------------
The deployed decision is made by ranking all available batter-games *within a
slate*. Batters on the same date compete for the same top-5%/top-N cutoff and
share weather/league/context shocks. Resampling individual batter rows would
therefore understate uncertainty. Sampling slate dates with replacement keeps
the full daily decision problem intact and gives each paired model the exact
same bootstrap draw.

The script expects two frozen 2023-2024 development prediction tables with the
same batter-game keys and outcomes. Ranking uses raw XGBoost probability.
Calibrated probability is never used for selection.

An optional whole-game-cluster sensitivity mode resamples games within each
slate and recomputes the selectors while preserving all batter rows from each
sampled game. It is secondary; date-clustered inference is the primary result
for the product use case.

2025 must not be present in either input.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['game_pk', 'batter_id']
SEED = 20260904


@dataclass(frozen=True)
class Selector:
    name: str
    frac: float | None = None
    top_n: int | None = None


SELECTORS = (
    Selector('daily_top5pct', frac=0.05),
    Selector('daily_top4', top_n=4),
)


def deterministic_order(g: pd.DataFrame) -> np.ndarray:
    return np.lexsort((
        g.batter_id.to_numpy(dtype=np.int64),
        g.game_pk.to_numpy(dtype=np.int64),
        -g.p_raw.to_numpy(dtype=np.float64),
    ))


def select_one_day(g: pd.DataFrame, selector: Selector) -> pd.DataFrame:
    order = deterministic_order(g)
    if selector.frac is not None:
        n = max(1, int(np.ceil(len(g) * selector.frac)))
    elif selector.top_n is not None:
        n = min(int(selector.top_n), len(g))
    else:
        raise RuntimeError('invalid selector')
    return g.iloc[order[:n]]


def daily_counts(frame: pd.DataFrame, selector: Selector) -> pd.DataFrame:
    rows = []
    for day, g in frame.groupby('game_date', sort=True):
        s = select_one_day(g, selector)
        rows.append({
            'game_date': pd.Timestamp(day),
            'successes': int(s.hr_in_game.sum()),
            'n': int(len(s)),
            'rate': float(s.hr_in_game.mean()),
        })
    return pd.DataFrame(rows)


def validate_pair(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = {'game_pk','batter_id','game_date','year','hr_in_game','p_raw'}
    for name, f in [('A', a), ('B', b)]:
        missing = need - set(f.columns)
        if missing:
            raise RuntimeError(f'{name} missing columns: {sorted(missing)}')
        if not f.year.between(2023, 2024).all():
            raise RuntimeError(f'{name} escaped 2023-2024 development years')
        if f.duplicated(KEYS).any():
            raise RuntimeError(f'{name} has duplicate batter-game keys')
        f['game_date'] = pd.to_datetime(f.game_date).dt.normalize()

    aa = a.sort_values(KEYS).reset_index(drop=True)
    bb = b.sort_values(KEYS).reset_index(drop=True)
    if not aa[KEYS].equals(bb[KEYS]):
        raise RuntimeError('paired models do not have identical batter-game keys')
    if not np.array_equal(aa.hr_in_game.to_numpy(), bb.hr_in_game.to_numpy()):
        raise RuntimeError('paired models disagree on outcomes')
    if not np.array_equal(aa.year.to_numpy(), bb.year.to_numpy()):
        raise RuntimeError('paired models disagree on years')
    if not np.array_equal(aa.game_date.to_numpy(), bb.game_date.to_numpy()):
        raise RuntimeError('paired models disagree on game dates')
    return aa, bb


def bootstrap_dates(
    a_counts: pd.DataFrame,
    b_counts: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict:
    m = a_counts.merge(b_counts, on='game_date', suffixes=('_a','_b'), validate='one_to_one')
    if len(m) != len(a_counts) or len(m) != len(b_counts):
        raise RuntimeError('date sets differ between paired models')
    n_days = len(m)
    if n_days < 100:
        raise RuntimeError(f'too few slate dates for bootstrap: {n_days}')

    sa = m.successes_a.to_numpy(dtype=np.float64)
    na = m.n_a.to_numpy(dtype=np.float64)
    sb = m.successes_b.to_numpy(dtype=np.float64)
    nb = m.n_b.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Chunked vectorization avoids constructing one enormous reps x dates array.
    ra = np.empty(reps, dtype=np.float64)
    rb = np.empty(reps, dtype=np.float64)
    chunk = 1000
    pos = 0
    while pos < reps:
        k = min(chunk, reps - pos)
        idx = rng.integers(0, n_days, size=(k, n_days), endpoint=False)
        ra[pos:pos+k] = sa[idx].sum(axis=1) / na[idx].sum(axis=1)
        rb[pos:pos+k] = sb[idx].sum(axis=1) / nb[idx].sum(axis=1)
        pos += k

    delta = rb - ra
    observed_a = float(sa.sum() / na.sum())
    observed_b = float(sb.sum() / nb.sum())
    observed_delta = observed_b - observed_a

    def dist(x: np.ndarray) -> dict:
        return {
            'mean': float(np.mean(x)),
            'median': float(np.median(x)),
            'ci95_low': float(np.quantile(x, 0.025)),
            'ci95_high': float(np.quantile(x, 0.975)),
        }

    return {
        'n_slate_dates': int(n_days),
        'n_replicates': int(reps),
        'observed_a': observed_a,
        'observed_b': observed_b,
        'observed_delta_b_minus_a': observed_delta,
        'bootstrap_a': dist(ra),
        'bootstrap_b': dist(rb),
        'bootstrap_delta_b_minus_a': dist(delta),
        'prob_delta_gt_0': float(np.mean(delta > 0)),
        'prob_delta_ge_0': float(np.mean(delta >= 0)),
    }


def _selector_n(total_rows: int, selector: Selector) -> int:
    if selector.frac is not None:
        return max(1, int(np.ceil(total_rows * selector.frac)))
    if selector.top_n is not None:
        return min(int(selector.top_n), total_rows)
    raise RuntimeError('invalid selector')


def _prepare_game_blocks(a: pd.DataFrame, b: pd.DataFrame):
    """Precompute paired per-game NumPy blocks for fast cluster sensitivity."""
    days = sorted(pd.Timestamp(x) for x in a.game_date.unique())
    prepared = []
    for day in days:
        ga = a[a.game_date.eq(day)]
        gb = b[b.game_date.eq(day)]
        games_a = sorted(ga.game_pk.astype(int).unique())
        games_b = sorted(gb.game_pk.astype(int).unique())
        if games_a != games_b:
            raise RuntimeError(f'paired game sets differ on {day.date()}')
        blocks = []
        for game_pk in games_a:
            xa = ga[ga.game_pk.eq(game_pk)].sort_values('batter_id')
            xb = gb[gb.game_pk.eq(game_pk)].sort_values('batter_id')
            if len(xa) != len(xb) or not np.array_equal(
                xa.batter_id.to_numpy(dtype=np.int64), xb.batter_id.to_numpy(dtype=np.int64)
            ):
                raise RuntimeError(f'paired batter rows differ for game {game_pk}')
            ya = xa.hr_in_game.to_numpy(dtype=np.int8)
            yb = xb.hr_in_game.to_numpy(dtype=np.int8)
            if not np.array_equal(ya, yb):
                raise RuntimeError(f'paired outcomes differ for game {game_pk}')
            blocks.append({
                'batter': xa.batter_id.to_numpy(dtype=np.int64),
                'hr': ya,
                'score_a': xa.p_raw.to_numpy(dtype=np.float64),
                'score_b': xb.p_raw.to_numpy(dtype=np.float64),
            })
        prepared.append((day, blocks))
    return prepared


def _score_sampled_blocks(blocks, sampled: np.ndarray, selector: Selector):
    batters = []
    hrs = []
    scores_a = []
    scores_b = []
    occurrences = []
    for occurrence, block_idx in enumerate(sampled, start=1):
        block = blocks[int(block_idx)]
        n = len(block['hr'])
        batters.append(block['batter'])
        hrs.append(block['hr'])
        scores_a.append(block['score_a'])
        scores_b.append(block['score_b'])
        occurrences.append(np.full(n, occurrence, dtype=np.int64))
    batter = np.concatenate(batters)
    hr = np.concatenate(hrs)
    sa = np.concatenate(scores_a)
    sb = np.concatenate(scores_b)
    occ = np.concatenate(occurrences)
    n_select = _selector_n(len(hr), selector)
    order_a = np.lexsort((batter, occ, -sa))[:n_select]
    order_b = np.lexsort((batter, occ, -sb))[:n_select]
    return int(hr[order_a].sum()), n_select, int(hr[order_b].sum()), n_select


def bootstrap_games_within_days(
    a: pd.DataFrame,
    b: pd.DataFrame,
    selector: Selector,
    reps: int,
    seed: int,
) -> dict | None:
    if reps <= 0:
        return None
    prepared = _prepare_game_blocks(a, b)
    if len(prepared) < 100:
        raise RuntimeError(f'too few slate dates for game-cluster sensitivity: {len(prepared)}')
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=np.float64)
    arates = np.empty(reps, dtype=np.float64)
    brates = np.empty(reps, dtype=np.float64)

    for r in range(reps):
        a_succ = a_n = b_succ = b_n = 0
        for _, blocks in prepared:
            sampled = rng.integers(0, len(blocks), size=len(blocks), endpoint=False)
            sa, na, sb, nb = _score_sampled_blocks(blocks, sampled, selector)
            a_succ += sa; a_n += na
            b_succ += sb; b_n += nb
        arates[r] = a_succ / a_n
        brates[r] = b_succ / b_n
        deltas[r] = brates[r] - arates[r]

    def dist(x: np.ndarray) -> dict:
        return {
            'mean': float(np.mean(x)),
            'median': float(np.median(x)),
            'ci95_low': float(np.quantile(x, 0.025)),
            'ci95_high': float(np.quantile(x, 0.975)),
        }

    return {
        'n_slate_dates': int(len(prepared)),
        'n_replicates': int(reps),
        'bootstrap_a': dist(arates),
        'bootstrap_b': dist(brates),
        'bootstrap_delta_b_minus_a': dist(deltas),
        'prob_delta_gt_0': float(np.mean(deltas > 0)),
        'prob_delta_ge_0': float(np.mean(deltas >= 0)),
    }


def run_scope(a: pd.DataFrame, b: pd.DataFrame, reps: int, game_reps: int, seed: int) -> dict:
    results = {}
    for i, selector in enumerate(SELECTORS):
        ac = daily_counts(a, selector)
        bc = daily_counts(b, selector)
        primary = bootstrap_dates(ac, bc, reps=reps, seed=seed + i)
        secondary = bootstrap_games_within_days(
            a, b, selector, reps=game_reps, seed=seed + 100 + i
        )
        results[selector.name] = {
            'date_cluster_primary': primary,
            'game_within_slate_secondary': secondary,
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='reference prediction parquet')
    ap.add_argument('--b', required=True, help='challenger prediction parquet')
    ap.add_argument('--label-a', default='reference')
    ap.add_argument('--label-b', default='challenger')
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--game-reps', type=int, default=0,
                    help='optional whole-game-cluster sensitivity replicates')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    if args.reps < 1000:
        raise RuntimeError('primary bootstrap requires at least 1000 replicates')

    a = pd.read_parquet(args.a)
    b = pd.read_parquet(args.b)
    a, b = validate_pair(a, b)

    payload = {
        'design': {
            'reference_model': args.label_a,
            'challenger_model': args.label_b,
            'primary_cluster': 'game_date/slate',
            'secondary_cluster': 'whole_game_within_slate',
            'paired_draws': True,
            'ranking_score': 'raw_xgboost_probability_or_frozen_proxy_score',
            'selectors': ['daily_top5pct','daily_top4'],
            'development_years': [2023, 2024],
            'bootstrap_seed': int(args.seed),
            'primary_replicates': int(args.reps),
            'secondary_game_replicates': int(args.game_reps),
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'combined_2023_2024': run_scope(a, b, args.reps, args.game_reps, args.seed),
        'by_year': {},
    }
    for j, year in enumerate((2023, 2024)):
        payload['by_year'][str(year)] = run_scope(
            a[a.year.eq(year)], b[b.year.eq(year)], args.reps, 0, args.seed + 1000 + j * 10
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print('[tail-bootstrap] 2025 NOT READ')


if __name__ == '__main__':
    main()
