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

An optional slower game-cluster sensitivity mode resamples games within each
slate and recomputes the selectors. It is secondary; date-clustered inference
is the primary result for the product use case.

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


def resample_games_within_day(g: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    games = g.game_pk.drop_duplicates().to_numpy(dtype=np.int64)
    sampled = rng.choice(games, size=len(games), replace=True)
    pieces = []
    # Give duplicate sampled games unique bootstrap IDs for deterministic ties
    # while preserving each full 18-row game cluster.
    for occurrence, game_pk in enumerate(sampled):
        part = g[g.game_pk.eq(game_pk)].copy()
        part['game_pk'] = np.int64(occurrence + 1)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_games_within_days(
    a: pd.DataFrame,
    b: pd.DataFrame,
    selector: Selector,
    reps: int,
    seed: int,
) -> dict | None:
    if reps <= 0:
        return None
    days = sorted(pd.Timestamp(x) for x in a.game_date.unique())
    a_by_day = {d: a[a.game_date.eq(d)].copy() for d in days}
    b_by_day = {d: b[b.game_date.eq(d)].copy() for d in days}
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=np.float64)
    arates = np.empty(reps, dtype=np.float64)
    brates = np.empty(reps, dtype=np.float64)

    for r in range(reps):
        a_succ = a_n = b_succ = b_n = 0
        for d in days:
            # Draw the game multiplicities once, then apply the identical draw to
            # both models so the comparison remains paired.
            ga = a_by_day[d]
            gb = b_by_day[d]
            games = ga.game_pk.drop_duplicates().to_numpy(dtype=np.int64)
            sampled = rng.choice(games, size=len(games), replace=True)
            pa = []
            pb = []
            for occurrence, game_pk in enumerate(sampled):
                xa = ga[ga.game_pk.eq(game_pk)].copy()
                xb = gb[gb.game_pk.eq(game_pk)].copy()
                xa['game_pk'] = np.int64(occurrence + 1)
                xb['game_pk'] = np.int64(occurrence + 1)
                pa.append(xa); pb.append(xb)
            sa = select_one_day(pd.concat(pa, ignore_index=True), selector)
            sb = select_one_day(pd.concat(pb, ignore_index=True), selector)
            a_succ += int(sa.hr_in_game.sum()); a_n += len(sa)
            b_succ += int(sb.hr_in_game.sum()); b_n += len(sb)
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
        'n_replicates': int(reps),
        'bootstrap_a': dist(arates),
        'bootstrap_b': dist(brates),
        'bootstrap_delta_b_minus_a': dist(deltas),
        'prob_delta_gt_0': float(np.mean(deltas > 0)),
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
                    help='optional slower within-slate game-cluster replicates')
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
            'paired_draws': True,
            'ranking_score': 'raw_xgboost_probability',
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
