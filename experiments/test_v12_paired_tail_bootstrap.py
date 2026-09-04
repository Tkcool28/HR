from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Pytest can choose experiments/ as the import root in CI. Put this file's
# directory explicitly on sys.path and import the sibling module normally so
# dataclass/module metadata is registered in sys.modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v12_paired_tail_bootstrap as boot


def synthetic_pair(n_days: int = 120):
    rows_a = []
    rows_b = []
    start = pd.Timestamp('2023-04-01')
    batter_seed = 100000
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)
        year = int(day.year)
        # 2 games x 18 batter-games = 36 candidates per slate. Daily top5%
        # therefore selects exactly two rows.
        ids = []
        for game_offset in range(2):
            game_pk = 700000 + d * 2 + game_offset
            for j in range(18):
                batter_id = batter_seed + d * 100 + game_offset * 18 + j
                hr = 1 if (game_offset == 0 and j in (0, 1)) else 0
                ids.append((game_pk, batter_id, hr))
        for rank, (game_pk, batter_id, hr) in enumerate(ids):
            # Challenger B puts both true HR rows first. Reference A puts only
            # one true HR row first and one non-HR row second.
            p_b = 1.0 - rank / 1000.0
            if rank == 1:  # second true HR
                p_a = 0.100
            elif rank == 2:  # non-HR promoted to A's second slot
                p_a = 0.999
            else:
                p_a = p_b
            common = dict(
                game_pk=game_pk,
                batter_id=batter_id,
                game_date=day,
                year=year,
                hr_in_game=hr,
            )
            rows_a.append({**common, 'p_raw': p_a, 'p_cal': p_a})
            rows_b.append({**common, 'p_raw': p_b, 'p_cal': p_b})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def test_date_cluster_bootstrap_detects_known_positive_delta():
    a, b = synthetic_pair()
    a, b = boot.validate_pair(a, b)
    sel = boot.Selector('daily_top5pct', frac=0.05)
    ac = boot.daily_counts(a, sel)
    bc = boot.daily_counts(b, sel)
    result = boot.bootstrap_dates(ac, bc, reps=2000, seed=123)
    assert result['observed_a'] == pytest.approx(0.5)
    assert result['observed_b'] == pytest.approx(1.0)
    assert result['observed_delta_b_minus_a'] == pytest.approx(0.5)
    assert result['bootstrap_delta_b_minus_a']['ci95_low'] > 0
    assert result['prob_delta_gt_0'] == pytest.approx(1.0)


def test_game_cluster_sensitivity_detects_known_positive_delta():
    # Exercise the secondary path that resamples whole 18-batter game clusters
    # inside each slate, matching the game-resampling suggestion directly.
    a, b = synthetic_pair(n_days=20)
    a, b = boot.validate_pair(a, b)
    result = boot.bootstrap_games_within_days(
        a,
        b,
        boot.Selector('daily_top5pct', frac=0.05),
        reps=100,
        seed=789,
    )
    assert result is not None
    assert result['bootstrap_delta_b_minus_a']['median'] > 0
    assert result['prob_delta_gt_0'] > 0.95


def test_identical_models_have_zero_paired_delta():
    a, _ = synthetic_pair()
    a, b = boot.validate_pair(a.copy(), a.copy())
    sel = boot.Selector('daily_top4', top_n=4)
    result = boot.bootstrap_dates(
        boot.daily_counts(a, sel), boot.daily_counts(b, sel), reps=1500, seed=456
    )
    assert result['observed_delta_b_minus_a'] == pytest.approx(0.0)
    assert result['bootstrap_delta_b_minus_a']['ci95_low'] == pytest.approx(0.0)
    assert result['bootstrap_delta_b_minus_a']['ci95_high'] == pytest.approx(0.0)
    assert result['prob_delta_gt_0'] == pytest.approx(0.0)


def test_pair_validation_fails_closed_on_2025():
    a, b = synthetic_pair()
    a.loc[a.index[0], 'year'] = 2025
    with pytest.raises(RuntimeError, match='escaped 2023-2024'):
        boot.validate_pair(a, b)


def test_pair_validation_fails_closed_on_key_mismatch():
    a, b = synthetic_pair()
    b.loc[b.index[0], 'batter_id'] += 999999
    with pytest.raises(RuntimeError, match='identical batter-game keys'):
        boot.validate_pair(a, b)


def test_daily_selection_is_deterministic_under_ties():
    a, _ = synthetic_pair(n_days=1)
    a['p_raw'] = 0.2
    sel = boot.select_one_day(a, boot.Selector('daily_top4', top_n=4))
    expected = a.sort_values(['game_pk','batter_id']).head(4)
    assert list(zip(sel.game_pk, sel.batter_id)) == list(zip(expected.game_pk, expected.batter_id))
