import numpy as np
import pandas as pd

from v12_yearly_context_migration import (
    YEARS,
    _adjacent_signs,
    _bootstrap_trend,
    _spearman_year,
    _slope,
    _top5_flag,
)


def synthetic_scored(trending: bool = True) -> pd.DataFrame:
    rows = []
    game_pk = 1
    for yi, year in enumerate(YEARS):
        # 60 dates/year, 40 hitters/date => exactly 2 top-5% selections.
        for di, day in enumerate(pd.date_range(f'{year}-04-01', periods=60, freq='D')):
            depth_rank = 5 + (yi * 3 if trending else 6)
            for batter in range(1, 41):
                model_rank = batter
                obvious_rank = batter
                # Model's second top-5 selection is a differentiated pick.
                if batter == 2:
                    obvious_rank = depth_rank
                elif batter == depth_rank:
                    obvious_rank = 2
                rows.append({
                    'game_pk': game_pk,
                    'batter_id': batter,
                    'game_date': day,
                    'year': year,
                    'model_rank': model_rank,
                    'obvious_rank': obvious_rank,
                    'slate_size': 40,
                    'full73_top5': model_rank <= 2,
                    'obvious_top5': obvious_rank <= 2,
                    'full73_only': model_rank <= 2 and obvious_rank > 2,
                    'obvious_rank_percentile': (obvious_rank - 1) / 39,
                    'hr_in_game': int(batter == 2 and di % 5 == 0),
                })
            game_pk += 1
    return pd.DataFrame(rows)


def test_top5_size():
    d = pd.DataFrame({
        'game_date': pd.to_datetime(['2024-08-01'] * 41),
        'model_rank': np.arange(1, 42),
    })
    assert int(_top5_flag(d, 'model_rank').sum()) == 3


def test_positive_depth_trend_bootstrap():
    d = synthetic_scored(True)
    out = _bootstrap_trend(d, reps=2000, seed=11, years=YEARS)
    assert out['depth_slope_bootstrap']['ci95_low'] > 0
    assert out['depth_slope_bootstrap']['prob_gt_0'] > 0.99
    # As full73-only moves farther away, overlap is constant in this fixture.
    assert abs(out['overlap_slope_bootstrap']['median']) < 1e-12


def test_flat_depth_trend():
    d = synthetic_scored(False)
    out = _bootstrap_trend(d, reps=1000, seed=12, years=YEARS)
    assert abs(out['depth_slope_bootstrap']['median']) < 1e-12


def test_slope_spearman_and_signs():
    vals = [1, 2, 3, 4, 5, 6]
    assert _slope(np.array(YEARS, float), np.array(vals, float)) > 0
    assert abs(_spearman_year(vals, YEARS) - 1.0) < 1e-12
    assert _adjacent_signs(vals) == ['+', '+', '+', '+', '+']


def test_2020_sensitivity_is_distinct_sequence():
    assert YEARS == [2019, 2020, 2021, 2022, 2023, 2024]
    assert [y for y in YEARS if y != 2020] == [2019, 2021, 2022, 2023, 2024]


def test_no_2025_in_frozen_years():
    assert 2025 not in YEARS
    assert max(YEARS) == 2024


if __name__ == '__main__':
    test_top5_size()
    test_positive_depth_trend_bootstrap()
    test_flat_depth_trend()
    test_slope_spearman_and_signs()
    test_2020_sensitivity_is_distinct_sequence()
    test_no_2025_in_frozen_years()
    print('yearly contextual migration synthetic controls PASS')
