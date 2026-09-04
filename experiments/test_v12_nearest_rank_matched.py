"""Synthetic controls for v12_nearest_rank_matched_test.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MOD_PATH = Path(__file__).with_name('v12_nearest_rank_matched_test.py')
spec = importlib.util.spec_from_file_location('matched', MOD_PATH)
matched = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(matched)


def make_positive_frame(n_days: int = 100) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp('2023-04-01')
    for i in range(n_days):
        day = start + pd.Timedelta(days=i)
        # One selected deep-disagreement candidate. Its nearest same-band
        # control is rank 18 and always misses. Extra controls prove matching
        # does not compare the selected hitter with the whole deep tail.
        rows.extend([
            dict(game_pk=10000+i*10+1, batter_id=20000+i*10+1, game_date=day, year=2023,
                 hr_in_game=1, model_top4=True, obvious_rank=17),
            dict(game_pk=10000+i*10+2, batter_id=20000+i*10+2, game_date=day, year=2023,
                 hr_in_game=0, model_top4=False, obvious_rank=18),
            dict(game_pk=10000+i*10+3, batter_id=20000+i*10+3, game_date=day, year=2023,
                 hr_in_game=1, model_top4=False, obvious_rank=70),
        ])
        # Populate the other predeclared bands so run_scope can test all three.
        rows.extend([
            dict(game_pk=10000+i*10+4, batter_id=20000+i*10+4, game_date=day, year=2023,
                 hr_in_game=1, model_top4=True, obvious_rank=5),
            dict(game_pk=10000+i*10+5, batter_id=20000+i*10+5, game_date=day, year=2023,
                 hr_in_game=0, model_top4=False, obvious_rank=6),
            dict(game_pk=10000+i*10+6, batter_id=20000+i*10+6, game_date=day, year=2023,
                 hr_in_game=1, model_top4=True, obvious_rank=9),
            dict(game_pk=10000+i*10+7, batter_id=20000+i*10+7, game_date=day, year=2023,
                 hr_in_game=0, model_top4=False, obvious_rank=10),
        ])
    return pd.DataFrame(rows)


def main() -> None:
    f = matched.validate(make_positive_frame(), {2023})

    p, diag = matched.match_band(f, f.obvious_rank.ge(17), 'obvious_rank_17_plus')
    assert len(p) == 100
    assert diag['unmatched_selected'] == 0
    assert diag['rank_gap_max'] == 1
    assert (p.control_obvious_rank == 18).all()
    assert not p.duplicated(['game_date','control_game_pk','control_batter_id']).any()
    print('PASS deterministic nearest-rank matching')

    b = matched.bootstrap_pairs(p, reps=2000, seed=123)
    d = b['paired_lift']
    dec = b['precommitted_decision']
    assert d['observed'] == 1.0
    assert d['ci95_low'] > 0
    assert dec['magnitude_floor_met'] is True
    assert dec['survives_operational_consideration'] is True
    assert dec['strong_signal_grade'] is True
    print('PASS positive-control bootstrap and magnitude rule')

    e = matched.exact_mcnemar_one_sided(p)
    assert e['selected_only_hr_pairs'] == 100
    assert e['control_only_hr_pairs'] == 0
    assert e['p_one_sided_selected_gt_control'] < 1e-20
    print('PASS exact matched-binary positive control')

    r, all_pairs = matched.run_scope(f, reps=2000, seed=456, include_holm=True)
    assert set(r) == set(matched.STRATA)
    assert len(all_pairs) == 300
    for item in r.values():
        assert item['exact_matched_binary_test']['holm_adjusted_p_across_3_strata'] < 1e-20
    print('PASS three-band Holm control')

    identical = p.copy()
    identical['control_hr'] = identical['selected_hr']
    bi = matched.bootstrap_pairs(identical, reps=2000, seed=789)
    assert bi['paired_lift']['observed'] == 0.0
    assert bi['precommitted_decision']['survives_operational_consideration'] is False
    assert bi['precommitted_decision']['magnitude_floor_met'] is False
    print('PASS identical-pair negative control')

    bad = f.copy()
    bad.loc[bad.index[0], 'year'] = 2025
    try:
        matched.validate(bad, {2023, 2024})
    except RuntimeError:
        pass
    else:
        raise AssertionError('2025 validation must fail closed')
    print('PASS 2025 fail-closed control')
    print('ALL MATCHED-PAIR CONTROLS PASS')


if __name__ == '__main__':
    main()
