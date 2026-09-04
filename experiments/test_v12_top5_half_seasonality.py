import numpy as np
import pandas as pd

from experiments.v12_top5_half_seasonality import (
    phase_from_date,
    deterministic_top5_flags,
    daily_selected_counts,
    phase_bootstrap,
    interaction_bootstrap,
    validate_pair,
)


def synthetic_pair():
    rows_o = []
    rows_f = []
    game_pk = 1000
    # 2023 and 2024, 60 slate dates per half per year = 240 total dates.
    # 20 hitters/date => exactly one top-5% selection per model.
    for year in [2023, 2024]:
        dates = list(pd.date_range(f"{year}-04-01", periods=60, freq="D"))
        dates += list(pd.date_range(f"{year}-07-01", periods=60, freq="D"))
        for di, day in enumerate(dates):
            second = day.month >= 7
            for batter in range(1, 21):
                # Obvious always selects batter 1.
                score_o = 2.0 if batter == 1 else 1.0 - batter / 100.0
                # Full73: same selection first half; selects batter 2 second half.
                score_f = 2.0 if (batter == (2 if second else 1)) else 1.0 - batter / 100.0
                # First half: selected batter 1 HR on 1/5 dates for both.
                # Second half: obvious batter 1 HR on 1/10 dates; full batter 2 HR on 3/10 dates.
                if not second:
                    hr = int(batter == 1 and di % 5 == 0)
                else:
                    local = di - 60
                    hr = int((batter == 1 and local % 10 == 0) or
                             (batter == 2 and local % 10 in {0, 1, 2}))
                common = {
                    "game_pk": game_pk,
                    "batter_id": batter,
                    "game_date": day,
                    "year": year,
                    "hr_in_game": hr,
                }
                rows_o.append({**common, "p_raw": score_o})
                rows_f.append({**common, "p_raw": score_f})
            game_pk += 1
    return pd.DataFrame(rows_o), pd.DataFrame(rows_f)


def test_phase_cut():
    x = pd.Series(pd.to_datetime(["2024-06-30", "2024-07-01"]))
    assert phase_from_date(x).tolist() == ["FIRST_HALF", "SECOND_HALF"]


def test_deterministic_top5_exact_size():
    d = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-08-01"] * 21),
        "game_pk": [1] * 21,
        "batter_id": np.arange(1, 22),
        "p_raw": np.linspace(0, 1, 21),
    })
    flags = deterministic_top5_flags(d)
    assert flags.sum() == 2  # ceil(21 * .05)


def test_positive_second_half_interaction():
    obvious, full73 = synthetic_pair()
    obvious, full73 = validate_pair(obvious, full73)
    obvious["phase"] = phase_from_date(obvious.game_date)
    full73["phase"] = phase_from_date(full73.game_date)
    od = daily_selected_counts(obvious)
    fd = daily_selected_counts(full73)

    first = phase_bootstrap(od, fd, "FIRST_HALF", reps=2000, seed=11)
    second = phase_bootstrap(od, fd, "SECOND_HALF", reps=2000, seed=12)
    inter = interaction_bootstrap(od, fd, reps=2000, seed=13)

    assert abs(first["observed_lift_full73_minus_obvious"]) < 1e-12
    assert second["observed_lift_full73_minus_obvious"] > 0.15
    assert inter["observed_interaction"] > 0.15
    assert inter["bootstrap_interaction"]["ci95_low"] > 0


def test_identical_models_zero_interaction():
    obvious, _ = synthetic_pair()
    obvious, same = validate_pair(obvious.copy(), obvious.copy())
    obvious["phase"] = phase_from_date(obvious.game_date)
    same["phase"] = phase_from_date(same.game_date)
    od = daily_selected_counts(obvious)
    sd = daily_selected_counts(same)
    inter = interaction_bootstrap(od, sd, reps=1000, seed=21)
    assert abs(inter["observed_interaction"]) < 1e-12
    assert abs(inter["bootstrap_interaction"]["ci95_low"]) < 1e-12
    assert abs(inter["bootstrap_interaction"]["ci95_high"]) < 1e-12


def test_2025_fails_closed():
    obvious, full73 = synthetic_pair()
    obvious.loc[0, "year"] = 2025
    try:
        validate_pair(obvious, full73)
    except RuntimeError:
        pass
    else:
        raise AssertionError("2025 must fail closed")


if __name__ == "__main__":
    test_phase_cut()
    test_deterministic_top5_exact_size()
    test_positive_second_half_interaction()
    test_identical_models_zero_interaction()
    test_2025_fails_closed()
    print("top5 half-seasonality synthetic controls PASS")
