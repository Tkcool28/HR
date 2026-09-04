from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v12_disagreement_strata_test as ds


def synthetic_disagreement(n_days: int = 120) -> pd.DataFrame:
    rows=[]
    start=pd.Timestamp('2023-04-01')
    for d in range(n_days):
        day=start+pd.Timedelta(days=d)
        game_pk=800000+d
        for rank in range(1,25):
            model_top4=rank in (1,5,9,17)
            obvious_top4=rank<=4
            # Construct a known positive contextual selection effect in every
            # disagreement stratum while leaving same-band controls at zero.
            hr=int(rank in (5,9,17))
            rows.append({
                'game_pk':game_pk,
                'batter_id':900000+d*100+rank,
                'game_date':day,
                'year':2023,
                'hr_in_game':hr,
                'model_top4':model_top4,
                'obvious_top4':obvious_top4,
                'obvious_rank':rank,
                'park_score':rank/24,
                'recent_batter_form_score':rank/24,
                'pitcher_vulnerability_score':rank/24,
                'pitch_matchup_score':rank/24,
            })
    return pd.DataFrame(rows)


def test_validate_rejects_2025():
    f=synthetic_disagreement()
    f.loc[f.index[0],'year']=2025
    try:
        ds.validate(f)
        assert False, 'expected validation failure'
    except RuntimeError as exc:
        assert 'escaped 2023-2024' in str(exc)


def test_all_three_predeclared_strata_detect_known_positive_selection():
    f=ds.validate(synthetic_disagreement())
    out=ds.run_scope(f,reps=1500,perm_reps=1500,seed=123,run_permutation=True)
    assert set(out)==set(ds.STRATA)
    for name,item in out.items():
        b=item['date_cluster_bootstrap']['delta_selected_minus_control']
        assert b['observed'] > 0
        assert b['ci95_low'] > 0
        p=item['within_date_randomization']
        assert p['p_one_sided_selected_gt_control'] < 0.01
        assert p['holm_adjusted_p_across_3_strata'] < 0.05


def test_holm_adjustment_is_monotone_and_bounded():
    x=ds.holm_adjust({'a':0.01,'b':0.02,'c':0.2})
    assert x['a'] == 0.03
    assert x['b'] == 0.04
    assert x['c'] == 0.2
