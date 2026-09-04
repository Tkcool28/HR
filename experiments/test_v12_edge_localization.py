from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v12_edge_localization as edge


def test_obvious_power_proxy_stays_long_horizon_batter_only():
    active = [
        'batter_hr_per_pa_season','batter_hr_per_pa_career',
        'batter_barrel_rate_season','batter_barrel_rate_career',
        'batter_xwoba_on_contact_season','batter_xwoba_on_contact_career',
        'batter_avg_ev_season','pitcher_hr_per_pa_season','park_hr_factor_3yr_prior',
        'batter_barrel_rate_30d',
    ]
    cols = edge._present_obvious(active)
    assert len(cols) == 7
    assert all(c.startswith('batter_') for c in cols)
    assert not any('30d' in c for c in cols)
    assert 'pitcher_hr_per_pa_season' not in cols
    assert 'park_hr_factor_3yr_prior' not in cols


def test_daily_rank_is_deterministic_under_score_ties():
    d = pd.DataFrame({
        'game_date': [pd.Timestamp('2023-04-01')]*4,
        'game_pk': [2,1,1,2],
        'batter_id': [20,20,10,10],
        'score': [0.5,0.5,0.5,0.5],
    })
    r = edge._assign_daily_rank(d,'score','rank')
    ranked = d.assign(rank=r).sort_values('rank')
    assert list(zip(ranked.game_pk,ranked.batter_id)) == [(1,10),(1,20),(2,10),(2,20)]


def test_paired_date_bootstrap_detects_differentiated_model_edge():
    rows=[]
    start=pd.Timestamp('2023-04-01')
    for i in range(120):
        day=start+pd.Timedelta(days=i)
        # Equal-sized top-four selectors. Shared picks contribute the same to
        # both selectors; model-only picks are constructed to be better than
        # obvious-only picks on every date.
        for j in range(2):
            rows.append({'game_date':day,'hr_in_game':j==0,'model_top4':True,'obvious_top4':True})
        rows.append({'game_date':day,'hr_in_game':1,'model_top4':True,'obvious_top4':False})
        rows.append({'game_date':day,'hr_in_game':0,'model_top4':True,'obvious_top4':False})
        rows.append({'game_date':day,'hr_in_game':0,'model_top4':False,'obvious_top4':True})
        rows.append({'game_date':day,'hr_in_game':0,'model_top4':False,'obvious_top4':True})
    f=pd.DataFrame(rows)
    masks={
        'model_top4':f.model_top4,
        'obvious_top4':f.obvious_top4,
        'shared_top4':f.model_top4 & f.obvious_top4,
        'model_only':f.model_top4 & ~f.obvious_top4,
        'obvious_only':f.obvious_top4 & ~f.model_top4,
    }
    out=edge._bootstrap_segment_rates(f,masks,reps=2000,seed=123)
    d=out['model_only_minus_obvious_only']
    assert d['ci95_low'] > 0
    assert d['prob_delta_gt_0'] == pytest.approx(1.0)
    top=out['model_top4_minus_obvious_top4']
    assert top['ci95_low'] > 0
