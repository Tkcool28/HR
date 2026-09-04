"""Synthetic pre-execution controls for the frozen 2025 one-shot.

No real 2025 data are read here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    assert spec.loader; spec.loader.exec_module(mod); return mod


ev=load('holdout_eval',ROOT/'experiments/v12_evaluate_2025_one_shot.py')
acq=load('holdout_acq',ROOT/'trusted_v12/acquire_2025_holdout.py')


def synthetic(n_dates=120,n_per_day=20):
    rows=[]
    for j,day in enumerate(pd.date_range('2025-03-20',periods=n_dates,freq='D')):
        for i in range(n_per_day):
            rows.append({
                'game_date':day,'game_pk':1_000_000+j*100+i//2,'batter_id':j*1000+i,
                'hr_in_game':int(i==0),
                # exact tie among top two verifies deterministic game/batter key behavior
                'model_score':1.0 if i<2 else float(n_per_day-i)/100,
                'obvious_score':1.0 if i in (2,3) else float(n_per_day-i)/1000,
            })
    d=pd.DataFrame(rows)
    d['model_rank']=ev.add_ranks(d,'model_score','model_rank')
    d['obvious_rank']=ev.add_ranks(d,'obvious_score','obvious_rank')
    return d


def test_daily_top5_exact_and_ties():
    d=synthetic()
    m=ev.pct_mask(d,'model_rank',.05)
    # 20 rows/day => ceil(1.0) = exactly one/day.
    assert int(m.sum())==120
    assert d.loc[m].groupby('game_date').size().eq(1).all()
    # tied model scores choose outcome-independent smallest game_pk/batter_id.
    assert d.loc[m,'batter_id'].mod(1000).eq(0).all()


def test_bootstrap_positive_and_zero_controls():
    d=synthetic()
    model=ev.pct_mask(d,'model_rank',.05)
    obvious=ev.pct_mask(d,'obvious_rank',.05)
    pos=ev.bootstrap_pair(d,obvious,model,10_000,20260904)
    assert pos['delta_ci95_low']>0
    assert pos['prob_delta_gt_0']==1.0
    zero=ev.bootstrap_pair(d,model,model,10_000,20260904)
    assert zero['delta_ci95_low']==0.0==zero['delta_ci95_high']
    assert zero['prob_delta_gt_0']==0.0


def test_bip_contract_frozen():
    expected={
        'single','double','triple','home_run','field_out','grounded_into_double_play',
        'force_out','fielders_choice','fielders_choice_out','sac_bunt','sac_fly',
        'sac_fly_double_play','sac_bunt_double_play','field_error','triple_play',
        'double_play','other_out',
    }
    assert acq.BIP_EVENTS==expected
    assert len(acq.BIP_EVENTS)==17


def test_acquisition_windows_cover_schedule_edges():
    first=pd.Timestamp('2025-03-18'); last=pd.Timestamp('2025-09-28')
    wins=list(acq._windows(first,last))
    assert wins
    assert wins[0][0] < first.date()
    assert wins[-1][1] > last.date()
    # Windows progress forward and intentionally overlap via halos.
    assert all(a<b for a,b in wins)


def main():
    test_daily_top5_exact_and_ties()
    test_bootstrap_positive_and_zero_controls()
    test_bip_contract_frozen()
    test_acquisition_windows_cover_schedule_edges()
    print('2025 one-shot synthetic controls PASS; no real 2025 data read')


if __name__=='__main__': main()
