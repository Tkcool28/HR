from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
CUR = ROOT/'data/curated'
FEAT = ROOT/'features/v1.2_trusted'
MODELS = ROOT/'models/v1.2_trusted'


def test_context_regular_only_and_unique():
    c = pd.read_parquet(CUR/'game_context_v12_trusted.parquet')
    assert set(c.game_type.unique()) == {'R'}
    assert c.game_pk.is_unique
    assert c.season.between(2015, 2024).all()
    assert c.venue_id.notna().all()


def test_known_stadium_transitions_are_distinct():
    c = pd.read_parquet(CUR/'game_context_v12_trusted.parquet')
    atl16 = set(c[(c.home_team_name == 'Atlanta Braves') & (c.season == 2016)].venue_id)
    atl17 = set(c[(c.home_team_name == 'Atlanta Braves') & (c.season == 2017)].venue_id)
    tex19 = set(c[(c.home_team_name == 'Texas Rangers') & (c.season == 2019)].venue_id)
    tex20 = set(c[(c.home_team_name == 'Texas Rangers') & (c.season == 2020)].venue_id)
    assert atl16 and atl17 and atl16 != atl17
    assert tex19 and tex20 and tex19 != tex20
    tor21 = c[(c.home_team_name == 'Toronto Blue Jays') & (c.season == 2021)]
    assert tor21.venue_id.nunique() >= 2


def test_pa_and_pitch_level_subset_regular_context():
    cids = set(pd.read_parquet(CUR/'game_context_v12_trusted.parquet', columns=['game_pk']).game_pk.astype(int))
    pa = pd.read_parquet(CUR/'pa_v12_trusted.parquet', columns=['game_pk','at_bat_number','year'])
    pl = pd.read_parquet(CUR/'pitch_level_v12_trusted.parquet', columns=['game_pk','year'])
    assert not (set(pa.game_pk.astype(int)) - cids)
    assert not (set(pl.game_pk.astype(int)) - cids)
    assert pa.year.max() <= 2024 and pl.year.max() <= 2024
    assert not pa.duplicated(['game_pk','at_bat_number']).any()


def test_feature_target_universe_exact_starting_nine():
    f = pd.read_parquet(FEAT/'game_features.parquet', columns=['game_pk','batter_id','batter_side','lineup_slot','year','split'])
    assert not f.duplicated(['game_pk','batter_id']).any()
    assert f.groupby('game_pk').size().eq(18).all()
    assert f.groupby(['game_pk','batter_side']).size().eq(9).all()
    slots = f.groupby(['game_pk','batter_side']).lineup_slot.apply(lambda s: tuple(sorted(s.astype(int))))
    assert slots.map(lambda x: x == tuple(range(1,10))).all()
    assert set(f.loc[f.year <= 2022, 'split']) == {'train'}
    assert set(f.loc[f.year >= 2023, 'split']) == {'val'}


def test_active_feature_contract_numeric_and_finite_policy():
    cols = json.loads((FEAT/'feature_list.json').read_text())
    assert 'pitcher_top_pitch' not in cols
    f = pd.read_parquet(FEAT/'game_features.parquet', columns=cols)
    assert all(pd.api.types.is_numeric_dtype(f[c]) for c in cols)
    assert not np.isinf(f.to_numpy(dtype=float, na_value=np.nan)).any()
    assert all(f[c].nunique(dropna=False) > 1 for c in cols)


def test_all_asof_dates_strictly_before_game():
    f = pd.read_parquet(FEAT/'game_features.parquet')
    asof = [c for c in f.columns if c.endswith('_as_of')]
    assert asof
    gd = pd.to_datetime(f.game_date)
    for c in asof:
        d = pd.to_datetime(f[c])
        m = d.notna()
        assert (d[m] < gd[m]).all(), c
    m = pd.to_datetime(f.max_as_of_date).notna()
    assert (pd.to_datetime(f.loc[m,'max_as_of_date']) < gd[m]).all()


def test_park_sources_prior_only():
    pf = pd.read_parquet(CUR/'park_factors_v12_trusted.parquet')
    for r in pf.itertuples(index=False):
        src = [] if not r.source_years_used else [int(x) for x in str(r.source_years_used).split(',') if x]
        assert all(r.year - 3 <= y < r.year for y in src)


def test_training_artifacts_and_independent_design():
    m = json.loads((MODELS/'metrics.json').read_text())
    assert m['design']['holdout_2025_read'] is False
    assert m['design']['tune_selection'] == '2021'
    assert m['design']['calibration_fit'] == '2022'
    assert m['design']['independent_assessment'] == '2023-2024'
    for key in ['lr_test','xgb_raw_test','xgb_calibrated_test']:
        assert 0 < m[key]['brier'] < 0.25
        assert 0.45 < m[key]['auc'] < 0.90
    assert (MODELS/'xgb_production.joblib').exists()
    assert (MODELS/'isotonic_production.joblib').exists()
