from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
CORE = ROOT/'features/v1.2_core_baseline'
EXT = ROOT/'features/v1.2_trusted'

EXPECTED_QOC = {
    'batter_barrel_rate_30d',
    'batter_barrel_rate_season',
    'batter_barrel_rate_career',
    'batter_xwoba_on_contact_30d',
    'batter_xwoba_on_contact_season',
    'batter_xwoba_on_contact_career',
    'batter_avg_ev_30d',
    'batter_avg_ev_season',
    'batter_ev90_30d',
    'batter_avg_la_30d',
    'batter_hard_hit_pct_30d',
    'batter_fb_pct_30d',
    'batter_sweet_spot_pct_30d',
    'batter_iso_xbp_30d',
    'pitcher_barrel_rate_allowed_30d',
    'pitcher_barrel_rate_allowed_season',
    'pitcher_xwoba_on_contact_allowed_30d',
    'pitcher_hard_hit_pct_allowed_30d',
    'pitcher_avg_ev_allowed_30d',
    'pitcher_iso_xbp_allowed_30d',
}


def lists():
    core = json.loads((CORE/'feature_list.json').read_text())
    ext = json.loads((EXT/'feature_list.json').read_text())
    return core, ext


def test_qoc_is_controlled_20_feature_addition():
    core, ext = lists()
    assert len(core) == 53
    assert len(ext) == 73
    assert ext[:53] == core
    assert set(ext[53:]) == EXPECTED_QOC
    assert not (set(core) & EXPECTED_QOC)


def test_core_matrix_is_unchanged_by_qoc_augmentation():
    core_cols, ext_cols = lists()
    ids = ['game_pk','batter_id','pitcher_id','game_date','hr_in_game']
    a = pd.read_parquet(CORE/'game_features.parquet', columns=ids + core_cols)
    b = pd.read_parquet(EXT/'game_features.parquet', columns=ids + core_cols)
    assert len(a) == len(b)
    pd.testing.assert_frame_equal(a, b, check_dtype=True, check_exact=True)


def test_qoc_features_are_numeric_finite_and_nondegenerate():
    _, ext = lists()
    qoc = ext[53:]
    f = pd.read_parquet(EXT/'game_features.parquet', columns=qoc)
    assert all(pd.api.types.is_numeric_dtype(f[c]) for c in qoc)
    arr = f.to_numpy(dtype='float64')
    assert np.isfinite(arr).all()
    assert all(f[c].nunique(dropna=False) > 1 for c in qoc)


def test_qoc_rate_ranges_and_contact_ranges():
    _, ext = lists()
    qoc = ext[53:]
    f = pd.read_parquet(EXT/'game_features.parquet', columns=qoc)
    rate_cols = [c for c in qoc if any(x in c for x in ['barrel_rate','hard_hit_pct','fb_pct','sweet_spot_pct','ev90'])]
    for c in rate_cols:
        assert f[c].between(0.0, 1.0).all(), c
    for c in [c for c in qoc if 'xwoba_on_contact' in c]:
        assert f[c].between(0.0, 2.0).all(), c
    # Very soft contact and bunts can legitimately be below 40 mph, especially
    # when a 30-day window has only one or two tracked BBE.  Reject impossible
    # or clearly corrupt values without excluding valid Statcast observations.
    for c in [c for c in qoc if 'avg_ev' in c]:
        assert f[c].between(0.0, 125.0).all(), c
    for c in [c for c in qoc if 'avg_la' in c]:
        assert f[c].between(-90.0, 90.0).all(), c


def test_all_qoc_asof_dates_are_strictly_pregame():
    f = pd.read_parquet(EXT/'game_features.parquet')
    qoc_asof = [c for c in f.columns if c.endswith('_as_of') and any(name in c for name in EXPECTED_QOC)]
    assert len(qoc_asof) == len(EXPECTED_QOC)
    gd = pd.to_datetime(f.game_date)
    for c in qoc_asof:
        d = pd.to_datetime(f[c])
        m = d.notna()
        assert (d[m] < gd[m]).all(), c
    m = pd.to_datetime(f.max_as_of_date).notna()
    assert (pd.to_datetime(f.loc[m,'max_as_of_date']) < gd[m]).all()


def test_qoc_summary_declares_ablation_and_no_2025():
    s = json.loads((EXT/'_summary.json').read_text())
    assert s['n_core_features'] == 53
    assert s['n_qoc_features'] == 20
    assert s['n_features'] == 73
    assert set(s['qoc_feature_names']) == EXPECTED_QOC
    assert s['ablation_parent'] == 'v1.2_core_baseline'
    assert s['holdout_2025_read'] is False
