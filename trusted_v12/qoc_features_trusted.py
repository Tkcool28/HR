"""Leakage-safe Statcast quality-of-contact features for trusted v1.2.

The feature surface is frozen before any extended-model 2023-2024 comparison.
All rollups use only BIP dates strictly earlier than each target game date.

Important semantics:
- ``estimated_woba_using_speedangle`` is a BIP/contact expectation.  We name
  those features ``xwoba_on_contact`` rather than claiming full hitter xwOBA,
  which would also incorporate non-contact outcomes such as BB/K/HBP.
- Barrel is reconstructed from MLB's published expanding EV/LA barrel zone,
  not the old v1.1 98+ mph / 26-30 degree-only approximation.
"""
from __future__ import annotations

from datetime import timedelta
import gc
import numpy as np
import pandas as pd


# Frozen priors carried over from the earlier v1.1 concept where sensible.
# These regularize short windows; denominators are metric-specific so missing
# xwOBA/EV/LA tracking cannot silently depress averages.
BARREL_PRIOR = 0.06
XWOBA_CONTACT_PRIOR = 0.32
HARD_HIT_PRIOR = 0.35
EV90_PRIOR = 0.30
FB_PRIOR = 0.25
SWEET_SPOT_PRIOR = 0.33
ISO_XBP_PRIOR = 0.40
DEFAULT_EV = 88.5
DEFAULT_LA = 12.0


def _barrel_mask(ev: np.ndarray, la: np.ndarray) -> np.ndarray:
    """Reconstruct MLB's published barrel zone from exit velo + launch angle.

    Published anchors:
      98 mph -> 26..30 degrees
      99 mph -> 25..31 degrees (published rounded anchor)
      100 mph -> 24..33 degrees
      116+ mph -> 8..50 degrees

    The continuous boundary below is the standard interpolation implied by
    those anchors: lower bound widens 1 degree per mph; upper widens 1.5
    degrees per mph, both capped at MLB's 8..50 terminal zone.
    """
    ev = np.asarray(ev, dtype='float64')
    la = np.asarray(la, dtype='float64')
    eligible = np.isfinite(ev) & np.isfinite(la) & (ev >= 98.0)
    delta = np.maximum(ev - 98.0, 0.0)
    low = np.maximum(26.0 - delta, 8.0)
    high = np.minimum(30.0 + 1.5 * delta, 50.0)
    return eligible & (la >= low) & (la <= high)


def materialize_bip_qoc(bip: pd.DataFrame) -> pd.DataFrame:
    required = [
        'game_pk','game_date','batter','pitcher','events','launch_speed',
        'launch_angle','estimated_woba_using_speedangle','bb_type'
    ]
    missing = [c for c in required if c not in bip.columns]
    if missing:
        raise RuntimeError(f'missing BIP QoC columns: {missing}')

    out = pd.DataFrame({
        'game_pk': pd.to_numeric(bip.game_pk, errors='raise').astype('int64'),
        'game_date': pd.to_datetime(bip.game_date).dt.normalize(),
        'batter': pd.to_numeric(bip.batter, errors='raise').astype('int64'),
        'pitcher': pd.to_numeric(bip.pitcher, errors='raise').astype('int64'),
        'launch_speed': pd.to_numeric(bip.launch_speed, errors='coerce').astype('float32'),
        'launch_angle': pd.to_numeric(bip.launch_angle, errors='coerce').astype('float32'),
        'xwoba_contact': pd.to_numeric(bip.estimated_woba_using_speedangle, errors='coerce').astype('float32'),
    })
    ev = out.launch_speed.to_numpy(dtype='float64', na_value=np.nan)
    la = out.launch_angle.to_numpy(dtype='float64', na_value=np.nan)
    xw = out.xwoba_contact.to_numpy(dtype='float64', na_value=np.nan)

    out['n_bbe'] = np.int8(1)
    out['n_ev'] = np.isfinite(ev).astype('int8')
    out['n_la'] = np.isfinite(la).astype('int8')
    out['n_ev_la'] = (np.isfinite(ev) & np.isfinite(la)).astype('int8')
    out['n_xwoba'] = np.isfinite(xw).astype('int8')
    out['is_barrel'] = _barrel_mask(ev, la).astype('int8')
    out['is_hard_hit'] = (np.isfinite(ev) & (ev >= 95.0)).astype('int8')
    out['is_ev90'] = (np.isfinite(ev) & (ev >= 90.0)).astype('int8')
    out['is_fb'] = (np.isfinite(la) & (la >= 20.0) & (la <= 40.0)).astype('int8')
    out['is_sweet_spot'] = (np.isfinite(la) & (la >= 8.0) & (la <= 32.0)).astype('int8')

    out['sum_xwoba'] = np.where(np.isfinite(xw), xw, 0.0).astype('float32')
    out['sum_ev'] = np.where(np.isfinite(ev), ev, 0.0).astype('float32')
    out['sum_la'] = np.where(np.isfinite(la), la, 0.0).astype('float32')

    evs = bip.events.astype('string')
    # Keep the old v1.1 extra-base-power proxy for controlled restoration.
    out['isoxbp'] = (
        evs.eq('home_run').astype('int16') * 4
        + evs.eq('triple').astype('int16') * 3
        + evs.eq('double').astype('int16') * 2
    ).astype('int16')

    return out


def _shrunk(num, den, prior: float, k: float) -> np.ndarray:
    num = np.asarray(num, dtype='float32')
    den = np.asarray(den, dtype='float32')
    return ((num + np.float32(prior * k)) / (den + np.float32(k))).astype('float32')


def _mean(sum_v, den, default: float) -> np.ndarray:
    sum_v = np.asarray(sum_v, dtype='float32')
    den = np.asarray(den, dtype='float32')
    return np.where(den > 0, sum_v / np.maximum(den, 1), np.float32(default)).astype('float32')


def _build_ect(bf, q: pd.DataFrame, entity: str):
    value_cols = [
        'n_bbe','n_ev','n_la','n_ev_la','n_xwoba','is_barrel','is_hard_hit',
        'is_ev90','is_fb','is_sweet_spot','sum_xwoba','sum_ev','sum_la','isoxbp'
    ]
    agg = q.groupby([entity,'game_date'], as_index=False, sort=False)[value_cols].sum()
    return bf.EntityCumTables(agg, entity_col=entity, date_col='game_date')


def _roll(ect, target_view: pd.DataFrame, specs: dict[str, tuple[str, object]]):
    sums: dict[str,np.ndarray] = {}
    asofs: dict[str,np.ndarray] = {}
    for key, (value_col, window) in specs.items():
        s, a = ect.rollup(target_view, value_col, window, None)
        sums[key] = s
        asofs[key] = a
    return sums, asofs


def compute_qoc_features(targets: pd.DataFrame, bip: pd.DataFrame, bf) -> tuple[pd.DataFrame, list[str]]:
    """Add the frozen trusted Statcast QoC tranche to target rows."""
    q = materialize_bip_qoc(bip)
    if not q.game_date.dt.year.between(2015, 2024).all():
        raise RuntimeError('QoC source escaped 2015-2024')

    print(
        '[qoc_trusted] source rows=%d xwoba_coverage=%.4f EV_coverage=%.4f LA_coverage=%.4f reconstructed_barrel_rate=%.4f'
        % (
            len(q), float(q.n_xwoba.mean()), float(q.n_ev.mean()),
            float(q.n_la.mean()), float(q.is_barrel.sum()/max(q.n_ev_la.sum(),1))
        ), flush=True
    )

    # ---------- batter ----------
    ect_b = _build_ect(bf, q, 'batter')
    tb = pd.DataFrame({'batter': targets.batter_id.to_numpy(), 'game_date': targets.game_date.to_numpy()})
    bspec = {
        'bbe_30': ('n_bbe', timedelta(days=30)),
        'bbe_season': ('n_bbe', 'season'),
        'bbe_career': ('n_bbe', None),
        'evla_30': ('n_ev_la', timedelta(days=30)),
        'evla_season': ('n_ev_la', 'season'),
        'evla_career': ('n_ev_la', None),
        'barrel_30': ('is_barrel', timedelta(days=30)),
        'barrel_season': ('is_barrel', 'season'),
        'barrel_career': ('is_barrel', None),
        'nxw_30': ('n_xwoba', timedelta(days=30)),
        'nxw_season': ('n_xwoba', 'season'),
        'nxw_career': ('n_xwoba', None),
        'xw_30': ('sum_xwoba', timedelta(days=30)),
        'xw_season': ('sum_xwoba', 'season'),
        'xw_career': ('sum_xwoba', None),
        'nev_30': ('n_ev', timedelta(days=30)),
        'nev_season': ('n_ev', 'season'),
        'ev_30': ('sum_ev', timedelta(days=30)),
        'ev_season': ('sum_ev', 'season'),
        'ev90_30': ('is_ev90', timedelta(days=30)),
        'hard_30': ('is_hard_hit', timedelta(days=30)),
        'nla_30': ('n_la', timedelta(days=30)),
        'la_30': ('sum_la', timedelta(days=30)),
        'fb_30': ('is_fb', timedelta(days=30)),
        'sweet_30': ('is_sweet_spot', timedelta(days=30)),
        'isoxbp_30': ('isoxbp', timedelta(days=30)),
    }
    bs, ba = _roll(ect_b, tb, bspec)

    targets['batter_barrel_rate_30d'] = _shrunk(bs['barrel_30'], bs['evla_30'], BARREL_PRIOR, 30)
    targets['batter_barrel_rate_season'] = _shrunk(bs['barrel_season'], bs['evla_season'], BARREL_PRIOR, 80)
    targets['batter_barrel_rate_career'] = _shrunk(bs['barrel_career'], bs['evla_career'], BARREL_PRIOR, 200)
    targets['batter_xwoba_on_contact_30d'] = _shrunk(bs['xw_30'], bs['nxw_30'], XWOBA_CONTACT_PRIOR, 30)
    targets['batter_xwoba_on_contact_season'] = _shrunk(bs['xw_season'], bs['nxw_season'], XWOBA_CONTACT_PRIOR, 80)
    targets['batter_xwoba_on_contact_career'] = _shrunk(bs['xw_career'], bs['nxw_career'], XWOBA_CONTACT_PRIOR, 200)
    targets['batter_avg_ev_30d'] = _mean(bs['ev_30'], bs['nev_30'], DEFAULT_EV)
    targets['batter_avg_ev_season'] = _mean(bs['ev_season'], bs['nev_season'], DEFAULT_EV)
    targets['batter_ev90_30d'] = _shrunk(bs['ev90_30'], bs['nev_30'], EV90_PRIOR, 30)
    targets['batter_avg_la_30d'] = _mean(bs['la_30'], bs['nla_30'], DEFAULT_LA)
    targets['batter_hard_hit_pct_30d'] = _shrunk(bs['hard_30'], bs['nev_30'], HARD_HIT_PRIOR, 30)
    targets['batter_fb_pct_30d'] = _shrunk(bs['fb_30'], bs['nla_30'], FB_PRIOR, 30)
    targets['batter_sweet_spot_pct_30d'] = _shrunk(bs['sweet_30'], bs['nla_30'], SWEET_SPOT_PRIOR, 30)
    targets['batter_iso_xbp_30d'] = _shrunk(bs['isoxbp_30'], bs['bbe_30'], ISO_XBP_PRIOR, 30)

    basof = {
        'batter_barrel_rate_30d': 'barrel_30',
        'batter_barrel_rate_season': 'barrel_season',
        'batter_barrel_rate_career': 'barrel_career',
        'batter_xwoba_on_contact_30d': 'xw_30',
        'batter_xwoba_on_contact_season': 'xw_season',
        'batter_xwoba_on_contact_career': 'xw_career',
        'batter_avg_ev_30d': 'ev_30',
        'batter_avg_ev_season': 'ev_season',
        'batter_ev90_30d': 'ev90_30',
        'batter_avg_la_30d': 'la_30',
        'batter_hard_hit_pct_30d': 'hard_30',
        'batter_fb_pct_30d': 'fb_30',
        'batter_sweet_spot_pct_30d': 'sweet_30',
        'batter_iso_xbp_30d': 'isoxbp_30',
    }
    for out_col, key in basof.items():
        targets[out_col + '_as_of'] = ba[key]

    # ---------- starting pitcher allowed ----------
    ect_p = _build_ect(bf, q, 'pitcher')
    tp = pd.DataFrame({'pitcher': targets.pitcher_id.to_numpy(), 'game_date': targets.game_date.to_numpy()})
    pspec = {
        'bbe_30': ('n_bbe', timedelta(days=30)),
        'evla_30': ('n_ev_la', timedelta(days=30)),
        'evla_season': ('n_ev_la', 'season'),
        'barrel_30': ('is_barrel', timedelta(days=30)),
        'barrel_season': ('is_barrel', 'season'),
        'nxw_30': ('n_xwoba', timedelta(days=30)),
        'xw_30': ('sum_xwoba', timedelta(days=30)),
        'nev_30': ('n_ev', timedelta(days=30)),
        'ev_30': ('sum_ev', timedelta(days=30)),
        'hard_30': ('is_hard_hit', timedelta(days=30)),
        'isoxbp_30': ('isoxbp', timedelta(days=30)),
    }
    ps, pa = _roll(ect_p, tp, pspec)

    targets['pitcher_barrel_rate_allowed_30d'] = _shrunk(ps['barrel_30'], ps['evla_30'], BARREL_PRIOR, 30)
    targets['pitcher_barrel_rate_allowed_season'] = _shrunk(ps['barrel_season'], ps['evla_season'], BARREL_PRIOR, 80)
    targets['pitcher_xwoba_on_contact_allowed_30d'] = _shrunk(ps['xw_30'], ps['nxw_30'], XWOBA_CONTACT_PRIOR, 30)
    targets['pitcher_hard_hit_pct_allowed_30d'] = _shrunk(ps['hard_30'], ps['nev_30'], HARD_HIT_PRIOR, 30)
    targets['pitcher_avg_ev_allowed_30d'] = _mean(ps['ev_30'], ps['nev_30'], DEFAULT_EV)
    targets['pitcher_iso_xbp_allowed_30d'] = _shrunk(ps['isoxbp_30'], ps['bbe_30'], ISO_XBP_PRIOR, 30)

    pasof = {
        'pitcher_barrel_rate_allowed_30d': 'barrel_30',
        'pitcher_barrel_rate_allowed_season': 'barrel_season',
        'pitcher_xwoba_on_contact_allowed_30d': 'xw_30',
        'pitcher_hard_hit_pct_allowed_30d': 'hard_30',
        'pitcher_avg_ev_allowed_30d': 'ev_30',
        'pitcher_iso_xbp_allowed_30d': 'isoxbp_30',
    }
    for out_col, key in pasof.items():
        targets[out_col + '_as_of'] = pa[key]

    active = list(basof) + list(pasof)

    # Strong sanity checks before these features can enter training.
    for c in active:
        vals = pd.to_numeric(targets[c], errors='coerce')
        if vals.isna().any() or not np.isfinite(vals.to_numpy()).all():
            raise RuntimeError(f'non-finite QoC feature: {c}')
        if vals.nunique(dropna=False) <= 1:
            raise RuntimeError(f'degenerate QoC feature: {c}')
    for c in [x for x in active if 'barrel_rate' in x or 'hard_hit_pct' in x or 'fb_pct' in x or 'sweet_spot_pct' in x or 'ev90' in x]:
        if not targets[c].between(0.0, 1.0).all():
            raise RuntimeError(f'rate outside [0,1]: {c}')
    for c in [x for x in active if 'xwoba_on_contact' in x]:
        if not targets[c].between(0.0, 2.0).all():
            raise RuntimeError(f'xwOBA-on-contact outside sane range: {c}')

    del ect_b, ect_p, bs, ba, ps, pa, q
    gc.collect()
    print(f'[qoc_trusted] added {len(active)} frozen active QoC features', flush=True)
    return targets, active
