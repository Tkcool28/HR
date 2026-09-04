"""Augment the frozen trusted v1.2 core matrix with leakage-safe Statcast QoC.

The 53-feature core matrix is preserved verbatim under ``v1.2_core_baseline``
before QoC is added.  The active trusted matrix then becomes core + QoC so the
same trainer/evaluation architecture can be used for a controlled ablation.

2025 is never read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
CORE = ROOT/'features/v1.2_trusted'
SNAP = ROOT/'features/v1.2_core_baseline'
BIP = ROOT/'data/raw/bip_all.parquet'
DELIVERED_BUILDER = ROOT/'features/v1.2/build_features.py'
HERE = Path(__file__).resolve().parent


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    matrix_path = CORE/'game_features.parquet'
    flist_path = CORE/'feature_list.json'
    summary_path = CORE/'_summary.json'
    if not matrix_path.exists() or not flist_path.exists():
        raise RuntimeError('trusted core matrix must exist before QoC augmentation')

    # Preserve an exact within-run core snapshot before modifying the active
    # trusted matrix.  This is intentionally copy-based rather than rebuild-
    # based so the ablation differs only by the added feature columns.
    if SNAP.exists():
        shutil.rmtree(SNAP)
    SNAP.mkdir(parents=True)
    shutil.copy2(matrix_path, SNAP/'game_features.parquet')
    shutil.copy2(flist_path, SNAP/'feature_list.json')
    if summary_path.exists():
        shutil.copy2(summary_path, SNAP/'_summary.json')

    targets = pd.read_parquet(matrix_path)
    core_features = json.loads(flist_path.read_text())
    if len(core_features) != 53:
        raise RuntimeError(f'expected frozen 53-feature core, got {len(core_features)}')
    if not targets.year.between(2015, 2024).all():
        raise RuntimeError('core matrix escaped 2015-2024')

    bip = pd.read_parquet(BIP)
    bip['game_date'] = pd.to_datetime(bip.game_date).dt.normalize()
    if not bip.game_date.dt.year.between(2015, 2024).all():
        raise RuntimeError('trusted regular-season BIP escaped 2015-2024')

    bf = import_path('delivered_build_v12_for_qoc', DELIVERED_BUILDER)
    qoc = import_path('trusted_qoc_features', HERE/'qoc_features_trusted.py')
    targets, qoc_features = qoc.compute_qoc_features(targets, bip, bf)

    overlap = sorted(set(core_features) & set(qoc_features))
    if overlap:
        raise RuntimeError(f'QoC feature names collide with core: {overlap}')
    if len(qoc_features) != 20:
        raise RuntimeError(f'expected 20 QoC features, got {len(qoc_features)}')

    active = core_features + qoc_features
    nonnumeric = [c for c in active if not pd.api.types.is_numeric_dtype(targets[c])]
    if nonnumeric:
        raise RuntimeError(f'nonnumeric active QoC matrix features: {nonnumeric}')
    arr = targets[active].to_numpy(dtype='float64', copy=False)
    if np.isinf(arr).any():
        raise RuntimeError('infinite value in active QoC matrix')

    # Extend the max-as-of provenance to include the newly added QoC columns.
    asof_cols = [c for c in targets.columns if c.endswith('_as_of')]
    if not asof_cols:
        raise RuntimeError('no as-of provenance columns')
    asof_frame = targets[asof_cols].apply(pd.to_datetime)
    targets['max_as_of_date'] = asof_frame.max(axis=1)
    gd = pd.to_datetime(targets.game_date)
    for c in asof_cols:
        d = pd.to_datetime(targets[c])
        m = d.notna()
        if not (d[m] < gd[m]).all():
            raise RuntimeError(f'QoC/core as-of violation: {c}')
    m = pd.to_datetime(targets.max_as_of_date).notna()
    if not (pd.to_datetime(targets.loc[m, 'max_as_of_date']) < gd[m]).all():
        raise RuntimeError('max_as_of_date violation after QoC augmentation')

    targets.to_parquet(matrix_path, index=False)
    flist_path.write_text(json.dumps(active, indent=2))
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({
        'n_rows': int(len(targets)),
        'n_games': int(targets.game_pk.nunique()),
        'n_features': int(len(active)),
        'n_core_features': int(len(core_features)),
        'n_qoc_features': int(len(qoc_features)),
        'qoc_feature_names': qoc_features,
        'feature_list_sha256': hashlib.sha256(flist_path.read_bytes()).hexdigest(),
        'ablation_parent': 'v1.2_core_baseline',
        'holdout_2025_read': False,
    })
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f'[augment_qoc] preserved {len(core_features)}-feature core snapshot at {SNAP}', flush=True)
    print(f'[augment_qoc] active matrix now {len(targets):,} rows x {len(active)} features ({len(qoc_features)} QoC)', flush=True)


if __name__ == '__main__':
    main()
