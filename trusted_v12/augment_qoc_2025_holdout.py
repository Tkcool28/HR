"""Add the frozen 20 QoC features to the authorized 2015-2025 holdout matrix.

The workflow supplies a runtime copy of qoc_features_trusted.py whose only
semantic change is allowing source year 2025. All formulas, windows, priors,
and strict-before-game-date rollups remain identical.
"""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/workspace/hr_model')
CORE=ROOT/'features/v1.2_2025_holdout'
BIP=ROOT/'data/raw/bip_all.parquet'
DELIVERED=ROOT/'features/v1.2/build_features.py'
QOC_RUNTIME=ROOT/'qoc_features_2025_runtime.py'


def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    assert spec.loader; spec.loader.exec_module(mod); return mod


def main():
    matrix=CORE/'game_features.parquet'; flist=CORE/'feature_list.json'; summary=CORE/'_summary.json'
    targets=pd.read_parquet(matrix); core=json.loads(flist.read_text())
    if len(core)!=53: raise RuntimeError(f'expected 53 core, got {len(core)}')
    if not targets.year.between(2015,2025).all(): raise RuntimeError('holdout core escaped 2015-2025')
    bip=pd.read_parquet(BIP); bip['game_date']=pd.to_datetime(bip.game_date).dt.normalize()
    if not bip.game_date.dt.year.between(2015,2025).all(): raise RuntimeError('holdout BIP escaped 2015-2025')
    bf=imp('delivered_holdout_qoc',DELIVERED); qoc=imp('trusted_qoc_2025_runtime',QOC_RUNTIME)
    targets,qcols=qoc.compute_qoc_features(targets,bip,bf)
    if len(qcols)!=20: raise RuntimeError(f'expected 20 QoC, got {len(qcols)}')
    if set(core)&set(qcols): raise RuntimeError('QoC/core collision')
    active=core+qcols
    bad=[c for c in active if not pd.api.types.is_numeric_dtype(targets[c])]
    if bad: raise RuntimeError(f'nonnumeric active features: {bad}')
    if np.isinf(targets[active].to_numpy(dtype='float64',copy=False)).any(): raise RuntimeError('infinite active value')
    asof=[c for c in targets.columns if c.endswith('_as_of')]
    if not asof: raise RuntimeError('missing as-of provenance')
    af=targets[asof].apply(pd.to_datetime); targets['max_as_of_date']=af.max(axis=1)
    gd=pd.to_datetime(targets.game_date)
    for c in asof:
        d=pd.to_datetime(targets[c]); m=d.notna()
        if not (d[m] < gd[m]).all(): raise RuntimeError(f'as-of violation: {c}')
    m=pd.to_datetime(targets.max_as_of_date).notna()
    if not (pd.to_datetime(targets.loc[m,'max_as_of_date']) < gd[m]).all(): raise RuntimeError('max as-of violation')
    if set(targets.loc[targets.year==2025,'split'])!={'holdout'}: raise RuntimeError('2025 split changed')
    targets.to_parquet(matrix,index=False); flist.write_text(json.dumps(active,indent=2))
    s=json.loads(summary.read_text()); s.update({
        'n_features':len(active),'n_core_features':len(core),'n_qoc_features':len(qcols),
        'qoc_feature_names':qcols,'feature_list_sha256':hashlib.sha256(flist.read_bytes()).hexdigest(),
        'authorized_holdout_2025_materialized':True,
    }); summary.write_text(json.dumps(s,indent=2))
    print(f'[augment_qoc_holdout] active matrix {len(targets):,} rows x {len(active)} features')
    print(f'[augment_qoc_holdout] 2025 rows={int(targets.year.eq(2025).sum()):,}; strict pregame provenance PASS')

if __name__=='__main__': main()
