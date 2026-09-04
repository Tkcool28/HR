"""Targeted joint-pruning candidates after broad/subfamily ablations.

Purpose: test whether individually mixed/noisy feature groups can be removed
together without retuning. The 73-feature aggressive model remains the frozen
reference. 2025 is never read.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/hr_model')
HERE = Path(__file__).resolve().parent
FEAT_DIR = ROOT/'features/v1.2_trusted'
FEAT = FEAT_DIR/'game_features.parquet'
FLIST = FEAT_DIR/'feature_list.json'
AGGRESSIVE = ROOT/'models/v1.2_trusted/metrics.json'
OUT = ROOT/'models/v1.2_pruning'


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    mod = importlib.util.module_from_spec(spec); assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def daily_order(g: pd.DataFrame) -> np.ndarray:
    return np.lexsort((g.batter_id.to_numpy(np.int64), g.game_pk.to_numpy(np.int64), -g.p_raw.to_numpy(float)))


def daily_stats(pred: pd.DataFrame) -> dict:
    out = {}
    for label, selector in [('top1',1),('top2',2),('top4',4),('top8',8),('top5pct',0.05),('top2pct',0.02)]:
        parts = []
        for _, g in pred.groupby('game_date', sort=True):
            o = daily_order(g)
            n = max(1, int(np.ceil(len(g)*selector))) if isinstance(selector,float) else min(selector,len(g))
            parts.append(g.iloc[o[:n]])
        s = pd.concat(parts, ignore_index=True)
        out[label] = {
            'n': int(len(s)),
            'hr_rate': float(s.hr_in_game.mean()),
            'mean_cal_probability': float(s.p_cal.mean()),
        }
    return out


def main() -> None:
    base = load_module('v12_feature_family_ablations.py','family_base')
    sub = load_module('v12_feature_subfamily_ablations.py','subfamily_defs')
    OUT.mkdir(parents=True, exist_ok=True)

    active = json.loads(FLIST.read_text())
    ref = json.loads(AGGRESSIVE.read_text())
    if len(active) != 73 or ref['design']['holdout_2025_read'] is not False:
        raise RuntimeError('reference/holdout contract mismatch')
    fam = sub.groups(active)

    usage = set(fam['pitcher_pitch_usage'])
    batter_xwoba_long = set(fam['batter_xwoba_long'])
    pitcher_vs = set(fam['pitcher_vs_pitch_hr_rates'])

    variants = {
        'full_73': set(),
        'prune_usage': usage,
        'prune_batter_xwoba_long': batter_xwoba_long,
        'prune_usage_plus_batter_xwoba_long': usage | batter_xwoba_long,
        'prune_usage_plus_pitcher_vs_pitch': usage | pitcher_vs,
        'prune_pitcher_vs_pitch_plus_batter_xwoba_long': pitcher_vs | batter_xwoba_long,
        'prune_usage_plus_pitcher_vs_pitch_plus_batter_xwoba_long': usage | pitcher_vs | batter_xwoba_long,
    }

    needed = list(dict.fromkeys(active + ['hr_in_game','game_pk','batter_id','year','game_date']))
    df = pd.read_parquet(FEAT, columns=needed)
    if not df.year.between(2015,2024).all():
        raise RuntimeError('pruning data escaped 2015-2024')

    results = {}
    predictions = []
    for name, removed in variants.items():
        cols = [c for c in active if c not in removed]
        r, _, p = base.evaluate_model(name, df, cols, dict(ref['best_params']), int(ref['best_round']))
        dates = df[df.year.isin([2023,2024])][['game_pk','batter_id','game_date']].copy()
        p = p.merge(dates, on=['game_pk','batter_id'], how='left', validate='one_to_one')
        p['game_date'] = pd.to_datetime(p.game_date).dt.normalize()
        d = daily_stats(p)
        r['removed_features'] = sorted(removed)
        r['daily'] = d
        results[name] = r
        predictions.append(p)
        print(
            f"[prune] {name}: features={len(cols)} calBrier={r['calibrated']['brier']:.6f} "
            f"AUC={r['calibrated']['auc']:.4f} ECE={r['ece']:.5f} "
            f"top4/day={d['top4']['hr_rate']:.4f} top8/day={d['top8']['hr_rate']:.4f}",
            flush=True,
        )

    full = results['full_73']
    for ours, expected, label in [
        (full['raw']['brier'], ref['xgb_raw_test']['brier'], 'raw brier'),
        (full['raw']['auc'], ref['xgb_raw_test']['auc'], 'raw auc'),
        (full['calibrated']['brier'], ref['xgb_calibrated_test']['brier'], 'cal brier'),
        (full['calibrated']['auc'], ref['xgb_calibrated_test']['auc'], 'cal auc'),
        (full['ece'], ref['test_ece'], 'ece'),
    ]:
        if abs(float(ours)-float(expected)) > 1e-7:
            raise RuntimeError(f'full-model mismatch {label}')

    rows=[]
    for name,r in results.items():
        row={
            'model':name,
            'n_features':r['n_features'],
            'n_removed':len(r['removed_features']),
            'cal_brier':r['calibrated']['brier'],
            'delta_cal_brier_vs_full':r['calibrated']['brier']-full['calibrated']['brier'],
            'cal_auc':r['calibrated']['auc'],
            'delta_cal_auc_vs_full':r['calibrated']['auc']-full['calibrated']['auc'],
            'cal_ap':r['calibrated']['ap'],
            'ece':r['ece'],
        }
        for k,v in r['daily'].items():
            row[f'daily_{k}_hr_rate']=v['hr_rate']
            row[f'daily_{k}_delta_vs_full']=v['hr_rate']-full['daily'][k]['hr_rate']
        rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'targeted_pruning_summary.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'development_predictions.parquet',index=False)
    (OUT/'targeted_pruning_results.json').write_text(json.dumps({
        'design':{
            'params_and_rounds_frozen':True,
            'development_assessment':'2023-2024',
            'daily_ranking_score':'raw_xgboost_probability',
            'sealed_final_holdout':'2025',
            'holdout_2025_read':False,
        },
        'results':results,
    },indent=2))

    idx=table.set_index('model')
    order=['full_73']+sorted([m for m in variants if m!='full_73'],key=lambda m:idx.loc[m,'daily_top4_hr_rate'],reverse=True)
    lines=['# Targeted joint-pruning candidates','', '**2025 was not read or evaluated.**','',
           '| Model | Features | Cal Brier | AUC | ECE | Top1/day | Top2/day | Top4/day | Top8/day | Daily top5% |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in order:
        r=idx.loc[m]
        lines.append(
            f"| {m} | {int(r['n_features'])} | {r['cal_brier']:.6f} | {r['cal_auc']:.4f} | {r['ece']:.5f} | "
            f"{100*r['daily_top1_hr_rate']:.2f}% | {100*r['daily_top2_hr_rate']:.2f}% | "
            f"{100*r['daily_top4_hr_rate']:.2f}% | {100*r['daily_top8_hr_rate']:.2f}% | {100*r['daily_top5pct_hr_rate']:.2f}% |"
        )
    (OUT/'targeted_pruning_report.md').write_text('\n'.join(lines)+'\n')
    print((OUT/'targeted_pruning_report.md').read_text(),flush=True)


if __name__=='__main__':
    main()
