"""Fine-grained frozen-parameter feature ablations for trusted v1.2.

Second-stage destruction test after the broad family ablations.  The full
73-feature aggressive model remains the reference.  Hyperparameters and the
194 boosting rounds are frozen; no ablation is retuned.  2025 is never read.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/workspace/hr_model')
HERE = Path(__file__).resolve().parent
FEAT_DIR = ROOT/'features/v1.2_trusted'
FEAT = FEAT_DIR/'game_features.parquet'
FLIST = FEAT_DIR/'feature_list.json'
AGGRESSIVE = ROOT/'models/v1.2_trusted/metrics.json'
OUT = ROOT/'models/v1.2_subablations'


def import_base():
    path = HERE/'v12_feature_family_ablations.py'
    spec = importlib.util.spec_from_file_location('family_ablation_base', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def groups(active: list[str]) -> dict[str, list[str]]:
    def exact(names):
        cols = [c for c in names if c in active]
        if len(cols) != len(names):
            missing = sorted(set(names)-set(cols))
            raise RuntimeError(f'missing expected active features: {missing}')
        return cols

    batter_vs_pitch = sorted([c for c in active if c.startswith('batter_hr_per_pa_vs_')])
    pitcher_vs_pitch = sorted([c for c in active if c.startswith('pitcher_hr_per_pa_vs_')])
    pitch_usage = sorted([c for c in active if c.startswith('pitcher_usage_')])
    if not (len(batter_vs_pitch) == len(pitcher_vs_pitch) == len(pitch_usage) == 9):
        raise RuntimeError(
            f'pitch subfamily cardinality mismatch: batter={len(batter_vs_pitch)} '
            f'pitcher={len(pitcher_vs_pitch)} usage={len(pitch_usage)}'
        )

    out = {
        # Pitch-matchup block decomposition.
        'batter_vs_pitch_hr_rates': batter_vs_pitch,
        'pitcher_vs_pitch_hr_rates': pitcher_vs_pitch,
        'pitcher_pitch_usage': pitch_usage,
        'top_pitch_pair': exact([
            'batter_strength_on_pitcher_top_pitch',
            'top_pitch_total_pitches_30d',
        ]),

        # Barrel decomposition.
        'batter_barrel_30d': exact(['batter_barrel_rate_30d']),
        'batter_barrel_long': exact(['batter_barrel_rate_season','batter_barrel_rate_career']),
        'pitcher_barrel_30d': exact(['pitcher_barrel_rate_allowed_30d']),
        'pitcher_barrel_season': exact(['pitcher_barrel_rate_allowed_season']),

        # xwOBA-on-contact decomposition.
        'batter_xwoba_30d': exact(['batter_xwoba_on_contact_30d']),
        'batter_xwoba_long': exact(['batter_xwoba_on_contact_season','batter_xwoba_on_contact_career']),
        'pitcher_xwoba_30d': exact(['pitcher_xwoba_on_contact_allowed_30d']),

        # Other contact-quality concepts.
        'batter_contact_shape_30d': exact([
            'batter_avg_ev_30d','batter_ev90_30d','batter_avg_la_30d',
            'batter_hard_hit_pct_30d','batter_fb_pct_30d',
            'batter_sweet_spot_pct_30d','batter_iso_xbp_30d',
        ]),
        'batter_avg_ev_season': exact(['batter_avg_ev_season']),
        'pitcher_contact_shape_30d': exact([
            'pitcher_hard_hit_pct_allowed_30d','pitcher_avg_ev_allowed_30d',
            'pitcher_iso_xbp_allowed_30d',
        ]),
    }

    for name, cols in out.items():
        if not cols or len(cols) != len(set(cols)):
            raise RuntimeError(f'invalid subfamily {name}: {cols}')
    return out


def main() -> None:
    base = import_base()
    OUT.mkdir(parents=True, exist_ok=True)

    active = json.loads(FLIST.read_text())
    aggressive = json.loads(AGGRESSIVE.read_text())
    if len(active) != 73 or aggressive['n_features'] != 73:
        raise RuntimeError('expected frozen 73-feature reference')
    if aggressive['design']['holdout_2025_read'] is not False:
        raise RuntimeError('sealed 2025 contract violated')

    best = dict(aggressive['best_params'])
    rounds = int(aggressive['best_round'])
    fam = groups(active)
    (OUT/'subfamilies.json').write_text(json.dumps(fam, indent=2))

    needed = list(dict.fromkeys(active + ['hr_in_game','game_pk','batter_id','year']))
    df = pd.read_parquet(FEAT, columns=needed)
    if not df.year.between(2015, 2024).all():
        raise RuntimeError('subablation data escaped 2015-2024')

    experiments = [('full_73', [])] + [(f'no_{name}', cols) for name, cols in fam.items()]
    results = {}
    selections = {}
    preds = []
    for name, removed in experiments:
        removed_set = set(removed)
        cols = [c for c in active if c not in removed_set]
        print(f'[subablation] {name}: {len(cols)} features; removed={len(removed)}', flush=True)
        r, s, p = base.evaluate_model(name, df, cols, best, rounds)
        r['removed_features'] = removed
        results[name] = r
        selections[name] = s
        preds.append(p)
        print(
            f"[subablation] {name}: Brier={r['calibrated']['brier']:.6f} "
            f"AUC={r['calibrated']['auc']:.4f} ECE={r['ece']:.5f} "
            f"raw-top5={r['raw_rank_buckets']['top5pct']['observed_hr_rate']:.4f}",
            flush=True,
        )

    full = results['full_73']
    # Full model must reproduce the frozen reference proper metrics.
    checks = [
        (full['raw']['brier'], aggressive['xgb_raw_test']['brier'], 'raw brier'),
        (full['raw']['auc'], aggressive['xgb_raw_test']['auc'], 'raw auc'),
        (full['calibrated']['brier'], aggressive['xgb_calibrated_test']['brier'], 'cal brier'),
        (full['calibrated']['auc'], aggressive['xgb_calibrated_test']['auc'], 'cal auc'),
        (full['ece'], aggressive['test_ece'], 'ece'),
    ]
    for ours, ref, label in checks:
        if abs(float(ours)-float(ref)) > 1e-7:
            raise RuntimeError(f'full reference mismatch {label}: {ours} vs {ref}')

    # Pooled raw ranking overlap diagnostics against full model.
    for name, r in results.items():
        overlap = {}
        if name != 'full_73':
            for key, full_set in selections['full_73'].items():
                cur = selections[name][key]
                inter = len(full_set & cur)
                overlap[key] = {
                    'retention_vs_full': inter/len(full_set),
                    'jaccard_vs_full': inter/len(full_set | cur),
                }
        r['ranking_overlap_vs_full'] = overlap

    rows = []
    for name, r in results.items():
        row = {
            'model': name,
            'n_features': r['n_features'],
            'n_removed': len(r['removed_features']),
            'cal_brier': r['calibrated']['brier'],
            'delta_cal_brier_vs_full': r['calibrated']['brier'] - full['calibrated']['brier'],
            'cal_auc': r['calibrated']['auc'],
            'delta_cal_auc_vs_full': r['calibrated']['auc'] - full['calibrated']['auc'],
            'cal_ap': r['calibrated']['ap'],
            'ece': r['ece'],
        }
        for key, stat in r['raw_rank_buckets'].items():
            row[f'{key}_hr_rate'] = stat['observed_hr_rate']
            row[f'{key}_delta_vs_full'] = stat['observed_hr_rate'] - full['raw_rank_buckets'][key]['observed_hr_rate']
            if name != 'full_73':
                row[f'{key}_retention_vs_full'] = r['ranking_overlap_vs_full'][key]['retention_vs_full']
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT/'subablation_summary.csv', index=False)
    pd.concat(preds, ignore_index=True).to_parquet(OUT/'development_predictions.parquet', index=False)
    (OUT/'subablation_results.json').write_text(json.dumps({
        'design': {
            'reference': 'full_73 aggressive model',
            'params_and_rounds_frozen': True,
            'train': '2015-2021',
            'calibration': '2022',
            'development_assessment': '2023-2024',
            'ranking_score': 'raw_xgboost_probability',
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'subfamilies': fam,
        'results': results,
    }, indent=2))


if __name__ == '__main__':
    main()
