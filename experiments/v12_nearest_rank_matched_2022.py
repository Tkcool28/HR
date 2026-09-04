"""Replay the frozen nearest-rank disagreement test on 2022.

This imports the exact matching/inference implementation used for the 2023-24
matched follow-up and applies it to the already-defined 5-8, 9-16, and 17+
obvious-power rank bands. No cutoff or decision threshold is changed.

2022 is a freshness/replication check, not a pristine final holdout. The input
must come from raw full73 scores trained only through 2021. 2025 is rejected.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

MOD_PATH = Path(__file__).with_name('v12_nearest_rank_matched_test.py')
spec = importlib.util.spec_from_file_location('matched_core', MOD_PATH)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-pairs', required=True)
    ap.add_argument('--reps', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20260904)
    args = ap.parse_args()

    f = core.validate(pd.read_parquet(args.input), {2022})
    if set(map(int, f.year.unique())) != {2022}:
        raise RuntimeError('freshness input must contain 2022 only')
    results, pairs = core.run_scope(f, args.reps, args.seed, True)

    payload = {
        'design': {
            'purpose': 'frozen 2022 freshness replay of nearest-rank disagreement test',
            'strata': ['5-8','9-16','17+'],
            'matching': 'identical implementation to 2023-24 matched test',
            'ranking_model': 'raw full73 XGBoost trained 2015-2021 only',
            'calibration_used': False,
            'precommitted_observed_magnitude_floor': core.MAGNITUDE_FLOOR,
            'precommitted_strong_ci_floor': core.STRONG_CI_FLOOR,
            'freshness_year': 2022,
            'pristine_holdout_claimed': False,
            'sealed_final_holdout': '2025',
            'holdout_2025_read': False,
        },
        'year_2022': results,
    }

    outj = Path(args.out_json)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(payload, indent=2))
    outp = Path(args.out_pairs)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(outp, index=False)

    print(json.dumps(payload, indent=2), flush=True)
    print('[2022-nearest-rank-freshness] 2025 NOT READ', flush=True)


if __name__ == '__main__':
    main()
