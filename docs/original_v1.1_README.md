# MLB Home Run Probability Model

A calibrated home-run probability model trained on Statcast / Baseball Savant
data 2015–2024, with 2025 held out as a final, owner-gated holdout.

## Status

**v1.1** (current). The model is trained, calibrated, and independently
audited. The 2025 holdout predictions are written to disk but the 2025
outcomes are NOT read — that read is owner-gated and only happens with
explicit sign-off.

## Pipeline (data → features → train → audit)

| Stage | Input | Output | Spec |
|---|---|---|---|
| Data engineering | Baseball Savant BIP CSV chunks, 2015–2024 | `data/curated/pa.parquet`, `data/curated/game.parquet`, `data/curated/park_factors.parquet`, `data/raw/bip_all.parquet` | split by year: train 2015–2022, val 2023–2024, holdout 2025 |
| Feature engineering | curated parquets | `features/v1.1/game_features.parquet` (532,554 rows × 113 features) | anti-leak rolling windows, as-of dates, no 2025 outcome use |
| Model training | v1.1 features | `models/v1.1/{lr_baseline,xgb_v1,isotonic_v1}.joblib`, `models/v1.1/holdout_predictions.parquet` | LR baseline + XGBoost 25 Optuna trials + isotonic calibration on val only |
| Audit | trainer outputs | `reports/trainer_v11_verify_report.md` | 11 independent checks, all 36 sub-checks PASS |

## Headline results (val 2023–2024)

| Model | Brier ↓ | AUC ↑ | Top-5% ↑ |
|---|---:|---:|---:|
| v1 LR | 0.0957 | 0.6211 | 20.6% |
| v1 XGB+iso | 0.0954 | 0.6271 | 21.0% |
| **v1.1 LR** | **0.0954** | **0.6300** | **22.1%** |
| **v1.1 XGB+iso** | **0.0951** | **0.6366** | **22.7%** |

v1.1 adds 20 features: batter xwOBA, barrel rate, EV / hard-hit rate, launch
angle, park factors (3-yr and by-hand). Walk-forward backtest on 2019–2024
holds in a healthy range (Brier 0.089–0.104, AUC 0.63–0.66).

## Repository layout

```
hr_model/
├── README.md                       # this file
├── data/
│   ├── raw/
│   │   └── bip_all.parquet         # consolidated BIP data 2015-2024 (18.7 MB)
│   ├── curated/
│   │   ├── pa.parquet              # pitch-level PAs 2015-2024 (60 MB)
│   │   ├── game.parquet            # game-level table with corrected team→park
│   │   ├── roster.parquet
│   │   ├── park_factors.parquet    # 309 (park, year) rows
│   │   └── splits/                 # train/val/holdout PA IDs
│   ├── splits/                     # split id files
│   └── scratch/                    # intermediate
├── features/
│   ├── v1/                         # v1 features (no QoC, kept for splits)
│   └── v1.1/                       # v1.1 features (with QoC + park)
├── models/
│   ├── v1/                         # v1 trained models
│   ├── v1.1/                       # v1.1 trained models (current)
│   └── holdout_predictions.parquet # 2025 calibrated predictions, NO outcomes
├── src/
│   ├── data/                       # data engineering source
│   ├── features/
│   │   └── v1.1/                   # v1.1 feature build script
│   └── models/                     # train.py, train_v11.py
├── tests/                          # integrity tests
└── reports/                        # all reports
```

## Anti-leak invariants (all enforced, all PASS)

1. Holdout `max_as_of_date` <= 2024-12-31
2. Holdout `hr_in_game` is NaN everywhere
3. Rolling windows are time-bounded (searchsorted-based cumsum lookups)
4. No 2025 row in train or val
5. Park factors computed only from BIP data 2015-2024
6. Isotonic calibrator fit on val block only (X_thresholds_ domain confirms)
7. Shuffle-target falsification: model Brier < shuffle Brier (real signal, no leakage)

## How to inspect

```python
import pandas as pd
features = pd.read_parquet('features/v1.1/game_features.parquet')
holdout = pd.read_parquet('models/v1.1/holdout_predictions.parquet')
```

To retrain (DO NOT touch 2025 outcomes at any point):
```bash
python3 src/models/train_v11.py    # writes models/v1.1/
```

To run the integrity tests:
```bash
python3 tests/test_no_lookahead.py
python3 tests/test_bip_integrity.py
python3 tests/test_calibration.py
python3 tests/test_split_integrity.py
```

## What is NOT in this repo

- Raw BIP CSV chunks (~1 GB, ~650 files) — re-derivable from
  `data/raw/bip_all.parquet` or directly from Baseball Savant
- The 2025 holdout outcomes — owner-gated, never read by the trainer
- The full team plan YAML files (kept under `.mavis/plans/` in the
  development environment)
