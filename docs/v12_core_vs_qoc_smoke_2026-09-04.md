# Trusted v1.2 core vs QoC smoke ablation — 2026-09-04

2025 remained sealed and was not read or evaluated.

## Controlled comparison

Both runs used the same repaired historical pipeline, target universe, regular-season context, actual venue park factors, 2015-2020 tuning train, 2021 Optuna selection, 2022 isotonic calibration, and 2023-2024 development assessment. Both used the same 15-trial seeded Optuna smoke search.

The QoC run preserved the original 53 core feature columns bit-for-bit and added exactly 20 leakage-safe Statcast contact-quality features, producing a 73-feature matrix. All delivered temporal tests and trusted QoC provenance tests passed.

## 2023-2024 comparison

| Metric | 53-feature core | 73-feature core + QoC | Delta |
|---|---:|---:|---:|
| Calibrated Brier | 0.102745 | 0.102544 | -0.000201 |
| Calibrated AUC | 0.611066 | 0.616544 | +0.005477 |
| Calibrated AP | 0.164753 | 0.166383 | +0.001629 |
| Calibrated log loss | 0.356178 | 0.354998 | -0.001180 |
| Calibrated top-5% HR rate | 21.5281% | 21.8765% | +0.3484 pp |
| Calibration ECE | 0.008439 | 0.006362 | -0.002077 |
| Raw XGB AUC | 0.611571 | 0.617611 | +0.006040 |
| Raw XGB AP | 0.167583 | 0.170770 | +0.003187 |
| Raw XGB top-5% HR rate | 21.7836% | 21.9926% | +0.2090 pp |
| LR Brier | 0.102715 | 0.102579 | -0.000136 |
| LR AUC | 0.609769 | 0.614568 | +0.004799 |
| LR AP | 0.167038 | 0.170147 | +0.003109 |

Base HR prevalence on the 2023-2024 assessment rows was 11.8381%. The QoC calibrated top-5% bucket therefore hit at about 1.848x overall prevalence, versus about 1.819x for the core model.

## Year-by-year top-5% result

| Year | Core | Core + QoC | Delta |
|---|---:|---:|---:|
| 2023 | 21.9456% | 22.8677% | +0.9221 pp |
| 2024 | 21.0108% | 21.1979% | +0.1872 pp |

Brier and AUC also improved in both 2023 and 2024 individually, so the aggregate gain is not caused by one season masking deterioration in the other.

## Smoke-tune behavior

The core 15-trial 2021 Optuna search was already fairly flat (best Brier about 0.103717; median about 0.103757), reducing concern that the QoC gain simply came from a lucky hyperparameter draw. The 73-feature QoC smoke search improved the best 2021 tuning Brier to about 0.103582.

## Decision

**QoC graduates.** The contact-quality block produced a modest but broad improvement in independent-of-tuning development metrics, including the practically important top-5% ranking metric, while also improving Brier and calibration. It is retained for the aggressive tuning phase.

Because the 2023-2024 results are now being used to make feature/model decisions, those seasons are henceforth treated as a development assessment rather than a final untouched holdout. The sealed 2025 season remains reserved for the eventual one-shot final test.
