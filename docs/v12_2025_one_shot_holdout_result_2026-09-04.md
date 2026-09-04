# v1.2 Frozen 2025 One-Shot Holdout Result — 2026-09-04

## Verdict

The frozen full-73 v1.2 HR model generalized successfully to the previously sealed 2025 holdout.

The primary product-facing result was strong:

- 2025 daily top-5% HR rate: **22.6274%** (515 HR / 2,276 selections)
- 2025 top-4-per-day HR rate: **25.8152%** (190 / 736)
- 2025 top-2-per-day HR rate: **29.8913%** (110 / 368)
- 2025 top-1-per-day HR rate: **28.8043%** (53 / 184)
- full 2025 target-universe HR rate: **11.7307%**

The primary top-5% result exceeded the frozen obvious-power comparator:

- full73 top-5%: **22.6274%**
- obvious-power top-5%: **21.1775%**
- observed delta: **+1.4499 percentage points**
- 10,000 paired slate-date bootstrap delta mean: **+1.4523 pp**
- paired bootstrap 95% interval: **[0.0000, +2.9106 pp]**
- Pr(delta > 0): **0.9734**

This is evidence that the model is doing more than simply ranking the most obvious power hitters.

## Frozen evaluation contract

The 2025 result was evaluated under the contract frozen before 2025 outcome acquisition:

- active features: 73
- feature-list SHA256: `861d89a62a71d2da32a3092cd7d0fe36b25413314b8c1be8b4755f7e9c66d9f5`
- model: frozen full-73 aggressive XGBoost champion
- boosting rounds: 194
- fit years: 2015–2023
- calibration year: 2024
- holdout year: 2025
- ranking authority: raw XGBoost score
- primary selector: daily top 5%
- primary comparator: frozen obvious-power composite
- primary uncertainty: 10,000 paired slate-date bootstrap replicates
- post-hoc tuning/subgroup selection: false
- sportsbook odds: not used
- 2026 data: not used

## Holdout surface

- target rows: 43,740
- games: 2,430
- slate dates: 184
- target universe: exactly 18 hitters per included game
- 2025 holdout split rows: 43,740
- exact active feature names/order matched the pre-acquisition frozen reference
- all holdout feature provenance passed strict `< game_date` checks

## Authorized 2025 acquisition

Fresh 2025 Statcast acquisition used the same trusted source lineage as the historical rebuild and authoritative MLB regular-season game IDs.

- unique regular-season game IDs: 2,430
- deduplicated regular-season pitch rows: 712,528
- processed BIP rows: 124,441
- season span included the March Tokyo games through September 28

BIP processing preserved the frozen historical contract:

1. `description == hit_into_play`
2. one BIP per `(game_pk, at_bat_number)`
3. fixed 17-event BIP whitelist
4. non-null `launch_speed`

2025 park factors were built strictly from observations through 2024. Earlier 2025 observations could contribute only to later-2025 rolling/season features through strict pregame rollups.

## Execution history and plumbing-only recovery

### Initial authorized run

- Actions run: `33903253357`
- exact head: `712f463257622b628ab65d0ef80e348cbf5adfcc`

This run successfully acquired 2025 but failed before the holdout feature matrix or any 2025 model score was produced.

Failure:

- `ModuleNotFoundError: trusted_v12`
- cause: script-by-path sibling import plumbing

No 2025 performance metric had been computed or inspected at that point, so the model/evaluation contract remained unbiased by the holdout outcome.

The run uploaded the exact acquired 2025 data as an immutable artifact:

- artifact ID: `9949402645`
- artifact SHA256: `a343f43cae8db56b898f1ecae424814fd58cf865240d66e24cbce858c6d65656`

### Plumbing repair

Only the import path was repaired. The model, features, parameters, thresholds, ranking rule, comparator, and evaluation metrics were unchanged.

Repair commit:

- `8100ffe6412f4077912bf4755036da507b2fe839`

### Successful frozen recovery execution

The recovery workflow reused the exact immutable acquisition artifact above after SHA verification rather than acquiring a different 2025 dataset.

- successful Actions run: `33910029028`
- exact workflow head: `28e7b1adde54da110c59c12439e6d2834192b187`
- conclusion: SUCCESS
- output artifact ID: `9951133562`
- output artifact SHA256: `3b42f6a80944f53247cb400e7e0be0bf5ac407439ae402fc68bbf034ca06ddec`
- frozen post-run contract: PASS

The workflow emitted:

`FROZEN 2025 ONE-SHOT CONTRACT PASS; PRIMARY RESULT LOCKED`

## Primary 2025 result

| Metric | Full73 | Frozen obvious-power comparator |
|---|---:|---:|
| selections | 2,276 | 2,276 |
| HR | 515 | 482 |
| HR rate | **22.6274%** | 21.1775% |
| delta | **+1.4499 pp** | — |

Full73 top-5% enrichment over the complete target universe:

- base: 11.7307%
- top-5%: 22.6274%
- absolute lift: **+10.8967 pp**
- rate multiple: approximately **1.93× base**

Paired 10,000-replicate slate-date bootstrap:

- reference mean: 21.1717%
- full73 mean: 22.6240%
- delta mean: +1.4523 pp
- delta median: +1.4608 pp
- 95% interval: [0.0000, +2.9106 pp]
- Pr(delta > 0): 97.34%

The lower bound landing exactly at zero should not be overstated as conventional threshold-based significance. It is nevertheless strong supportive holdout evidence that the full model improved on the frozen obvious-power ranking.

## Secondary 2025 model metrics

| Metric | Raw | Calibrated |
|---|---:|---:|
| Brier | 0.101502 | **0.101427** |
| ROC AUC | **0.625256** | 0.624617 |
| Average Precision | **0.174653** | 0.171276 |
| Log loss | **0.351586** | 0.352059 |
| ECE (10-bin) | 0.009334 | **0.002291** |

Calibration fit:

- raw slope: 1.04557
- raw intercept: -0.00223
- raw mean prediction: 12.6421%
- calibrated slope: 0.93844
- calibrated intercept: -0.11623
- calibrated mean prediction: 11.6945%
- observed rate: 11.7307%

The calibrated probabilities transported well to 2025: mean calibrated probability was within about 0.04 percentage points of the observed HR rate, and ECE fell to 0.23%.

## 2025 tail-ranking results

| Selector | HR rate |
|---|---:|
| daily top 10% | 20.2853% |
| daily top 5% | **22.6274%** |
| daily top 2% | 25.6122% |
| daily top 1% | 27.7228% |
| top 1/day | 28.8043% |
| top 2/day | **29.8913%** |
| top 4/day | **25.8152%** |
| top 8/day | 23.0978% |

The top-4/day result is especially relevant to the intended product usage: shortlist roughly four to five HR candidates on a daily slate.

## Structural comparison with obvious power

- top-5 overlap: 1,420 / 2,276 = 62.39%
- full73-only selections: 856
- full73-only HR rate: **20.6776%**
- mean obvious-power rank of full73-only selections: 33.32
- median obvious-power rank: 26

Thus approximately 37.6% of the model's top-5% selections were not in the obvious-power top-5%, and those differentiated selections still homered at about 20.68%.

## Comparison with 2023–2024 development assessment

| Metric | 2023–24 dev | 2025 sealed holdout |
|---|---:|---:|
| raw AUC | 0.61794 | **0.62526** |
| calibrated AUC | 0.61751 | **0.62462** |
| calibrated Brier | 0.10250 | **0.10143** |
| daily top 5% | 21.69% | **22.63%** |
| top 1/day | 26.46% | **28.80%** |
| top 2/day | 25.77% | **29.89%** |
| top 4/day | 24.23% | **25.82%** |
| top 8/day | 22.67% | **23.10%** |

The sealed holdout did not expose a dev-only ranking illusion. The actionable daily tail generalized and was stronger in 2025 on the principal shortlist metrics.

## Interpretation

The correct conclusion is not that the model is perfect or that it has already beaten sportsbook pricing. No sportsbook market data was used in this holdout evaluation.

The justified conclusions are:

1. The frozen model contains real predictive ranking signal that generalized to an untouched season.
2. The model's highest-ranked daily candidates materially outperform the full hitter universe.
3. The intended top-4/top-5 daily product bucket generalized strongly.
4. The model improves on a frozen obvious-power-only comparator in the sealed holdout.
5. Calibration transported unusually cleanly to 2025.
6. Actual sportsbook/book-ranking superiority remains a separate benchmark to be tested after this frozen result.

## Holdout status after execution

2025 is now opened and must never again be described as a pristine holdout for this architecture.

Future changes informed by these results require a different forward or untouched validation surface.
