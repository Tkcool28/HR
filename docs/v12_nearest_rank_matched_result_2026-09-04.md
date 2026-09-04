# v1.2 nearest-rank matched result — 2026-09-04

## Scope

This records the precommitted same-slate nearest-obvious-rank matched follow-up for the frozen full73 champion.

The design and decision thresholds were committed before the result in `docs/v12_matched_pair_precommit_2026-09-04.md`.

**2025 was not read or evaluated and remains sealed.**

## Precommitted decision rule

A disagreement band survives for operational consideration only if:

1. observed paired HR-rate lift is at least **+5.0 percentage points**; and
2. the 95% paired slate-date bootstrap CI is entirely above zero.

Strong-signal grade additionally requires the CI lower bound to be at least **+2.0 pp**.

## Matching quality

Controls were chosen from the same slate date and same predeclared obvious-power rank band, nearest in obvious rank, without replacement, with outcome-blind deterministic ties.

| Band | Selected total | Matched | Match rate | Mean rank gap | Median gap | Max gap |
|---|---:|---:|---:|---:|---:|---:|
| 5–8 | 352 | 332 | 94.32% | 1.17 | 1 | 3 |
| 9–16 | 326 | 326 | 100% | 1.15 | 1 | 4 |
| **17+** | **177** | **177** | **100%** | **1.02** | **1** | **2** |

The 17+ comparison is therefore extremely tight in obvious-power rank: 90% of matched controls are one rank away and none are more than two ranks away.

## Combined 2023–24 matched results

| Obvious rank | Selected HR rate | Matched-control HR rate | Paired lift | 95% paired date-bootstrap CI | Holm-adjusted matched p | +5 pp floor? | CI > 0? | Survives? |
|---|---:|---:|---:|---:|---:|---|---|---|
| 5–8 | 21.39% | 18.37% | +3.01 pp | [-3.42, +9.68] pp | 0.351 | No | No | **No** |
| 9–16 | 23.01% | 19.63% | +3.37 pp | [-3.05, +9.76] pp | 0.351 | No | No | **No** |
| **17+** | **27.12%** | **18.08%** | **+9.04 pp** | **[-0.56, +18.24] pp** | **0.108** | **Yes** | **No** | **No** |

### Interpretation

The earlier same-band 17+ comparison was 27.12% versus 11.23% (+15.89 pp). Nearest-rank matching raises the control rate to 18.08% and reduces the estimated lift to **+9.04 pp**.

Therefore part of the original 17+ separation was indeed band-composition: full73 tended to promote the stronger portion of the broad 17+ obvious-power tail.

However, the residual matched point estimate remains large enough to clear the precommitted +5 pp magnitude floor. It does **not** clear the statistical-survival rule because the clustered CI crosses zero. The exact one-sided matched-binary p-value is 0.0361 before multiplicity adjustment and 0.1084 after Holm correction across the three predeclared bands.

**Decision: do not promote 17+ to an operational hidden-pick rule.** It remains a plausible but unresolved differentiated signal.

## Year diagnostics for 17+

| Year | N pairs | Selected rate | Matched control | Lift | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| 2023 | 113 | 28.32% | 17.70% | +10.62 pp | [-0.93, +22.41] pp |
| 2024 | 64 | 25.00% | 18.75% | +6.25 pp | [-9.68, +21.31] pp |

Both years retain a positive point estimate after nearest-rank matching, but neither yearly CI independently resolves the effect. The smaller 2024 sample is especially wide.

## Freshness check

The already-frozen next step is to replay the exact same disagreement bands, matching implementation, and +5 pp / CI thresholds on **2022** using raw full73 scores from a model trained only through 2021.

2022 is not a pristine final holdout: it was part of the broader model-development/calibration architecture. For this check, calibration is not used and the raw XGBoost rank is out-of-sample relative to the 2015–2021 base fit. The purpose is replication/freshness only.

No 2022 result may change the band cutoffs or decision thresholds.

## Current verdict

- The nearest-rank test materially reduced the apparent 17+ lift, validating the need for the control.
- The residual +9.04 pp point estimate is potentially meaningful but **fails the precommitted survival criterion** because its CI crosses zero.
- 5–8 and 9–16 also fail the operational rule after matching.
- No hidden-pick policy is promoted from 2023–24.
- 2025 remains sealed.
