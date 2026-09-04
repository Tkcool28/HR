# v1.2 nearest-rank matched result — 2026-09-04

## Scope

This records the precommitted same-slate nearest-obvious-rank matched follow-up for the frozen full73 champion and the subsequently frozen 2022 freshness replay.

The design and decision thresholds were committed before the 2023–24 matched result in `docs/v12_matched_pair_precommit_2026-09-04.md`.

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

**Decision: do not promote 17+ to an operational hidden-pick rule.**

## Year diagnostics for 17+

| Year | N pairs | Selected rate | Matched control | Lift | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| 2023 | 113 | 28.32% | 17.70% | +10.62 pp | [-0.93, +22.41] pp |
| 2024 | 64 | 25.00% | 18.75% | +6.25 pp | [-9.68, +21.31] pp |

Both years retain a positive point estimate after nearest-rank matching, but neither yearly CI independently resolves the effect. The smaller 2024 sample is especially wide.

## Frozen 2022 freshness replay

The exact same rank bands, nearest-rank matching implementation, +5 pp magnitude floor, +2 pp strong-CI floor, and 10,000-replicate date-cluster bootstrap were replayed on 2022 without changing any cutoff after seeing 2023–24.

For this replay:

- the full73 raw XGBoost architecture was trained on **2015–2021 only**;
- 2022 was scored out-of-sample with **raw XGBoost probabilities**;
- isotonic calibration was **not** fit/evaluated for the freshness result;
- the obvious-power proxy used the same five frozen long-horizon batter features and preprocessing based on 2015–2021;
- 2022 is explicitly a development freshness/replication check, not a pristine final holdout.

The reproducible workflow rebuilt the trusted 406,728-row / 73-feature 2015–24 matrix, scored 43,740 2022 batter-games over 179 slate dates, and passed the no-2025 contract.

### 2022 matched results

| Obvious rank | N pairs | Selected rate | Matched control | Lift | 95% paired CI | Holm p | Precommitted grade |
|---|---:|---:|---:|---:|---:|---:|---|
| **5–8** | 156 | **23.72%** | 12.82% | **+10.90 pp** | **[+2.92, +18.95] pp** | **0.024** | **Strong** |
| 9–16 | 142 | 16.90% | 14.08% | +2.82 pp | [-5.56, +10.96] pp | 0.627 | Fail |
| **17+** | 93 | **12.90%** | **13.98%** | **-1.08 pp** | **[-10.64, +8.33] pp** | **0.668** | **Fail** |

### Replication interpretation

The primary deep-disagreement lead **does not replicate in 2022**. Rank-17+ full73 promotions went from a +9.04 pp matched point estimate in combined 2023–24 to **-1.08 pp** in the frozen 2022 replay. The 2022 interval is wide, but its center is essentially zero/slightly negative and it fails both the magnitude and CI criteria.

This materially weakens the hypothesis that **obvious-power rank 17+** is a stable hidden-pick regime. It should be **demoted, not refined into another post-hoc cutoff**.

A different band, 5–8, happens to be very strong in 2022: +10.90 pp with a lower CI bound of +2.92 pp and Holm p=.024, satisfying the precommitted strong-signal thresholds **for that year**. We will not switch the hidden-pick hypothesis to 5–8 after observing that result, because 5–8 failed the combined 2023–24 matched test (+3.01 pp, CI crossing zero). Treating the best band in each period as the target would be exactly the subgroup chasing this process is designed to prevent.

The band instability is itself useful evidence: contextual promotions appear capable of working, but **the location of the strongest disagreement signal is not stable enough to encode as a rank-band betting policy from development data**.

## Relationship to the broader top-5% result

This failure does **not** invalidate the previously resolved broad ranking result:

- full73 daily top 5%: **21.69%**
- obvious-power daily top 5%: **20.28%**
- lift: **+1.41 pp**
- paired 2023–24 date-bootstrap CI: **+0.42 to +2.39 pp**

That test asks a broader, pre-existing question: whether the contextual model improves the actionable candidate pool relative to long-horizon batter power. The deep-disagreement analyses asked a much narrower follow-up question about where that advantage concentrates.

Current evidence therefore supports the model as a **better broad top-5% ranking engine**, but does not support a special 17+ (or any other disagreement-band) production rule.

## Final current verdict

1. The nearest-rank test materially reduced the apparent 17+ lift, validating the tighter control.
2. The remaining 2023–24 +9.04 pp 17+ point estimate failed its precommitted CI criterion.
3. The exact frozen 2022 replay then produced **-1.08 pp** for 17+, so the deep-disagreement hypothesis does **not** receive replication support.
4. The strong 2022 5–8 result is recorded but is **not promoted post hoc**, because that band failed in 2023–24.
5. No disagreement-rank hidden-pick policy is promoted.
6. The broader full73 daily top-5% advantage over obvious-power remains the strongest current evidence of actionable contextual ranking signal.
7. Further development-data subgroup mining should stop unless a new hypothesis is justified independently and predeclared; repeated slicing of 2023–24 would further erode inferential value.
8. **2025 remains sealed.**
