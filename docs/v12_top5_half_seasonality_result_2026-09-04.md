# v1.2 top-5 half-seasonality result — 2026-09-04

## Scope

This records the single predeclared powered seasonality test for the frozen full73 contextual model versus the frozen long-horizon batter-only obvious-power proxy.

The experiment was precommitted before any half-season outcome was inspected in:

- `docs/v12_dynamic_bucket_regime_precommit_2026-09-04.md`
- `docs/v12_top5_half_maturity_thresholds_precommit_2026-09-04.md`

The primary split was frozen as:

- **FIRST_HALF:** Opening Day through June 30
- **SECOND_HALF:** July 1 through end of regular season

No calendar-month outcome table was emitted. No 5–8 / 9–16 / 17+ disagreement-bucket seasonality analysis was run.

**2025 was not read or evaluated and remains sealed.**

## Authoritative reproducible run

- GitHub Actions run: **33891442637**
- exact head: **79a61b635ef87f4dcafdc6de35b27d813441e06c**
- job: **101083807872**
- conclusion: **SUCCESS**
- artifact: `v12-top5-half-seasonality`
- artifact ID: **9944075131**
- artifact SHA256: **71cda78be4c5bde2f51281c6eb30bceacc6f6fcd3847a9043ae867ccc46934a7**

The run rebuilt the trusted 406,728-row / 73-feature 2015–2024 matrix, re-scored the frozen full73 champion, rebuilt the same obvious-power proxy, ran 10,000 paired slate-date bootstrap replicates, ran the predeclared half-vs-half interaction, audited recent-feature maturity, and passed the no-month / no-2025 contract.

Synthetic controls passed before real inference, including:

- June 30 / July 1 phase boundary
- deterministic exact daily top-5% selection
- known-positive second-half interaction
- identical-model zero interaction
- genuine support-count versus rate-feature discrimination
- 2025 fail-closed

## Primary top-5% results

| Phase | Slate dates | Selected batter-games | Obvious-power HR rate | Full73 HR rate | Full73 lift | 95% paired slate-date CI | P(lift > 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FIRST_HALF** | 182 | 2,259 | 19.3006% | **20.6729%** | **+1.3723 pp** | **[0.0000, +2.7608] pp** | 97.17% |
| **SECOND_HALF** | 177 | 2,213 | 21.2833% | **22.7293%** | **+1.4460 pp** | **[+0.0466, +2.8689] pp** | 97.84% |

The FIRST_HALF percentile lower bound lands exactly at zero under the frozen 10,000-replicate empirical bootstrap, so it should be described as **borderline rather than strictly clearing zero**. SECOND_HALF clears zero, but only narrowly at the lower endpoint.

The important result is not which half has a prettier standalone interval; it is the predeclared interaction below.

## Predeclared half-season interaction

Definition:

> SECOND_HALF full73-vs-obvious lift minus FIRST_HALF full73-vs-obvious lift

Observed interaction:

- FIRST_HALF lift: **+1.3723 pp**
- SECOND_HALF lift: **+1.4460 pp**
- SECOND minus FIRST: **+0.0737 pp**
- bootstrap mean: **+0.0813 pp**
- bootstrap median: **+0.0938 pp**
- 95% CI: **[-1.8807, +2.0395] pp**
- P(interaction > 0): **53.46%**

### Decision

**There is no evidence that the broad contextual top-5% advantage is materially stronger in the second half than the first half.**

The observed lifts are essentially the same: +1.37 pp versus +1.45 pp. The interaction is near zero and its interval is wide and centered around no difference.

Therefore the predeclared gate for a calendar-month localization follow-up is **not triggered**.

Do not inspect monthly outcome tables or return to sparse monthly disagreement buckets from this result.

## Baseball environment by half

The underlying full-slate HR rate did move in the direction expected from a stronger later-season hitting environment:

- FIRST_HALF base HR rate: **11.3915%**
- SECOND_HALF base HR rate: **12.2940%**
- difference: approximately **+0.90 pp** later

This indicates a meaningful change in the raw HR environment in these 2023–24 samples.

However, the model's **incremental advantage over obvious-power** did not materially change with it.

That distinction is important: hitters as a whole homered more often later, but the extra ranking value supplied by full73 context remained approximately stable.

## Feature-maturity audit

All 51 active features containing `14d` or `30d` had zero explicit NaN rate in both halves because the feature pipeline emits defined/smoothed values. Therefore NaN rate alone is not an informative maturity measure here.

The more useful diagnostic is genuine historical support.

### Full slate population

| Support metric | FIRST_HALF | SECOND_HALF |
|---|---:|---:|
| batter PA, prior 14d: median | 39 | 40 |
| batter PA, prior 14d: <20 | **19.70%** | **14.14%** |
| batter PA, prior 30d: median | 70 | 83 |
| batter PA, prior 30d: <20 | **13.98%** | **6.23%** |
| pitcher PA, prior 30d: median | 91 | 98 |
| pitcher PA, prior 30d: zero | **10.10%** | **3.65%** |
| pitcher PA, prior 30d: <20 | **13.01%** | **6.08%** |
| top-pitch support, prior 30d: median pitches | 348.5 | 375 |
| top-pitch support: zero | **10.10%** | **3.65%** |
| top-pitch support: <100 pitches | **19.92%** | **9.25%** |

### Full73 daily top-5% selections

| Support metric | FIRST_HALF | SECOND_HALF |
|---|---:|---:|
| batter PA, prior 14d: median | 51 | 51 |
| batter PA, prior 14d: <20 | **7.30%** | **1.90%** |
| batter PA, prior 30d: median | 103 | 106 |
| batter PA, prior 30d: <20 | **6.77%** | **0.45%** |
| pitcher PA, prior 30d: median | 88 | 93 |
| pitcher PA, prior 30d: zero | **11.86%** | **5.11%** |
| pitcher PA, prior 30d: <20 | **15.18%** | **8.45%** |
| top-pitch support, prior 30d: median pitches | 336 | 359 |
| top-pitch support: zero | **11.86%** | **5.11%** |
| top-pitch support: <100 pitches | **22.13%** | **12.83%** |

### Maturity interpretation

Claude's proposed confound is real: recent-history support is clearly thinner earlier in the season, particularly for pitcher and top-pitch information and, to a lesser extent, batter rolling windows.

Despite that, the full73 top-5 contextual lift is almost unchanged between halves.

This means:

1. we should **not** claim that the later-season contextual lift is stronger because rolling features mature; the half interaction gives no evidence of a stronger later lift;
2. we should **not** call early-season contextual information ineffective; the FIRST_HALF point estimate is already +1.37 pp;
3. the most defensible interpretation is that the broad top-5 contextual advantage appears **robust to a substantial change in rolling-history maturity** in 2023–24.

The maturity table is a measurement/confound diagnostic, not independent confirmation of predictive signal.

## Diagnostic bug and correction

The first green implementation had an over-broad support-feature detector that could classify rate-valued `*_hr_per_pa_vs_*_30d` columns as if they were support counts. This did **not** affect:

- model scores
- top-5 selections
- HR outcomes
- FIRST_HALF / SECOND_HALF lifts
- bootstrap intervals
- half-season interaction

It only polluted the descriptive support table with meaningless count thresholds applied to rate features.

Before freezing this result, the detector was restricted to four genuine active support/count fields:

- `batter_pa_14d`
- `batter_pa_30d`
- `pitcher_pa_30d`
- `top_pitch_total_pitches_30d`

A synthetic anti-regression test now asserts that `*_hr_per_pa_vs_*` rates cannot enter the support table. The authoritative exact-head run above reproduced the primary inference identically with the corrected diagnostic.

## Relationship to the full-sample top-5 result

Previously resolved 2023–24 broad comparison:

- full73 daily top 5%: **21.69%**
- obvious-power daily top 5%: **20.28%**
- lift: **+1.41 pp**
- paired 95% CI: **+0.42 to +2.39 pp**

The half-season decomposition is entirely consistent with that full-sample result:

- FIRST_HALF: +1.37 pp
- SECOND_HALF: +1.45 pp

Rather than discovering that the broad edge exists only later, this experiment suggests the contextual advantage is **distributed surprisingly evenly across the two broad season phases**.

## Current decision

1. The powered top-5% metric was the correct instrument for the first seasonality test; disagreement-bucket cells would have been underpowered.
2. There is no resolved FIRST_HALF-vs-SECOND_HALF heterogeneity in full73's broad top-5 contextual lift.
3. The second-half raw HR environment is higher, and rolling-history support is materially more mature, but neither produces a detectable increase in full73's incremental advantage over obvious-power.
4. Do **not** drill calendar months from this result; the predeclared follow-up gate did not trigger.
5. Do **not** resume sparse disagreement-bucket mining on development outcomes.
6. The strongest current actionable-development finding remains the broad full73 daily top-5% advantage over obvious-power, now with evidence that its magnitude is broadly stable across season halves.
7. No dynamic season-phase or hidden-bucket production policy is justified from <=2024 evidence.
8. **2025 remains sealed.**
