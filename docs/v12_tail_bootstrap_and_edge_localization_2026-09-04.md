# v1.2 tail bootstrap and edge localization — 2026-09-04

## Scope

This report records clustered uncertainty for the frozen 73-feature champion and the first explicit analysis of whether its actionable tail is doing more than selecting obvious long-horizon power hitters.

**2025 was not read or evaluated and remains sealed.**

Development evidence is 2023-2024 only. Daily ranking uses raw XGBoost probability.

## Frozen-model clustered bootstrap

The full73 champion and equal-retuned pruned62 challenger were re-scored from frozen contracts over the same 86,112-row 2023-2024 development assessment.

Primary inference resamples full slate dates with replacement for 10,000 paired replicates. A secondary sensitivity analysis resamples whole games within each slate for 500 paired replicates while preserving complete game batter clusters.

### Full73 vs pruned62

Combined 2023-2024:

| Selector | Full73 observed | Pruned62 observed | Delta pruned62-full73 | 95% paired date-bootstrap CI | Pr(delta > 0) |
|---|---:|---:|---:|---:|---:|
| daily top 5% | **21.6905%** | 21.6458% | -0.0447 pp | [-0.7314, +0.6562] pp | 44.3% |
| top 4/day | **24.2340%** | 23.8162% | -0.4178 pp | [-2.0195, +1.1838] pp | 28.9% |

Whole-game-cluster sensitivity agrees directionally:

- daily top-5% delta pruned62-full73 mean: -0.1490 pp; 95% CI [-0.9392, +0.6485] pp
- top-4/day delta pruned62-full73 mean: -0.4370 pp; 95% CI [-2.1588, +1.2204] pp

Interpretation: the reduced challenger does not have evidence of a tail advantage. The full73 architecture remains preferred from the combined evidence (global metrics, equal retune, and tail point estimates), but the tail difference between these two closely related models is not independently resolved by 2023-2024 alone.

## Obvious-power proxy

To ask whether the model is merely selecting familiar power bats, an intentionally simple **long-horizon batter-only proxy** was constructed. It is not historical public ownership, sportsbook pricing, or consensus.

The proxy uses the available season/career long-horizon batter power/contact-quality features:

- batter barrel rate, season
- batter barrel rate, career
- batter xwOBA on contact, season
- batter xwOBA on contact, career
- batter average exit velocity, season

Each feature is ranked within the daily slate and averaged to form an obvious-power score. It deliberately excludes park, pitcher vulnerability, recent form, platoon, and pitch-matchup context.

## Full73 vs obvious-power proxy

### Daily top 5% — clear full73 advantage

Combined 2023-2024:

- obvious-power proxy: **20.2818%**
- full73: **21.6905%**
- observed lift: **+1.4088 percentage points**
- paired 10,000-date-bootstrap 95% CI: **+0.4219 to +2.3924 pp**
- Pr(full73 delta > 0): **99.71%**

By year:

- 2023: 20.2043% -> 22.2025%, delta **+1.9982 pp**, 95% CI **+0.5459 to +3.4864 pp**, Pr(delta>0)=99.61%
- 2024: 20.3604% -> 21.1712%, delta **+0.8108 pp**, 95% CI **-0.5372 to +2.1720 pp**, Pr(delta>0)=87.53%

This is the strongest current evidence that the contextual model is adding practical ranking information beyond obvious long-horizon batter power. The combined top-5% result is positive with a paired CI entirely above zero and the direction is positive in both years.

### Top 4/day — no full73 advantage overall

Combined 2023-2024:

- obvious-power proxy: **24.6518%**
- full73: **24.2340%**
- observed delta: **-0.4178 pp**
- 95% paired CI: **-2.7159 to +1.8802 pp**
- Pr(full73 delta > 0): 34.39%

The year split is unstable:

- 2023: full73 +1.5278 pp over proxy; CI crosses zero; Pr(delta>0)=79.83%
- 2024: full73 -2.3743 pp vs proxy; CI crosses zero narrowly; Pr(delta>0)=6.84%

Therefore the current evidence does **not** support a simple policy of replacing obvious-power ranking with the model's exact top four every day. The model's most defensible current advantage is in the broader top-5% candidate set.

## Overlap and differentiated picks

Across 359 development slate dates:

- full73 top-four selections: 1,436
- shared with obvious-power top four: 581
- overlap: **40.46%**
- full73-only selections: 855
- obvious-power-only selections: 855

Hit rates:

| Segment | N | HRs | HR rate |
|---|---:|---:|---:|
| shared top four | 581 | 151 | **25.99%** |
| full73-only | 855 | 197 | 23.04% |
| obvious-only | 855 | 203 | 23.74% |

Combined paired bootstrap for full73-only minus obvious-only:

- observed difference: -0.70 pp
- 95% CI: **-4.60 to +3.19 pp**
- Pr(delta>0): 35.51%

So the differentiated top-four set as a whole is not yet an edge. This prevents interpreting the model's lower overlap as automatically valuable contrarian discovery.

## Predeclared obvious-rank strata inside full73-only picks

Before observing the edge-localization result, full73-only selections were stratified by where the obvious-power proxy ranked them:

| Obvious-power rank | N | HRs | HR rate | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|
| 5-8 | 352 | 74 | 21.02% | 22.86% | 19.21% |
| 9-16 | 326 | 75 | 23.01% | 23.16% | 22.82% |
| **17+** | **177** | **48** | **27.12%** | **28.32%** | **25.00%** |

The 17+ stratum is a notable exploratory signal because it is both the least obvious group and directionally stable across the two years. It is **not promoted yet**: selecting the best-looking stratum after seeing three subgroup outcomes creates selection risk even though the bands themselves were predeclared.

Next test: evaluate all three bands together against non-model-selected hitters in the same obvious-power rank band, using paired 10,000-date clustered resampling and multiplicity-aware inference.

## Context fingerprint of differentiated full73 picks

Within-day mean percentile scores:

| Context group | All hitters | Shared top4 | Full73-only | Obvious-only |
|---|---:|---:|---:|---:|
| park | 50.21% | 70.98% | **69.93%** | 58.92% |
| recent batter form | 50.21% | 86.12% | 77.89% | **84.17%** |
| pitcher vulnerability | 50.21% | 55.29% | **57.17%** | 46.92% |
| pitch matchup | 50.21% | 59.34% | 55.72% | **57.66%** |

The initial fingerprint suggests the model's differentiated promotions are characterized more by **park and pitcher vulnerability** than by simply stronger recent batter form. This is consistent with the earlier destructive ablation in which removing the two park features materially hurt top-four ranking.

This fingerprint is descriptive, not causal. The follow-up disagreement-strata test will determine whether the deepest promotions actually carry repeatable outcome separation before additional subgroup drilling.

## Current verdict

1. **Full73 remains the model architecture.** Equal retuning and clustered sensitivity do not support pruning to 62 features.
2. **The model has a statistically resolved advantage over a batter-only obvious-power proxy in the daily top 5%.** This is currently the clearest practical shine zone.
3. **Exact top-4/day replacement is not supported.** The simple power proxy is slightly better overall there, with substantial 2023/2024 regime variation.
4. **Deep-disagreement picks (obvious-power rank 17+) are the strongest differentiated discovery lead**, but require a multiplicity-aware follow-up before use as policy.
5. 2025 remains sealed.
