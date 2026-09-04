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

So the differentiated top-four set as a whole is not an edge. This prevents interpreting the model's lower overlap as automatically valuable contrarian discovery.

## Predeclared obvious-rank strata inside full73-only picks

Before observing the edge-localization result, full73-only selections were stratified by where the obvious-power proxy ranked them:

| Obvious-power rank | N | HRs | HR rate | 2023 | 2024 |
|---|---:|---:|---:|---:|---:|
| 5-8 | 352 | 74 | 21.02% | 22.86% | 19.21% |
| 9-16 | 326 | 75 | 23.01% | 23.16% | 22.82% |
| **17+** | **177** | **48** | **27.12%** | **28.32%** | **25.00%** |

The 17+ stratum was treated as exploratory after the first look rather than promoted from the point estimate alone. A multiplicity-aware follow-up then evaluated all three predeclared bands together.

## Multiplicity-aware same-band follow-up

For each band, full73-selected hitters were compared with non-full73 hitters from the **same obvious-power rank band**. Inference used:

- 10,000 slate-date bootstrap replicates;
- 10,000 exact within-date hypergeometric label-randomization replicates;
- one-sided alternative: full73-selected HR rate > same-band control HR rate;
- Holm correction across all three predeclared bands.

Combined 2023-2024:

| Obvious rank | Selected N | Selected HR rate | Same-band control rate | Lift | 95% date-bootstrap CI | Pr(lift>0) | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5-8 | 352 | 21.02% | 17.71% | +3.31 pp | [-1.74, +8.37] pp | 89.79% | 0.1064 |
| **9-16** | **326** | **23.01%** | **17.60%** | **+5.41 pp** | **[+0.83, +10.27] pp** | **98.88%** | **0.0350** |
| **17+** | **177** | **27.12%** | **11.23%** | **+15.89 pp** | **[+9.68, +22.36] pp** | **100.00%** | **0.00030** |

The two deeper disagreement bands survive multiplicity control. Rank 5-8 is positive but unresolved.

### Year stability of the deepest band

Obvious-rank 17+ full73 promotions:

- **2023:** 113 selected, 32 HR, **28.32%** vs 11.65% same-band controls; lift **+16.67 pp**; 95% date-bootstrap CI **+8.58 to +24.86 pp**
- **2024:** 64 selected, 16 HR, **25.00%** vs 10.80% controls; lift **+14.20 pp**; 95% CI **+4.78 to +24.54 pp**

Direction is positive and the yearly clustered intervals stay above zero in both seasons despite the smaller 2024 sample.

### Context fingerprint in the deepest disagreement band

For obvious-rank 17+ hitters, mean within-slate context percentiles were:

| Context group | Full73 selected | Same-band controls | Difference |
|---|---:|---:|---:|
| park | **73.89%** | 49.72% | **+24.17 pp** |
| recent batter form | **73.04%** | 48.08% | **+24.96 pp** |
| pitcher vulnerability | **58.67%** | 50.19% | **+8.47 pp** |
| pitch matchup | **54.79%** | 49.82% | **+4.97 pp** |

This is consistent with what the model was built to do: promote a batter whom long-horizon power alone does not make obvious when the **current environment, recent contact quality, and opposing pitcher context** improve the HR setup.

### Important remaining confound

The same-band control is materially stronger than comparing against the whole slate, but it is not yet perfectly rank-matched. For example, a selected hitter with obvious-power rank 18 sits in the 17+ band alongside control hitters ranked 80 or 150. Some of the 17+ lift could therefore come from full73 selecting the better end of the long-horizon tail rather than from contextual information alone.

**Do not promote a production hidden-pick rule until a same-slate nearest-rank matched test is run.** The next test pairs each full73 promotion with an unselected hitter of the nearest available obvious-power rank on the same date, without replacement, and uses paired binary-outcome inference plus slate-date bootstrap uncertainty.

## Context fingerprint of differentiated full73 picks

Within-day mean percentile scores:

| Context group | All hitters | Shared top4 | Full73-only | Obvious-only |
|---|---:|---:|---:|---:|
| park | 50.21% | 70.98% | **69.93%** | 58.92% |
| recent batter form | 50.21% | 86.12% | 77.89% | **84.17%** |
| pitcher vulnerability | 50.21% | 55.29% | **57.17%** | 46.92% |
| pitch matchup | 50.21% | 59.34% | 55.72% | **57.66%** |

The initial fingerprint suggests the model's differentiated promotions are characterized more by **park and pitcher vulnerability** than by simply stronger long-horizon batter power. In the deepest 17+ disagreement band, recent form becomes a much stronger differentiator as well.

These fingerprints are descriptive, not causal. They are being used to understand what the model is recognizing, not to create post-hoc manual betting filters.

## Current verdict

1. **Full73 remains the model architecture.** Equal retuning and clustered sensitivity do not support pruning to 62 features.
2. **The model has a statistically resolved advantage over a batter-only obvious-power proxy in the daily top 5%.** This remains the broadest defensible shine zone.
3. **Exact top-4/day replacement is not supported.** The simple power proxy is slightly better overall there, with substantial 2023/2024 regime variation.
4. **The model has statistically resolved differentiated discovery in the predeclared 9-16 and 17+ obvious-power rank bands after Holm multiplicity correction.** The 17+ development signal is especially large and positive in both years.
5. **A nearest-rank same-slate match is still required before treating deep disagreement as an operational hidden-pick rule.** This explicitly tests whether the signal survives controlling the residual rank-position confound inside the broad bands.
6. 2025 remains sealed.
