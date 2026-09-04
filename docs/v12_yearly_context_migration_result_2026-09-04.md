# v1.2 yearly contextual-migration result — 2026-09-04

## Scope

This records the predeclared continuous migration test from `docs/v12_yearly_context_migration_precommit_2026-09-04.md`.

Question:

> Across successive seasons, does the frozen full73 model's contextual top-5% selection migrate systematically toward or away from the long-horizon batter-only obvious-power ranking?

No fixed 5–8 / 9–16 / 17+ bucket thresholds were tested. No calendar-month outcome table was emitted.

**2025 was not read or scored and remains sealed.**

## Authoritative reproducible run

- GitHub Actions run: **33893462389**
- exact head: **9bba0c86fefb2c472495bbf799648c26b224c827**
- job: **101090445748**
- conclusion: **SUCCESS**
- artifact: `v12-yearly-context-migration`
- artifact ID: **9944901186**
- artifact SHA256: **210f24fce7dad62806b8b3ec1780b719be6f6d71bad9189db5faff9e335d27f3**

Synthetic positive/flat migration controls passed before real inference. The trusted 406,728-row / 73-feature matrix was rebuilt from repo inputs. Every target year was scored by a model trained only through year-1 with the frozen champion hyperparameters and 194 boosting rounds.

Walk-forward target years:

- 2019: train 2015–2018
- 2020: train 2015–2019
- 2021: train 2015–2020
- 2022: train 2015–2021
- 2023: train 2015–2022
- 2024: train 2015–2023

No yearly tuning or calibration was performed; rankings use raw XGBoost probability.

## Primary yearly migration table

`obvious depth` is the continuous within-slate rank percentile of a full73-only top-5% selection under the obvious-power proxy. Larger values mean the contextual model reached farther away from obvious-power.

| Year | Full73 top-5 n | Shared with obvious top-5 | Overlap | Full73-only n | Mean obvious depth | Median obvious rank | Full73-only HR rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 2,218 | 1,165 | **52.52%** | 1,053 | **14.68%** | 32 | 22.32% |
| 2020 | 838 | 448 | **53.46%** | 390 | **16.17%** | 28 | 18.46% |
| 2021 | 2,270 | 1,337 | **58.90%** | 933 | **11.85%** | 24 | 22.62% |
| 2022 | 2,275 | 1,329 | **58.42%** | 946 | **12.46%** | 25 | 19.13% |
| 2023 | 2,252 | 1,381 | **61.32%** | 871 | **11.03%** | 22 | 21.70% |
| 2024 | 2,220 | 1,404 | **63.24%** | 816 | **11.75%** | 22 | 19.36% |

The broad direction is the opposite of the hypothesis suggested by the sparse 2022 versus 2023–24 disagreement-bucket snapshots:

> **Over 2019–2024, the frozen contextual model is converging toward the obvious-power board, not systematically migrating farther away from it.**

There is year-to-year noise in depth, but the longer-run convergence is strong in both continuous depth and top-5 overlap.

## Continuous depth trend

Observed mean-depth slope:

- **-0.008422 rank-percentile units per year**
- equivalently about **-0.842 percentage points of obvious-rank depth per year**
- Spearman(year, yearly mean depth): **-0.8286**
- adjacent depth signs: `+ - + - +`

10,000 paired slate-date bootstrap replicates:

- bootstrap mean slope: **-0.008433/year**
- median: **-0.008428/year**
- 95% CI: **[-0.010198, -0.006693] / year**
- `P(slope > 0) = 0.0000`

The yearly path is not monotonic one step at a time, but its longer-run direction is strongly negative.

## Top-5 overlap trend

Observed overlap slope:

- **+0.021914 per year**
- equivalently about **+2.19 percentage points of overlap per year**
- Spearman(year, yearly overlap): **+0.9429**
- adjacent overlap signs: `+ + - + +`

10,000 paired slate-date bootstrap replicates:

- bootstrap mean slope: **+0.021908/year**
- median: **+0.021882/year**
- 95% CI: **[+0.017578, +0.026401] / year**
- `P(slope > 0) = 1.0000`

This complementary metric gives the same answer as depth: contextual and obvious-power top-5 boards have become progressively more similar across the historical sequence.

## Predeclared sensitivity excluding 2020

The result is not an artifact of the shortened 2020 season.

Excluding 2020:

- mean-depth slope: **-0.006091/year**
- depth Spearman: **-0.8000**
- bootstrap 95% CI: **[-0.007382, -0.004771]**
- overlap slope: **+0.020418/year**
- overlap Spearman: **+0.9000**
- overlap bootstrap 95% CI: **[+0.016010, +0.024828]**

The same convergence appears without 2020.

## Successful contextual selections

The average obvious-depth of the full73-only hitters who actually homered was:

| Year | Successful full73-only n | Mean obvious depth | Median obvious depth |
|---|---:|---:|---:|
| 2019 | 235 | 13.98% | 11.52% |
| 2020 | 72 | 13.62% | 9.86% |
| 2021 | 211 | 11.83% | 8.92% |
| 2022 | 181 | 12.43% | 9.29% |
| 2023 | 189 | 10.22% | 8.12% |
| 2024 | 158 | 12.42% | 8.89% |

This secondary outcome-weighted view also does **not** show successful contextual HRs progressively moving deeper into the obvious-power board. It is noisier, but if anything it trends shallower across the six-year sequence.

No hidden-rank production rule is inferred from these outcome-weighted diagnostics.

## How the model promotes deeper hitters

The same contextual feature-group composites used in the prior edge-localization work were evaluated among full73-only top-5 selections. These are model-mechanism diagnostics, not independent confirmation of edge.

Overall Spearman association with obvious-rank depth among differentiated selections:

- **park:** `+0.1481`
- **pitcher vulnerability:** `+0.0853`
- **pitch matchup:** `-0.0009`
- **recent batter form:** `-0.2581`

Interpretation:

1. **Park is the clearest positive marker of deeper promotions.** When full73 reaches farther away from the obvious-power board, favorable park context tends to be stronger.
2. **Pitcher vulnerability is a weaker positive marker** of deeper promotions.
3. The broad pitch-matchup composite is essentially uncorrelated with how deep the promoted hitter was in obvious rank.
4. **Recent batter form is negatively associated with depth.** Strong recent form tends to accompany contextual selections that are already closer to the obvious-power region rather than the deepest promotions.

This is a useful refinement of the earlier plausibility story: the model's deepest deviations from long-horizon batter power are more associated with **environment/opponent context (especially park, then pitcher vulnerability)** than with simply chasing a hot recent hitter.

### Yearly mechanism levels

Mean context percentile among full73-only selections:

| Year | Park | Recent batter form | Pitcher vulnerability | Pitch matchup |
|---|---:|---:|---:|---:|
| 2019 | 63.67% | 73.67% | 55.56% | 55.02% |
| 2020 | 70.42% | 67.67% | 56.52% | 54.13% |
| 2021 | 67.72% | 73.64% | 57.75% | 54.39% |
| 2022 | 72.45% | 69.77% | 57.03% | 52.41% |
| 2023 | 66.51% | 73.04% | 56.62% | 55.69% |
| 2024 | 68.23% | 71.79% | 56.53% | 56.15% |

There is no simple feature-family migration that explains the entire convergence. Park remains strongly elevated in differentiated picks every year; recent form remains very elevated too, while pitcher and pitch-matchup composites are more moderate.

## Relationship to the prior disagreement-bucket result

The prior matched disagreement experiment found:

- 2022: the 5–8 matched band looked strongest;
- 2023–24: the 17+ point estimate looked strongest but failed the precommitted matched CI gate and did not replicate in 2022.

Those sparse-band snapshots suggested a possible drift toward deeper disagreement.

The continuous six-season test rejects that interpretation as a general structural trend. The broader model behavior is instead:

- higher top-5 overlap with obvious power over time;
- shallower average obvious rank among the remaining differentiated picks.

Therefore **do not make a prospective 2025 prediction that the signal should continue migrating toward 17+ or deeper ranks.** That hypothesis did not survive the broader historical test.

## What this does and does not mean

This result does **not** mean the contextual features stopped adding value. The previously resolved 2023–24 daily top-5% comparison remains:

- full73: **21.69%**
- obvious-power: **20.28%**
- lift: **+1.41 pp**
- paired 95% CI: **+0.42 to +2.39 pp**

And the half-season test found essentially the same contextual lift in both halves.

The migration result instead says that the way the contextual model earns that broad edge is **not becoming progressively more contrarian** relative to obvious long-horizon power. Over time the two top-5 boards have become more similar.

That may be a sign that the model is increasingly differentiating *within* the obvious-power neighborhood rather than discovering an ever-deeper sleeper population.

## Decision

1. The original migration hypothesis was worth testing and has a clear answer.
2. There is strong historical evidence of **convergence toward obvious-power**, not migration away from it, from 2019–2024.
3. The result survives the predeclared ex-2020 sensitivity.
4. No fixed disagreement-rank bucket or dynamic depth policy is justified.
5. Park and, more weakly, pitcher vulnerability are the strongest contextual correlates of the model's genuinely deeper promotions; recent batter form is associated with shallower differentiated promotions.
6. Do not mine additional rank thresholds or monthly subgroups from <=2024.
7. This completes the planned structural development diagnostic before the pre-2025 reset/review.
8. **2025 remains sealed.**
