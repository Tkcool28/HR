# V1.2 2019–2025 Full Picture + Historical HR Odds Next Step

Date: 2026-09-04
Branch: `feat/v1.2-trustworthy-rebuild`

## Scope and interpretation guardrail

This is a post-holdout retrospective summary. The years are **not equally independent**.

- 2019–2021: chronological walk-forward tuning-fold years; their outcomes participated in hyperparameter selection.
- 2022: chronological calibration/freshness year; not pristine.
- 2023–2024: development-assessment years repeatedly used for architecture/feature decisions.
- 2025: sealed one-shot holdout, frozen before outcome access; base fit 2015–2023, calibration 2024.

For the 2019–2024 year-by-year diagnostic, the Full73 architecture, champion hyperparameters, and 194 rounds are fixed, while the training set expands only through year-1. This is a stability/shape diagnostic, not seven independent holdouts.

## Obvious-power comparator consistency audit

A temporary concern arose after the holdout because the historical helper source nominally lists `batter_hr_per_pa_season/career` among preferred obvious-power fields. Those names do not exist in the delivered active matrix. The actual historical artifact records the runtime comparator as exactly these five fields:

1. `batter_barrel_rate_season`
2. `batter_barrel_rate_career`
3. `batter_xwoba_on_contact_season`
4. `batter_xwoba_on_contact_career`
5. `batter_avg_ev_season`

The sealed 2025 evaluator used these same five fields. The artifact recovery reproduced the locked 2025 primary result exactly:

- Full73 top 5%: 515 / 2,276 = 22.6274%
- obvious-power top 5%: 482 / 2,276 = 21.1775%
- Full73 minus obvious-power: +1.4499 pp

Verdict: **CONSISTENT_NO_COMPARATOR_CORRECTION_REQUIRED**. No Full73 result or comparator result changed.

## Year-by-year walk-forward/full-holdout picture

| Year | Role | Base HR | Raw AUC | Full73 top 5% | Obvious top 5% | Δ pp | Top 1/day | Top 2/day | Top 4/day | Top 8/day |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | tuning fold | 13.74% | .6359 | 24.89% | 24.44% | +0.45 | 22.91% | 25.42% | 24.30% | 25.49% |
| 2020 | tuning fold, shortened | 12.96% | .6162 | 21.00% | 21.00% | +0.00 | 20.90% | 23.13% | 22.39% | 23.13% |
| 2021 | tuning fold | 12.13% | .6491 | 22.91% | 21.37% | +1.54 | 32.42% | 25.55% | 25.00% | 23.15% |
| 2022 | calibration/freshness | 10.91% | .6231 | 20.57% | 19.30% | +1.27 | 26.82% | 21.51% | 22.07% | 22.00% |
| 2023 | development assessment | 12.27% | .6171 | 22.34% | 20.16% | +2.18 | 21.67% | 25.83% | 24.86% | 23.61% |
| 2024 | development assessment / later calibration | 11.40% | .6192 | 21.67% | 20.32% | +1.35 | 27.37% | 26.26% | 24.16% | 22.21% |
| **2025** | **sealed one-shot holdout** | **11.73%** | **.6253** | **22.63%** | **21.18%** | **+1.45** | **28.80%** | **29.89%** | **25.82%** | **23.10%** |

Key read:

- Full73 top 5% is above the obvious-power top 5% in six of seven seasons and ties in shortened 2020.
- The 2025 +1.45 pp contextual lift is not an anomalous holdout spike; it sits comfortably inside the historical pattern.
- The sealed 2025 top-four hit rate (25.82%) is consistent with, and slightly stronger than, the broader historical top-tail behavior.
- Because the older years have different development roles, this table is evidence of stability rather than a seven-fold independent significance claim.

## Exact rank sanity check

Pooled exact daily ranks across 2019–2025 (1,150 slate-days/picks at each exact rank):

| Exact model rank | HR rate |
|---:|---:|
| #1 | 26.35% |
| #2 | 24.87% |
| #3 | 24.09% |
| #4 | 21.74% |
| #5 | 22.43% |
| #6 | 24.17% |
| #7 | 20.17% |
| #8 | 22.17% |

Exact ranks are not perfectly monotonic; there are three adjacent inversions among ranks 1–8. That is not hidden. However, the broad ordering is downward (Spearman rank-vs-HR-rate = -0.714), and the **cumulative product cutoffs are monotonic**:

- top 1/day: 26.35%
- top 2/day: 25.61%
- top 4/day: 24.26%
- top 8/day: 23.25%

Thus the 2025 #2-over-#1 inversion is a local realized-season fluctuation rather than a pooled reversal of the model's highest-rank ordering. The #5/#6 and #7/#8 wiggles are useful reminders not to interpret exact rank as a perfectly calibrated ordinal ladder.

## Evidence artifact

Final post-holdout audit:

- workflow run: `33913327921`
- artifact: `v12-2019-2025-full-picture-final`
- artifact ID: `9952197193`
- artifact SHA256: `2376f46c715464f0451f96f8ec45f853ef847ad5adf39e71177e7785290279ee`
- exact-head commit for successful workflow: `3c94ba37e27c2be194f15180ea2958e860a000b5`

The workflow verified both source artifacts by SHA256 and reproduced the frozen 2025 Full73 and obvious-power results before emitting this summary.

## Historical HR-prop odds source — next phase

Preferred long-history candidate: **SportsDataIO Historical Odds / Betting Data Archive**.

Why it fits this project:

- MLB player props are archived by season, betting market, and sportsbook group.
- Historical prop responses include betting outcomes plus line movement, matching their production BettingMarket schema.
- SportsDataIO explicitly supports MLB `To Hit a Home Run` player props.
- Their historical product page states props/futures coverage from 2020 and MLB historical odds/game coverage from 2019.
- Their 2021 MLB release specifically documents full player/team-prop coverage for FanDuel, DraftKings, PointsBet, BetMGM, and William Hill for that season.
- Their MLB workflow documentation says pregame props can include opening price, all line movements, timestamps, and closing price.

Important procurement guardrail: historical coverage varies by year and data point. Before purchasing access, require a sample or explicit coverage confirmation for **MLB To Hit a Home Run** by season and sportsbook, especially 2020–2025 and preferably DraftKings/FanDuel/BetMGM/Caesars or predecessor William Hill.

Secondary/self-service source: **The Odds API**.

- Supports MLB `batter_home_runs`.
- Historical player-prop snapshots are available from 2023-05-03 onward at five-minute intervals.
- Good candidate for an immediate 2023–2025 independent benchmark, but insufficient for the full 2019/2020–2022 archive.

Proposed market-benchmark principle: treat sportsbook odds as an **external predictive benchmark**, not a Full73 model feature and not a retroactive development target. Compare ranking/predictive information first; ROI and price/value analysis comes afterward.
