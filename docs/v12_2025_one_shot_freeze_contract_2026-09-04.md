# v1.2 sealed 2025 one-shot freeze contract — 2026-09-04

## Authorization state

The user explicitly authorized opening and evaluating the previously sealed 2025 season on 2026-09-04, after completion of the <=2024 development reset review.

This document freezes the model, fit, selector, comparator, inference, and reporting contract **before any 2025 outcome is acquired, read, scored, or inspected**.

Historical sportsbook odds remain out of scope until the 2025 predictive result is locked. 2026 remains out of scope until the 2025 result is locked.

---

## 1. Frozen model architecture

The final v1.2 holdout model is the existing **full73 aggressive champion**.

Frozen architecture:

- exactly **73 active numeric features** from the trusted v1.2 feature contract;
- no feature additions, removals, redefinitions, or family pruning;
- champion XGBoost hyperparameters from `experiments/contracts/v12_full73_aggressive.json`;
- exactly **194 boosting rounds**;
- raw XGBoost probability is the canonical ranking score;
- isotonic calibration may affect probability/calibration metrics only and must never alter top-k membership.

No Optuna run, retuning, early-stopping search, architecture selection, threshold search, or feature selection may occur after 2025 is opened.

---

## 2. Frozen final pre-2025 fit

The one-shot production-like fit is:

- XGBoost base training years: **2015–2023**;
- all imputation statistics fit from **2015–2023 only**;
- isotonic calibration year: **2024 only**;
- 2025 contributes nothing to fitting, imputation, calibration, model selection, or selector definition.

The 73-feature list and hyperparameters/round count are frozen from <=2024 development and cannot change in response to 2025 results.

---

## 3. 2025 input/data contract

2025 must be rebuilt through the same trusted historical semantics, extended by one target year rather than through a bespoke holdout-only feature implementation.

Required rules:

- MLB regular season only (`gameType=R` / equivalent regular-season filter);
- authoritative MLB game identity and actual `venue_id` per game;
- target universe remains the first nine distinct batters per side in historical appearance order, exactly 18 target rows per valid game;
- starting pitchers use the same trusted historical reconstruction semantics;
- all rolling/career/season/pitch-type/QoC features must be strictly pregame (`max_as_of_date < game_date`);
- 2025 target rows may use historical observations through the day before each 2025 game, including earlier 2025 games when the feature definition is rolling/season-to-date;
- no future 2025 observation may enter a target row;
- 2025 park factor for target season 2025 must use only completed prior seasons **2022–2024** under the frozen three-year-prior shrinkage method;
- postseason/exhibition/spring-training data are excluded;
- actual 2025 HR outcomes are used only after feature construction for scoring/evaluation.

The acquisition/build implementation may be repaired if a genuine technical incompatibility is found, but any repair must preserve these already-frozen semantics and must not be informed by whether the 2025 model result improves or worsens.

---

## 4. Frozen obvious-power comparator

The comparison benchmark remains the already-defined long-horizon batter-only obvious-power proxy. It is **not** sportsbook consensus.

Use the same existing feature set/definition as the <=2024 edge-localization work, with no 2025-driven modification.

The proxy is ranked within each daily slate using the same deterministic tie-breaking convention as before.

---

## 5. Primary 2025 product test

The primary question is:

> Does the frozen full73 model retain useful daily top-5% HR concentration on the untouched 2025 season, and how does that candidate board compare with the frozen obvious-power board?

Primary selector:

- rank all eligible batter-games within each `game_date` by **raw full73 score**;
- select `ceil(0.05 * slate_size)`, minimum 1;
- deterministic outcome-independent tie breakers: `game_pk`, then `batter_id`.

Report for 2025:

- number of slate dates;
- number of selected batter-games;
- full73 daily top-5% HR rate;
- obvious-power daily top-5% HR rate;
- absolute lift: full73 minus obvious-power;
- paired **10,000-replicate slate-date bootstrap** 95% CI for the lift;
- `P(lift > 0)` from the paired bootstrap;
- overall 2025 target-universe HR base rate and top-5 lift versus base rate.

Interpretation is descriptive/confirmatory rather than ROI based:

- CI wholly above zero: resolved positive contextual lift over obvious-power;
- positive point estimate with CI crossing zero: directional but unresolved in one holdout season;
- zero/negative point estimate: no positive 2025 top-5 advantage over obvious-power.

No magnitude threshold is introduced for this primary broad-board test after seeing 2025. Market usefulness is a later sportsbook-price question.

---

## 6. Frozen secondary 2025 diagnostics

Report all of the following regardless of whether they look good or bad:

### Model-quality metrics

- raw Brier;
- calibrated Brier;
- raw ROC AUC;
- calibrated ROC AUC;
- raw AP;
- calibrated AP if supported by the harness;
- raw/calibrated log loss;
- ECE;
- calibration-in-the-large / calibration slope where supported without changing the frozen predictor.

### Raw daily ranking metrics

- daily top 10%;
- daily top 5%;
- daily top 2%;
- daily top 1%;
- top 1/day;
- top 2/day;
- top 4/day;
- top 8/day.

These are secondary diagnostics. None may replace the predeclared top-5% primary question after results are seen.

---

## 7. Secondary 2025 structural readout

Only after the primary holdout metrics are computed and written may the executor report the already-defined structural diagnostics:

- full73/obvious-power top-5 overlap;
- mean/median continuous obvious-rank depth of full73-only top-5 selections;
- whether 2025 continues or breaks the 2019–2024 historical convergence direction.

This is descriptive only.

Do **not** create, test, or promote new 5–8 / 9–16 / 17+ rules, new rank thresholds, month cuts, season-phase cuts, player archetypes, feature thresholds, or any other 2025-derived subgroup.

---

## 8. Full historical picture after the holdout is locked

After the 2025 one-shot result is saved unchanged, produce a retrospective summary spanning the available out-of-time sequence.

At minimum report 2019–2025 by year for:

- target-universe base HR rate;
- full73 daily top-5% HR rate;
- obvious-power daily top-5% HR rate where available under the same comparator;
- contextual lift;
- full73 daily top-4 and other frozen tail metrics;
- top-5 overlap with obvious-power;
- continuous differentiated-selection depth;
- sample/slate counts.

Historical rows must retain the already-frozen walk-forward semantics. The 2025 row uses the final pre-holdout production-like fit described above. Do not refit earlier years using future data simply to make the table cosmetically uniform.

The combined table is a perspective/robustness summary, not a second opportunity to optimize rules.

---

## 9. Explicit prohibited actions

Once 2025 acquisition begins, do not:

- change the 73-feature architecture;
- change champion hyperparameters or 194 rounds;
- retune on 2025;
- choose a new calibration year after seeing 2025;
- alter the obvious-power feature set;
- alter daily selector sizing;
- create a new subgroup to rescue a weak primary result;
- omit an unfavorable frozen secondary metric;
- use sportsbook odds to modify the predictor or evaluation before the predictive holdout verdict is locked;
- inspect 2026 to reinterpret or repair the 2025 result before it is frozen.

A genuine code/data defect may be corrected only if the correction is semantically required independent of result direction. The defect, correction, and before/after execution state must be documented.

---

## 10. Required evidence artifacts

The one-shot workflow must persist enough evidence to reproduce/audit the result:

- exact Git commit SHA;
- frozen contract copy/hash;
- active feature-list hash and count = 73;
- raw 2025 input acquisition metadata / row counts;
- trusted target/matrix structural summary;
- final model-fit metadata (2015–2023, 194 rounds);
- 2024 calibration metadata;
- 2025 raw and calibrated predictions;
- frozen obvious-power scores;
- primary paired-bootstrap JSON;
- complete secondary metrics JSON/CSV;
- full 2019–2025 summary table;
- explicit statement that no post-hoc tuning/subgroup selection occurred.

---

## 11. Authorization and state transition

User authorization to execute the 2025 holdout was received in-chat on 2026-09-04 after the pre-2025 development reset.

State after this contract commit, before 2025 acquisition:

> **V1.2_FULL73_FROZEN_AND_AUTHORIZED_FOR_SINGLE_2025_HOLDOUT_EXECUTION**

The next operation is implementation validation of the historical raw-input interface followed by the single frozen 2025 acquisition/build/score/evaluate workflow.
