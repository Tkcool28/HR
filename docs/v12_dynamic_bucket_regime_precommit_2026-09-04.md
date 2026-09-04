# v1.2 seasonality / dynamic-regime precommit — 2026-09-04

## Purpose

Test whether the frozen full73 contextual model's already-resolved **daily top-5% advantage over the long-horizon obvious-power proxy** changes materially by season phase before attempting any lower-powered disagreement-bucket regime model.

The disagreement bands (5–8 / 9–16 / 17+) remain scientifically interesting, but their matched samples are too sparse for a first-pass month-by-month regime analysis. The first seasonality instrument is therefore the broader daily top-5% comparison, which already has materially tighter full-sample uncertainty.

**2025 remains sealed and must not be read, scored, or used in any design decision until this <=2024 experiment is frozen.**

## Existing anchor result

Frozen 2023–24 development evidence:

- full73 daily top 5% HR rate: **21.69%**
- obvious-power daily top 5% HR rate: **20.28%**
- paired lift: **+1.41 percentage points**
- 95% paired slate-date bootstrap CI: **+0.42 to +2.39 pp**

This broad comparison is the primary powered instrument for the first seasonality test.

## Primary predeclared split

Use exactly one primary phase split before looking at phase outcomes:

- **FIRST_HALF:** Opening Day through June 30
- **SECOND_HALF:** July 1 through end of regular season

These dates are frozen before phase results are observed. Do not optimize the cutoff after seeing outcomes.

Why this split:

1. it preserves much more sample than six individual calendar-month cells;
2. it corresponds approximately to an early/cooler/lower-information portion and a later/warmer/more-mature in-season-information portion without claiming those mechanisms are proven;
3. it permits a later month-level follow-up only if the broad half-season contrast itself provides evidence worth localizing.

Calendar-month results are **not** part of the primary test and must not be inspected as a fishing pass in the same analysis.

## Primary question

For each half separately, compare the frozen full73 and frozen obvious-power proxy using **daily raw-score top 5% selection** with the same deterministic tie handling already used in the resolved top-tail analysis.

Report for FIRST_HALF and SECOND_HALF:

- number of slate dates
- number of selected batter-games for each method
- full73 HR rate
- obvious-power HR rate
- paired absolute lift in percentage points
- 10,000-replicate paired slate-date bootstrap 95% CI
- probability paired lift > 0

Also estimate the predeclared interaction:

> `SECOND_HALF paired lift - FIRST_HALF paired lift`

using a season/year-aware clustered bootstrap. Report its 95% CI. This interaction, not whichever half has the prettier standalone number, is the direct test of whether contextual advantage differs by season phase.

## Magnitude / evidence convention

Do not introduce a new arbitrary +5 pp floor for the broad top-5% phase test; that threshold was designed for rare hidden-pick disagreement effects and is inappropriate for a broad ranking lift whose full-sample effect is ~+1.4 pp.

For this phase experiment:

- **Evidence question:** does the paired CI within a half clear zero, and does the half-vs-half interaction have a CI that materially excludes zero?
- **Usefulness question:** is the estimated half-specific lift large enough relative to the existing +1.41 pp full-sample anchor to matter operationally?

No binary production rule is promoted solely because one half's point estimate is positive.

## Critical feature-maturity audit

A phase difference must not automatically be interpreted as baseball skill seasonality.

The model includes recent-form / recent-pitcher features whose reliability can vary with accumulated history. Before interpreting any phase effect, report by FIRST_HALF vs SECOND_HALF (and by year where practical):

1. missing / NaN rate before model imputation for every active feature containing `14d` or `30d`;
2. any explicit rolling-history support/count variables available in the feature surface (for example PA counts or pitch-count support) summarized by median / p10 / p25;
3. fraction of rows with effectively low history under predeclared support thresholds where a natural denominator exists;
4. same diagnostics for the top-5%-selected rows, not just the full slate;
5. overall HR base rate by half.

Interpretation order is frozen:

1. first establish whether a top-5% phase interaction exists;
2. then check whether recent-feature maturity/data completeness changes in the same direction;
3. only then discuss plausible baseball mechanisms such as weather, pitcher/hitter adjustment, accumulated scouting familiarity, or changing run environment.

The feature-maturity table is a **confound/measurement diagnostic**, not independent confirmation of predictive edge.

## Early-season non-failure rule

A weak or null FIRST_HALF contextual lift is **not by itself a failure of the full73 model**.

It may reflect:

- a genuinely compressed contextual HR environment;
- lower reliability of recent-form inputs;
- lower accumulated same-season evidence;
- or ordinary sampling uncertainty.

The primary model remains judged by its resolved full-season/full-development top-tail evidence. This experiment asks only whether that edge is phase-dependent.

## Follow-up gate

Only if the primary half-season analysis shows a meaningful phase difference or a clearly stronger phase with adequate sample may we perform one separately declared localization follow-up.

Permissible follow-ups must be precommitted **after** recording the half-season result and **before** viewing finer outcomes. Candidates include:

- calendar-month top-5% lift within the phase that showed heterogeneity;
- rolling 60/90-day top-5% lift persistence;
- later, a dynamic selector using only sufficiently powered top-tail metrics.

Do **not** return immediately to monthly 5–8 / 9–16 / 17+ disagreement-bucket cells. Those require substantially more accumulated evidence or a different hierarchical/partial-pooling design.

## Relationship to dynamic disagreement buckets

The long-term hypothesis remains that contextual advantage may move among types of promoted hitters over time. This document changes the order of attack:

1. establish whether the **powered broad top-5% contextual lift** is season-phase dependent;
2. audit whether any difference could be explained by feature maturity;
3. only with positive evidence, investigate finer temporal localization;
4. only after sufficient support, revisit whether disagreement-bucket location itself is predictable.

A valid future state remains **NO_ACTIONABLE_BUCKET / no special subgroup rule**. We do not require a hidden bucket to exist in every period.

## Historical evaluation architecture

Use only seasons <= 2024.

The immediate primary split is evaluated on the already-frozen 2023–24 full73-vs-obvious-power evidence surface so model architecture and comparator do not change.

Any later walk-forward regime model must train its base ranking model only on seasons preceding the scored season and must use only information available before the future prediction block.

## 2025 / 2026 sequencing

1. Finish and freeze this <=2024 seasonality experiment.
2. Freeze any justified follow-up or explicitly freeze **no dynamic policy**.
3. Run the preserved 2025 one-shot before inspecting 2026 outcomes for model-policy development.
4. Only after 2025 evaluation may 2026 be used as an additional forward confirmation/live-season study.

## Current status

This revision was committed **before any FIRST_HALF / SECOND_HALF top-5% result was inspected**.

The first powered test is now the broad daily top-5% full73-vs-obvious comparison, not monthly disagreement-band mining.

2025 remains sealed.
