# v1.2 dynamic bucket regime precommit — 2026-09-04

## Purpose

Test whether the full73 contextual model's disagreement-bucket advantage is **time-varying but predictable**, rather than assuming one permanent hidden-pick bucket.

This experiment is motivated by the observed instability of fixed obvious-power disagreement bands across development periods and by baseball domain structure: run-scoring/contact environments can differ materially between early season and warmer/later-season periods.

**2025 remains sealed and must not be read, scored, or used in any design decision until this experiment is frozen.**

## Important non-failure rule

A weak or null signal in April/May is **not, by itself, grounds to reject the regime hypothesis**.

The experiment must distinguish:

1. true inability to predict future bucket strength; from
2. insufficient accumulated in-season information; and
3. a season phase in which contextual HR edge is genuinely compressed.

No requirement is imposed that the same bucket be actionable from Opening Day onward.

## Frozen candidate bands

Use the already-declared obvious-power disagreement bands only:

- 5–8
- 9–16
- 17+

Do not invent new rank cutoffs after observing outcomes.

A fourth valid state is:

- **NO_ACTIONABLE_BUCKET**

The regime selector is allowed to abstain.

## Historical evaluation architecture

Use only seasons <= 2024.

For each scored season Y, the base full73 ranking model must be trained only on seasons < Y. No same-season outcome may enter the model before scoring that season.

The regime layer must itself be walk-forward: information available through date D may only predict a future block beginning after D.

## Season-phase treatment

Do not treat calendar months as interchangeable observations.

At minimum report results by the following predeclared phase labels:

- EARLY: Opening Day through May 31
- MID: June 1 through July 31
- LATE: August 1 through end of regular season

These are reporting strata, not outcome-chosen cutoffs.

Also retain continuous season progress (`days_since_opening_day`, `pct_regular_season_elapsed`) so later tests are not forced to assume discontinuities at month boundaries.

## First-pass question: persistence before prediction

Before building any meta-model, determine whether bucket strength has temporal structure.

For each band, compute matched contextual lift over trailing windows ending at D:

- trailing 30 days
- trailing 60 days
- trailing 90 days
- season-to-date

Then measure lift in the next non-overlapping 30-day block.

Primary first-pass statistics:

- correlation between trailing lift and next-block lift
- sign persistence probability
- rank persistence: whether the strongest trailing band remains strongest next block
- abstention value: whether low-information/low-separation periods are better treated as NO_ACTIONABLE_BUCKET

Do not declare failure solely because 30-day early-season windows are noisy. Evaluate longer accumulation windows and phase-specific reliability as predeclared above.

## Minimal regime selector

Only if persistence exists, test a simple selector before any complex ML meta-model.

Candidate selector inputs are restricted initially to:

- trailing 30/60/90-day matched lift by the three frozen bands
- season-to-date matched lift by band
- number of matched observations accumulated by band
- uncertainty width / lower confidence bound by band
- season progress
- league-wide HR rate to date
- league-wide barrel/contact-quality environment to date if available pregame and leakage-safe

No player identities, future outcomes, sportsbook prices, or 2025 data.

The simple selector chooses one of 5–8 / 9–16 / 17+ / NO_ACTIONABLE_BUCKET for the next 30-day block.

## Evaluation target

The selector is not judged on whether it always names a bucket.

Primary target:

> When the selector chooses a bucket, does that chosen bucket produce positive future matched lift with a practically meaningful magnitude?

Report:

- coverage (% of future blocks with a chosen bucket)
- selected-block mean/median lift
- pooled matched HR-rate lift
- slate/date-cluster bootstrap CI
- hit rate of choosing the actual best future band
- regret versus hindsight-best band
- performance of NO_ACTIONABLE_BUCKET abstention periods
- EARLY / MID / LATE separately

## Practical magnitude convention

For regime evaluation, uncertainty remains primary evidence and magnitude remains a usefulness screen.

Do not require every future block to clear +5 pp individually; that would be unrealistic for noisy 30-day HR samples.

Instead:

- inferential question: does pooled forward-selected lift clear zero with clustered uncertainty?
- usefulness question: is the pooled magnitude large enough to matter for high-variance HR props?

The prior +5 pp matched-lift threshold remains a reference point for a strong hidden-pick effect, but it is **not** imposed as a per-month binary failure criterion.

## Promotion rule before 2025

A dynamic regime policy may be frozen for the 2025 one-shot only if:

1. it is fully specified using <=2024 data;
2. it is walk-forward with no same/future-block leakage;
3. it demonstrates forward predictive value beyond simply choosing the historically strongest bucket;
4. performance is not dependent on one isolated month/year;
5. EARLY-season weakness alone does not invalidate it if MID/LATE forward behavior is reproducible and the selector appropriately abstains or reflects low confidence early;
6. all cutoffs, windows, and abstention logic are frozen before 2025 is opened.

If these conditions are not met, no dynamic bucket policy is promoted; the production interpretation remains the broader full73 top-tail ranking advantage only.

## 2025 / 2026 sequencing

1. Finish and freeze this <=2024 regime experiment.
2. Run the already-preserved 2025 one-shot before inspecting 2026 outcomes for model-policy development.
3. Only after 2025 evaluation is complete may 2026 be used as an additional forward confirmation / live-season study.

## Current status

This document freezes the initial dynamic-regime hypothesis and protects against declaring early-season null periods a failure merely because baseball offensive conditions and information accumulation differ across the season.

No regime result has been observed yet.
