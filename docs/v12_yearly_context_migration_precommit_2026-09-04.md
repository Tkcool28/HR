# v1.2 yearly contextual-migration precommit — 2026-09-04

## Question

Before opening the sealed 2025 holdout, test the user's original structural hypothesis:

> Across successive seasons, does the frozen full73 model's contextual top-tail selection migrate systematically **toward or away from** the obvious long-horizon batter-power ranking?

This is not a search for another fixed 5–8 / 9–16 / 17+ bucket. The primary migration variable is continuous so no rank threshold is chosen to fit development outcomes.

**2025 must not be read or scored in this experiment.**

## Historical sequence

Score seasons **2019, 2020, 2021, 2022, 2023, 2024**.

For each target season Y:

- use the frozen trusted **73 active features**;
- use the frozen full73 champion XGBoost hyperparameters;
- use the frozen **194 boosting rounds**;
- refit on all available trusted seasons **2015 through Y-1** only;
- compute imputation means from that training slice only;
- score Y with raw XGBoost probability;
- no yearly retuning, early stopping, feature selection, or calibration is permitted.

This is a retrospective walk-forward structural diagnostic, not a claim that 2019–2022 are pristine model-selection holdouts. The hyperparameters were selected during the broader <=2024 development process. Its purpose is to determine whether a multi-year directional pattern exists strongly enough to make an explicit prospective hypothesis for sealed 2025.

### 2020

The shortened 2020 season remains in the **primary** six-season sequence. Because its schedule was structurally unusual, the same trend statistics will also be reported excluding 2020 as a predeclared sensitivity. We will not decide whether to include/exclude 2020 after seeing which version is prettier.

## Obvious-power comparator

Use the same frozen long-horizon batter-only proxy already used in edge localization. No pitcher, park, recent-form, or pitch-matchup context is allowed.

For target year Y, any missing-value medians used by the proxy must be calculated from seasons <= Y-1 only. Proxy scores remain within-slate composites of the same frozen long-horizon batter features.

## Selector

Primary selector for both models:

- rank independently within `game_date`;
- full73 ranking score = raw XGBoost probability;
- obvious ranking score = frozen obvious-power composite;
- select `ceil(0.05 * slate_size)` with minimum one batter;
- use deterministic outcome-independent tie keys (`game_pk`, `batter_id`).

## Primary migration statistics

Define `full73_only` as daily full73 top-5% selections that are **not** in the obvious-power daily top 5%.

For every full73-only batter-game, define continuous obvious depth:

`obvious_rank_percentile = (obvious_rank - 1) / (slate_size - 1)`

where 0 means the very top of the obvious-power board and larger values mean a deeper / less-obvious candidate.

Report by year:

1. `n_full73_top5`
2. `n_full73_only`
3. top-5 overlap fraction = shared / full73 top-5
4. mean obvious-rank percentile of full73-only picks
5. median obvious-rank percentile of full73-only picks
6. 75th percentile obvious-rank percentile of full73-only picks
7. mean and median absolute obvious rank of full73-only picks (descriptive only; slate size varies)

### Direction-of-trend summaries

Primary continuous trend summaries:

- linear slope of yearly **mean obvious-rank percentile** versus calendar year;
- linear slope of yearly **top-5 overlap fraction** versus calendar year;
- Spearman correlation between year and yearly mean obvious-rank percentile;
- Spearman correlation between year and yearly overlap fraction;
- number/sign of the five adjacent year-to-year changes.

Uncertainty for the linear slopes will use a paired **slate-date cluster bootstrap**, resampling dates within each year and recomputing the yearly statistics and six-year slope. Report observed slope, bootstrap median, 95% percentile CI, and `P(slope > 0)`.

Interpretation is directional rather than threshold-driven:

- positive depth slope = contextual selections are migrating farther away from obvious-power;
- negative depth slope = contextual selections are migrating toward obvious-power;
- overlap slope gives the complementary convergence/divergence view.

No rank cutoff or required magnitude is chosen in advance.

## Outcome-weighted diagnostics — secondary only

To distinguish 'the model is merely reaching deeper' from 'its successful contextual selections are also migrating deeper', report by year among `full73_only`:

- HR rate;
- mean obvious-rank percentile among HR successes;
- median obvious-rank percentile among HR successes.

These are secondary descriptive diagnostics. They do **not** create a new hidden-pick rule or bucket.

## Mechanism diagnostics

For each full73-only selection, compute the same within-date contextual composite percentiles already defined in `v12_edge_localization.py`:

- park;
- recent batter form;
- pitcher vulnerability;
- pitch matchup.

For each year report the mean of each context-group score among full73-only selections, plus its linear year slope.

Also report the within-selection association between each context-group score and obvious-rank percentile (overall and by year where sample size permits).

These mechanism tables explain **how the frozen model is promoting deeper hitters**. They are not independent confirmation of predictive edge.

## Things this run must NOT do

- no 2025 reads or scores;
- no calendar-month outcome mining;
- no 5–8 / 9–16 / 17+ threshold search;
- no new feature selection;
- no Optuna / hyperparameter changes;
- no outcome-conditioned choice of year range;
- no deleting 2020 after inspecting results;
- no market-odds/ownership claims;
- no automatic production policy.

## After this run

This is intended to be the **last structural development diagnostic** before a reset/review of the entire v1.2 evidence trail.

After the result:

1. record the migration result in-repo;
2. summarize what survived and what failed across feature ablations, tail bootstrap, disagreement tests, seasonality, and migration;
3. decide whether architecture/evaluation policy are frozen enough for the authorized one-shot 2025 holdout;
4. only then, if authorized, open 2025.
