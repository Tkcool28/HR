# HR MODEL v1.1 — CRITICAL METHODOLOGY REPAIR HANDOFF

## Phase 1: Eliminate Temporal Leakage and Repair Historical Feature Construction

### Purpose

Audit and repair the existing HR model's **critical training/validation methodology defects** before doing any further model evaluation or development.

Focus on the historical modeling years already used for development:

- 2015–2022 training
- 2023–2024 validation/evaluation

**Do not score, inspect, evaluate, tune against, or otherwise use the 2025 holdout during this task.**

The 2025 data should remain sealed for later genuinely out-of-sample evaluation after the historical pipeline is repaired and frozen.

Do not redesign the entire model. Do not add sportsbook odds, live lineup acquisition, weather, frontend work, or unrelated features yet.

The immediate objective is to ensure that every historical feature available to a prediction for game date **D** uses only information that could have been known before that game.

## 1. Full-season pitcher pitch-usage leakage

The current pitch-type pitcher feature construction appears to calculate pitcher pitch usage using all pitches from the pitcher's entire season and then merge those values onto individual games by pitcher/year.

For a game on date D:

- include only pitches from games strictly before D;
- never include pitches from D itself;
- never include future games;
- preserve appropriate rolling/minimum-sample/prior treatment unless correctness requires change.

Audit all pitcher pitch-type / arsenal / usage / outcome aggregates, especially any values currently keyed only by pitcher + season.

Acceptance invariant for sampled rows:

`max(source_event_date_used_for_feature) < prediction_game_date`

Do not rely only on `_as_of` column names; enforce the temporal boundary in construction.

## 2. Same-season park-factor leakage

Current park-factor construction appears to allow complete current-season results to contribute to earlier games in that season.

Preferred simple repair: for season Y use only completed prior seasons such as Y-1, Y-2, Y-3. Do not include Y itself. Apply the same rule to handedness-specific factors.

Document exactly which seasons contribute to each game-year park factor.

## 3. Anti-leak tests are too narrow

Expand temporal-integrity testing beyond columns with explicit `_as_of` timestamps.

Cover at minimum:

- batter rolling statistics;
- pitcher rolling statistics;
- batter pitch-type statistics;
- pitcher pitch-type/arsenal statistics;
- park factors;
- quality-of-contact aggregates;
- handedness splits;
- season-level aggregates merged onto game rows.

Model-wide invariant:

**No statistic derived from game results may contain information from the prediction date or any later date.**

## 4. Historical matchup construction: use pregame starter semantics

Knowing the day's starting pitcher and batting lineup is **not** look-ahead information. Announced/probable starters and lineups are routinely known pregame.

However, historical reconstruction must represent the **pregame-known matchup**, not infer the prediction pitcher from how the completed game unfolded.

Investigate whether the current builder selects the pitcher who ultimately threw the most pitches to a batter. If so, replace that with the game's actual starting pitcher from a defensible historical field/source.

Do not remove pitcher or lineup identity merely because they correspond to the same day's game.

If exact historical announced-lineup reconstruction is unavailable, document the limitation rather than silently substituting a postgame-derived concept.

## 5. Plate-appearance construction

Inspect whether feature code collapses pitch-level data by something like `game_pk + batter_id + pitcher_id` and then sets `is_pa = 1`.

If confirmed, repair it. A batter can have multiple PAs against the same pitcher in one game.

Construct genuine PAs using the proper PA identifier, preferably `game_pk + at_bat_number`, retaining batter/pitcher identity as appropriate.

Verify:

- one row per completed PA;
- HR outcome belongs only to the PA in which it occurred;
- PA counts are genuine PA counts;
- HR/PA denominators and Bayesian PA priors use actual PAs.

## 6. Pitch-type HR attribution

Audit batter/pitcher HR statistics by pitch type.

If a PA contains FF, FF, SL, SL → HR, the HR belongs to the **terminal SL pitch**. It must not be credited as an HR against every pitch type seen during the PA.

Pitch usage/exposure may count every pitch. PA-ending outcomes must be attributed to the terminal result pitch.

## 7. `batter_strength_on_pitcher_top_pitch`

Inspect whether this feature is simply assigned from a fastball-specific batter feature rather than dynamically selecting performance against the pitcher's actual most-used pitch.

If confirmed:

1. determine the pitcher's top pitch using only pregame historical information;
2. identify that pitch type for the prediction row;
3. select the batter's historical performance against that pitch type.

If reliable implementation is not possible, remove the misleading feature rather than retain a mislabeled fastball proxy.

Add a deterministic unit test with pitchers whose top pitches differ.

## 8. Rebuild scope

After repair, regenerate **2015–2024 only**:

- curated/base tables as necessary;
- PA tables;
- historical features;
- training matrix;
- 2023–2024 validation matrix;
- relevant feature metadata.

Do **not** regenerate, inspect, score, or evaluate 2025 predictions during this task.

## 9. Required evidence

Return a repair report with:

- exact files changed;
- each finding classified CONFIRMED / NOT CONFIRMED / PARTIALLY CONFIRMED with source evidence;
- temporal-integrity samples across multiple seasons and feature families;
- PA reconciliation samples against source data;
- terminal-pitch attribution proof;
- actual-starting-pitcher reconstruction proof;
- full test results and new regression tests.

## 10. Stop condition

Once the 2015–2024 historical data/features are rebuilt and these integrity defects are resolved, **STOP**.

Do not yet:

- retune XGBoost;
- recalibrate probabilities;
- evaluate headline model performance;
- open the 2025 holdout;
- score 2025;
- add sportsbook odds;
- add weather;
- build live inference;
- optimize betting strategy.

The next phase will separately address tuning/calibration/evaluation design and only later the untouched 2025 holdout.
