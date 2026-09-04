# v1.2 adversarial audit — 2026-09-03

Scope: audit the delivered v1.2 correction without modifying model/data methodology. 2025 remains out of scope and was not materialized for the clean rebuild.

## Reproduction result

- Exact correction package reconstruction: PASS.
- All 17 reconstructed files matched the correction manifest SHA-256 values.
- Delivered temporal-integrity suite: 35 PASS / 0 FAIL.
- Independent structural checks: PASS.
- Clean rebuild from 2015-2024 historical inputs: PASS.
- Rebuilt outputs matched headline row-count invariants:
  - `pa_v12.parquet`: 1,756,558 rows
  - `game_starters_v12.parquet`: 23,191 rows
  - `pitch_level_v12.parquet`: 6,787,051 rows
  - `features/v1.2/game_features.parquet`: 198,037 rows
  - active features: 54

## Confirmed repairs that appear sound

- True PA grain is one row per `(game_pk, at_bat_number)`.
- Terminal-pitch HR attribution is used.
- Pitcher pitch-usage is rebuilt from individual pitch rows with separate per-pitch-type rolling histories.
- Same-day and future pitches are excluded from the 30-day usage windows.
- Dynamic pitcher top pitch is no longer hard-coded to FF; delivered FF top-pitch share is about 55.5%.
- Prior-only park factors exclude current-year outcomes.
- Starting pitcher identity matches the opposing starter table.
- No 2025 rows are present in v1.2 features.

## Critical defect: batting-universe construction

`build_target_rows()` defines a historical starting-lineup proxy as batters whose first PA occurred in inning 1. This is not a valid reconstruction of the starting nine. Lower lineup slots are excluded whenever the first inning ends before they bat, so inclusion depends on same-game offensive events.

Observed proxy hitters per team-game:
- median: 4
- 25th percentile: 3
- 75th percentile: 5
- team-games with <=4 proxy hitters: 29,957 / 46,382
- team-games with exactly 9 proxy hitters: 662 / 46,382
- 3 team-games even contain 10 proxy hitters, proving first-inning substitutions can enter the proxy.

Using the first nine distinct batters to appear for each side as an audit benchmark, proxy retention by appearance slot is:

| Slot | Retained |
|---:|---:|
| 1 | 100.0% |
| 2 | 100.0% |
| 3 | 100.0% |
| 4 | 64.0% |
| 5 | 35.4% |
| 6 | 17.6% |
| 7 | 7.8% |
| 8 | 3.3% |
| 9 | 1.4% |

Overall first-nine retention is 47.7%.

Historical game-HR rate differs materially between the selected and excluded first-nine populations:
- all first-nine benchmark rows: 11.85%
- proxy-included: 13.98%
- proxy-excluded: 9.90%

This is a major target-sampling/selection-bias defect. It is not caught by the delivered 35-check temporal suite.

## Split assignment

Source robustness issue: the builder initializes every target row as `train` and only overwrites known validation IDs. Unknown game IDs would therefore silently become training rows rather than failing closed.

However, this did **not** contaminate the delivered v1.2 matrix:
- feature games outside train+val IDs: 0
- feature rows outside train+val IDs: 0
- train rows dated 2023-2024: 0
- validation rows dated <=2022: 0
- train/validation game-ID overlap: 0

Treat this as a latent robustness defect, not a current statistical contamination finding.

## Game coverage gap

- starter/split games: 23,191
- game metadata rows: 23,054
- final feature games: 23,054
- 137 starter/split games are absent from `game.parquet` and therefore absent from the final feature universe.

This is inherited from the original package rather than introduced by the v1.2 correction. The original split-integrity failure was larger because it also included the sealed 2025 split; after restricting to 2015-2024, 137 historical games remain outside `game.parquet`.

## Historical park-identity defect

The leakage-safe v1.2 park-factor averaging still depends on the original pipeline's `park_id`. That `park_id` is not an actual venue identifier from each game: it is assigned from a static `home_team -> park_id` table.

The source comment claims a `(home_team, game_year) -> park_id` lookup, but the implementation does not use year. As a result, historical stadium changes and special-site games are misidentified.

Direct audit of the delivered BIP data shows:

- ATL: every home BIP from 2015-2024 is assigned `park_id=41` (Truist Park), including 4,178 BIPs in 2015 and 4,228 in 2016. Atlanta did not move from Turner Field to its new park until 2017.
- TEX: every home BIP from 2015-2024 is assigned `park_id=27` (Globe Life Field), including 21,876 BIPs from 2015-2019. The Rangers' 2019 season was the final season at Globe Life Park and they moved to Globe Life Field for 2020.
- TOR: every home BIP in 2020 and 2021 is assigned `park_id=28` (Rogers Centre), despite Toronto playing the majority of 2020 home games at Sahlen Field in Buffalo and opening 2021 home games at TD Ballpark in Dunedin before later using Buffalo and returning to Toronto.

This affects both direct game park features and the historical source observations used to compute future prior-3-year park factors. For example, Texas' 2020-2022 Globe Life Field prior factors can inherit observations from the old Globe Life Park because 2017-2019 were mislabeled as the new stadium.

The park table also lists special/neutral venues (London, Field of Dreams, Williamsport, etc.) but the ordinary mapping is based on MLB `home_team`, so those games cannot be reliably distinguished from the club's normal home stadium using this implementation.

This is a genuine model-data defect, separate from the same-season leakage that v1.2 correctly repaired.

## Feature-health diagnostics

For the delivered 54-feature matrix:
- duplicate `(batter_id, game_pk)` rows: 0
- constant numeric active features: 0
- numeric active features with >=10% missing values: 0
- active numeric features containing infinity: 0
- exact duplicate numeric active feature pairs: 0
- park factors equal neutral 100.0 for all 2015 rows and essentially none from 2016 onward, consistent with no prior seasons being available for 2015.

### Feature-surface reduction / methodology drift

v1.1 had 113 active features; the corrected v1.2 package has 54. A direct list diff shows:

- 31 feature names are retained in both versions.
- 82 v1.1 features are absent from v1.2.
- 23 feature names are new in v1.2.

The v1.2 repair report explicitly documents this as intentional rather than accidental. It dropped the batted-ball / QoC block (barrel rate, xwOBA, EV, hard-hit, launch-angle, ISO), BVP features, and other v1.1 features to reduce repair complexity, while adding corrected pitcher-usage/top-pitch features.

This transparency is positive, but the consequence is important: v1.2 is not a controlled "v1.1 with leakage fixed" experiment. It is a materially different feature model. Any later performance difference between v1.1 and v1.2 will confound leakage repair with feature removal/addition unless the clean unaffected v1.1 features are deliberately restored or an ablation design is used.

### Pitch-type semantic mismatch

`pitcher_top_pitch` is determined from all individual pitches thrown in the prior 30-day window, which is appropriate for an arsenal-usage feature. However, the batter/pitcher `*_hr_per_pa_vs_<PT>_30d` features use `terminal_pitch_type` from each PA. They therefore estimate HR rate among PAs that **ended** on pitch type PT, not performance across all exposures to PT.

`batter_strength_on_pitcher_top_pitch` consequently means roughly "batter HR rate in prior PAs ending on the pitcher's most-used pitch type," not a general measure of batter performance against every pitch of that type. This is not temporal leakage, but the naming/interpretation overstates the feature semantics.

## Trainer-interface defect

`pitcher_top_pitch` is included in `features/v1.2/feature_list.json` as an active string feature. The existing v1/v1.1 trainer contract converts the entire active feature matrix with `.astype("float32")`. A direct compatibility test fails with:

`ValueError: could not convert string to float: ''`

Thus the corrected v1.2 feature set is not currently consumable by the existing numeric training interface without an encoding/removal/interface change. This is operational/model-interface debt, not evidence of temporal leakage.

## Training / validation architecture findings

The v1.2 correction package does not provide a replacement training/evaluation architecture; the existing v1/v1.1 trainer remains the relevant implementation. That trainer has a material validation-reuse problem.

The same 2023-2024 validation block is used repeatedly for:

1. Optuna hyperparameter selection (25 trials in v1.1), optimizing validation Brier.
2. Early stopping inside each trial on the same validation block.
3. Selecting the final best boosting round via early stopping on that same validation block.
4. Reporting raw XGBoost validation Brier/AUC/AP/log-loss/top-5%.
5. Fitting `IsotonicRegression` on those same validation predictions and labels.
6. Reporting calibrated Brier/AUC/AP/log-loss on those same calibration-fit rows.
7. Computing `reliability_val.csv` on those same calibration-fit rows.
8. Running `tests/test_calibration.py`, including ECE and per-bin reliability, on those same calibration-fit rows.

Therefore the published calibrated validation performance is not an independent generalization estimate. In particular, the reported validation ECE of 0.0000 is not meaningful evidence of out-of-sample calibration because isotonic calibration was fit and then evaluated on the same predictions/labels.

Main-path preprocessing is cleaner: NaN imputation means and LR scaling parameters are computed from the training split only and then applied to validation/holdout.

### Walk-forward future-distribution leakage

The v1.1 `walkforward_backtest()` computes a single `col_means` vector from the complete train+validation union before iterating through seasons 2019-2024. Each historical fold therefore uses imputation statistics containing feature-distribution information from its test season and future seasons. Example: the 2019 fold's imputation vector can reflect 2020-2024 observations.

This is not target-label leakage, but it is future-distribution leakage and means the walk-forward is not a clean fold-local simulation. Preprocessing should be fitted independently from the prior-year training slice inside each fold.

The walk-forward also uses fixed XGBoost parameters rather than reproducing a nested tuning/calibration procedure, so it should be treated as a secondary stability diagnostic, not as an independent validation of the tuned/calibrated production pipeline.

## Acquisition / reproducibility findings

The original acquisition code contains two operational defects:

- `acquire.py` uses `glob.glob(...)` in `main()` without importing `glob`, so the combined-parquet stage is not reproducible as written.
- `re_chunk_if_capped()` immediately returns cached chunks without rechecking whether they are at the Savant cap. A previously cached capped/truncated response would therefore be preserved on a rerun instead of recursively split. `acquire_season()` only logs the number still capped rather than failing closed.

However, the delivered acquisition logs report `0 still capped after re-chunking` for every 2015-2025 season. Therefore there is no evidence from the delivered logs that the historical data used here was actually truncated by this bug. Treat it as latent acquisition fragility, not a confirmed current-data loss finding.

## Minor implementation debt observed

- `catcher_interference` is included in the generic `out` branch of PA outcome classification, making the later intended `walk` branch unreachable. This does not change HR labels but is a correctness/code-quality defect.
- Several NumPy/Pandas deprecation warnings and DataFrame fragmentation warnings occur during the build. These are maintainability/performance issues, not current statistical defects.

## Current audit verdict

The v1.2 correction substantially improves the temporal feature engine and correctly repairs the pitch-usage/top-pitch bug. The code is reproducible from historical inputs and the rebuilt feature matrix is structurally healthy. However, the batting-universe proxy is a critical conceptual defect that materially changes the sampled population; historical venue identity is also wrong for known stadium changes and special-site games; the active feature list is not trainer-compatible; the inherited training/evaluation architecture reuses the 2023-2024 validation block too aggressively to support clean calibrated-performance claims; and the v1.2 repair materially changes the feature surface, so it is not a controlled apples-to-apples repair of v1.1. No model performance claim should be accepted from this v1.2 matrix until the target-universe and park-identity issues are repaired and the evaluation design separates tuning, calibration fitting, and final assessment.