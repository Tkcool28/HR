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

This is a coverage issue requiring source-level explanation before final freeze, but it is small relative to the lineup-universe defect.

## Feature-health diagnostics

For the delivered 54-feature matrix:
- duplicate `(batter_id, game_pk)` rows: 0
- constant numeric active features: 0
- numeric active features with >=10% missing values: 0
- active numeric features containing infinity: 0
- exact duplicate numeric active feature pairs: 0
- park factors equal neutral 100.0 for all 2015 rows and essentially none from 2016 onward, consistent with no prior seasons being available for 2015.

## Trainer-interface defect

`pitcher_top_pitch` is included in `features/v1.2/feature_list.json` as an active string feature. The existing v1/v1.1 trainer contract converts the entire active feature matrix with `.astype("float32")`. A direct compatibility test fails with:

`ValueError: could not convert string to float: ''`

Thus the corrected v1.2 feature set is not currently consumable by the existing numeric training interface without an encoding/removal/interface change. This is operational/model-interface debt, not evidence of temporal leakage.

## Minor implementation debt observed

- `catcher_interference` is included in the generic `out` branch of PA outcome classification, making the later intended `walk` branch unreachable. This does not change HR labels but is a correctness/code-quality defect.
- Several NumPy/Pandas deprecation warnings and DataFrame fragmentation warnings occur during the build. These are maintainability/performance issues, not current statistical defects.

## Current audit verdict

The v1.2 correction substantially improves the temporal feature engine and correctly repairs the pitch-usage/top-pitch bug. The code is reproducible from historical inputs and the rebuilt feature matrix is structurally healthy. However, the batting-universe proxy is a critical conceptual defect that materially changes the sampled population, and the active feature list is not yet trainer-compatible. No model performance claim should be accepted from this v1.2 matrix until the target-universe issue is repaired and the resulting 2015-2024 feature matrix is rebuilt/revalidated.
