# v1.2 audit severity ledger — 2026-09-03

This is a compact severity ledger for the adversarial audit of the delivered v1.2 correction. It intentionally records findings without implementing repairs. The 2025 holdout remains sealed and was not materialized or evaluated.

## Blockers before any trustworthy retrain/performance claim

| Finding | Severity | Status / evidence |
|---|---|---|
| Historical batting target universe uses first-PA-in-inning-1 as a starting-lineup proxy | CRITICAL | Confirmed. Only 47.7% of first-nine benchmark hitters retained; slots 8/9 retained 3.3%/1.4%; retained hitters have 13.98% game-HR rate vs 9.90% excluded. |
| Postseason games are mixed into nominal regular-season train/validation targets | MAJOR | Confirmed. Conservative Oct-08+ check finds 275 playoff games and 2,367 feature targets; actual count is higher. Acquisition URL uses `game_type=R`, while Savant's season-type filter is `hfGT`, explaining the silent scope failure. |
| Historical park identity uses static home-team mapping instead of actual venue | MAJOR | Confirmed. ATL 2015-16 mapped to Truist; TEX 2015-19 mapped to Globe Life Field; TOR 2020-21 mapped to Rogers Centre. Mislabels also contaminate later prior-3-year factors. |
| 2023-24 validation block reused for tuning, early stopping, calibration fit, and calibrated reporting | MAJOR | Confirmed in inherited v1/v1.1 trainer. Calibrated validation metrics/ECE are not independent generalization estimates. |
| v1.2 active feature list includes string `pitcher_top_pitch` under numeric trainer contract | MAJOR operational | Confirmed. Existing `.astype("float32")` path fails. |

## Important but secondary findings

| Finding | Severity | Status / evidence |
|---|---|---|
| Walk-forward imputation means computed from full 2015-24 union before folds | MODERATE | Future-distribution leakage in historical folds; no target-label leakage. |
| v1.2 feature surface reduced from 113 to 54 | MODERATE methodology drift | Intentional/documented: 31 retained, 82 v1.1 omitted, 23 new. Prevents apples-to-apples attribution of performance changes solely to bug repairs. |
| `batter_strength_on_pitcher_top_pitch` combines all-pitch arsenal usage with terminal-pitch-type PA HR rates | MODERATE semantics | Not temporal leakage, but feature name/interpretation overstates what is measured. |
| 137 historical split/starter games absent from `game.parquet` | MODERATE coverage | Inherited from original package, not introduced by v1.2. |
| Unknown split IDs default to train | MODERATE latent | Delivered matrix happens to have zero unknown IDs, so no current contamination. |

## Minor / operational debt

| Finding | Severity | Status / evidence |
|---|---|---|
| Cached capped Savant chunks are trusted without re-checking cap | MINOR-to-MODERATE latent | Delivered logs report zero still-capped chunks, so no evidence current historical pull was truncated. |
| `glob` used without import in original acquisition script | MINOR operational | Reproducibility defect. |
| `catcher_interference` is captured by `out` branch before intended later `walk` branch | MINOR | Does not alter HR label. |
| NumPy/Pandas deprecations and fragmented-DataFrame warnings | MINOR | Maintainability/performance debt. |

## What the fast LLM got right

- The v1.2 correction reconstructs reproducibly from 2015-2024 historical inputs.
- True PA grain is repaired.
- Terminal-pitch HR attribution is repaired.
- Pitcher usage is rebuilt from actual pitch-level rows rather than PA-terminal proxies.
- Same-day and future pitches are excluded from rolling usage.
- Dynamic top pitch is genuinely fixed and negative controls reproduce the old bug.
- Prior-only park-factor averaging fixes the original same-season temporal leakage (although venue identity remains wrong).
- Starting-pitcher identity matches opposing starters.
- Numeric feature matrix health is strong: no constant numeric active features, no infinities, no exact duplicate numeric features, no numeric active feature with >=10% missingness.
- Delivered 35-check temporal suite passes and a clean rebuild reproduces the headline row-count invariants.

## Audit conclusion

The delivered work is not an unusable code dump. Its strongest data-engineering repairs are real, fast, and reproducible. However, its self-tests are insufficient protection against domain and data-contract mistakes: the lineup proxy, ineffective regular-season query filter, and static park mapping all produce plausible-looking outputs while materially changing the modeled population.

Current workflow verdict: promising as a high-velocity implementation accelerator **only with independent adversarial review**. Do not accept model-performance claims or open the 2025 holdout until the blocker class above is repaired and 2015-2024 is rebuilt/revalidated under a clean evaluation design.
