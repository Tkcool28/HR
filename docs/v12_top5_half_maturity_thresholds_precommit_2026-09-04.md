# v1.2 top-5 half-season feature-maturity thresholds — 2026-09-04

This addendum freezes the low-history diagnostics **before any FIRST_HALF / SECOND_HALF outcome is inspected**.

It supplements `docs/v12_dynamic_bucket_regime_precommit_2026-09-04.md` and does not change the primary top-5 seasonality test.

## Diagnostic thresholds

For active rolling-history support/count features:

- PA-based support: report both **zero support** and **<20 prior PA** as low-history diagnostics.
- Pitch-count support (`top_pitch_total_pitches_30d` or an equivalent total-pitch support variable): report both **zero support** and **<100 prior pitches**.

These thresholds are descriptive diagnostics only. They do not define model pass/fail and may not be changed after the half-season result is observed to make a mechanism story fit.

For every active feature containing `14d` or `30d`, independently report pre-imputation NaN rate by half for:

1. the full slate population; and
2. the full73 daily top-5%-selected population.

For support/count variables, also report median, p10, p25, zero-support fraction, and predeclared low-history fraction.

Overall in-game HR base rate is reported by half as context.

## Interpretation rule

The primary inference remains the paired full73-vs-obvious-power top-5% lift and the predeclared SECOND_HALF-minus-FIRST_HALF interaction.

Feature maturity is a confound/measurement diagnostic. It cannot by itself establish or rescue a seasonality effect.

**2025 remains sealed.**
