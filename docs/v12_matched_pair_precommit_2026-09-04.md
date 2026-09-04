# v1.2 nearest-rank matched-pair precommit — 2026-09-04

## Purpose

Precommit the decision rule and inference design **before** observing the nearest-rank matched-pair results for the frozen full73 champion.

**2025 remains sealed and must not be read or evaluated.**

## Frozen target

- Champion: 73-feature aggressive XGBoost model.
- Ranking score: raw XGBoost probability.
- Operational selector: daily top four.
- Obvious-power proxy: previously frozen long-horizon batter-only proxy.
- Previously declared disagreement bands: obvious-power ranks 5–8, 9–16, and 17+.
- Primary lead of interest: full73 top-four promotions from obvious-power rank 17+.

The three bands were defined before their HR outcomes were inspected in the first edge-localization result. The matched test will evaluate all three using the same fixed matching rule; no cutoff may be changed after results are observed.

## Same-slate nearest-rank match

For every full73 top-four hitter in a disagreement band, choose one hitter from the **same slate date and same obvious-power rank band** who is not in the full73 top four.

Controls are matched without replacement. The match minimizes absolute obvious-power rank distance with deterministic tie-breaking. This is intended to remove the residual confound whereby a selected rank-18 hitter could otherwise be compared with controls ranked far deeper in the 17+ tail.

## Inference

For each band:

1. Compute the paired observed lift: selected HR rate minus matched-control HR rate.
2. Resample complete slate dates with replacement for 10,000 paired bootstrap replicates; all matched pairs on a sampled date move together.
3. Report the 95% percentile CI for the paired lift.
4. Run an exact one-sided matched-binary McNemar/binomial test on discordant pairs.
5. Apply Holm correction across the three predeclared bands.
6. Report 2023 and 2024 separately as stability diagnostics, without moving the primary combined decision rule.

## Precommitted magnitude rule

A disagreement band **survives for operational consideration** only if both are true:

- the combined 2023–24 95% paired bootstrap CI for lift is entirely above 0; and
- the observed matched lift is at least **+5.0 absolute percentage points**.

A band receives a **strong-signal grade** only if, in addition:

- the lower endpoint of the combined 95% paired bootstrap CI is at least **+2.0 percentage points**.

Statistical significance without the +5 pp observed magnitude floor is not enough for an operational hidden-pick rule because HR props are high variance and the purpose is to find materially differentiated decisions, not microscopic improvements.

Holm-adjusted p-values are supporting inference; they do not override the magnitude rule.

## Freshness check if the matched signal survives

After the 2023–24 matched test is frozen, replay the **exact same band definition, matching rule, and decision thresholds** on 2022.

For 2022:

- score with the frozen full73 XGBoost architecture trained only on 2015–2021;
- use **raw** XGBoost ranking only;
- do not evaluate isotonic probabilities fit on 2022;
- construct the same obvious-power proxy using only information available before each 2022 game, with preprocessing based on pre-2022 training data;
- treat 2022 as a freshness/replication check, not a pristine final holdout.

No 2022 result may be used to change the 5–8 / 9–16 / 17+ cutoffs or the +5 pp / CI decision thresholds.

## Final holdout

2025 is the sealed one-shot final holdout. None of the work described here authorizes opening it.
