# v1.2 trusted feature ablation and pruning ledger — 2026-09-04

## Scope

This ledger records the feature-family destruction tests, daily-slate ranking diagnostics, fine-grained subfamily ablations, and targeted pruning candidates performed on the trusted 2015-2024 v1.2 rebuild.

**2025 was not read or evaluated and remains sealed.**

All comparisons below use the same repaired target universe and leakage-safe feature construction. Unless explicitly stated otherwise, ablation/pruning comparisons use the frozen winning aggressive XGBoost hyperparameters and 194 boosting rounds from the 73-feature champion. This deliberately prevents a removed feature family from being rescued by retuning.

## Frozen 73-feature reference

2023-2024 development assessment:

- calibrated Brier: 0.102500
- calibrated AUC: 0.6175
- ECE: 0.00626
- raw daily top-10% HR rate: 20.00%
- raw daily top-5% HR rate: 21.69%
- raw daily top-2% HR rate: 23.60%
- raw daily top-1% HR rate: 25.71%
- #1 hitter/day HR rate: 26.46%
- top-2/day HR rate: 25.77%
- top-4/day HR rate: 24.23%
- top-8/day HR rate: 22.67%
- top-4/day 2023: 24.58%
- top-4/day 2024: 23.88%

Daily selectors use raw XGBoost score within each game date. Calibration is evaluated separately so isotonic ties cannot alter the ranking bucket.

## Broad family ablations

| Removal | Features left | Cal Brier | AUC | ECE | Daily top-5% | Top-4/day | Top-4 retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| none / full_73 | 73 | 0.102500 | 0.6175 | 0.00626 | 21.69% | 24.23% | 100% |
| park | 71 | 0.102553 | 0.6170 | 0.00645 | 20.97% | 22.35% | 70.5% |
| all QoC | 53 | 0.102719 | 0.6114 | 0.00802 | 21.74% | 23.96% | 59.4% |
| batter QoC | 59 | 0.102739 | 0.6113 | 0.00805 | 21.71% | 23.75% | 60.4% |
| pitcher QoC | 67 | 0.102507 | 0.6167 | 0.00646 | 22.00% | 23.47% | 82.8% |
| long-horizon QoC | 67 | 0.102617 | 0.6151 | 0.00723 | 21.67% | 23.47% | 71.9% |
| recent 30d QoC | 59 | 0.102580 | 0.6163 | 0.00696 | 22.03% | 22.49% | 77.0% |
| pitch-type matchup block | 44 | 0.102509 | 0.6167 | 0.00606 | 21.49% | 23.33% | 80.9% |
| barrel family | 68 | 0.102547 | 0.6166 | 0.00639 | 21.56% | 24.03% | 79.5% |
| xwOBA family | 69 | 0.102478 | 0.6176 | 0.00635 | 21.91% | 23.89% | 82.6% |

### Broad-family decisions

- **Park stays.** Two park features are highly influential in the actionable shortlist; removing them drops top-4/day by 1.88 percentage points.
- **QoC stays as a family.** Removing the 20 QoC features materially worsens AUC/ECE and changes about 40% of top-four selections.
- **Pitch matchup requires decomposition, not wholesale deletion.** The broad block is mixed globally but changes the practical shortlist.
- No broad family was deleted directly from this pass.

## Fine-grained subfamily ablations

Key daily-slate results:

| Removal | Daily top-5% | Top-2% | #1/day | Top-2/day | Top-4/day | Top-8/day |
|---|---:|---:|---:|---:|---:|---:|
| full_73 | 21.69% | 23.60% | 26.46% | 25.77% | 24.23% | 22.67% |
| 9 pitcher usage proportions | 21.67% | 23.49% | 26.46% | 24.93% | **24.65%** | **23.08%** |
| batter long-horizon xwOBA | 21.78% | 23.54% | **27.30%** | 25.63% | **24.44%** | **23.22%** |
| pitcher-vs-pitch HR rates | 21.47% | 23.60% | 25.63% | 25.35% | 24.03% | 22.98% |
| batter-vs-pitch HR rates | 21.78% | 23.70% | 25.63% | 24.93% | **23.54%** | 22.95% |
| dynamic top-pitch pair | 21.74% | 23.13% | 26.18% | 25.07% | **23.68%** | 22.32% |
| batter barrel 30d | 21.91% | 23.18% | 24.51% | 24.65% | **23.82%** | 22.56% |
| pitcher barrel season | 21.85% | 23.60% | 26.46% | **23.96%** | **23.33%** | 22.53% |
| batter avg EV season | 21.98% | 23.23% | 24.51% | 25.63% | **23.96%** | 22.77% |
| batter contact-shape 30d | 22.05% | 24.27% | 23.96% | 25.07% | **23.89%** | 23.29% |
| pitcher contact-shape 30d | 21.65% | 23.60% | 24.23% | 24.37% | 24.03% | 22.77% |
| batter xwOBA 30d | 21.87% | 23.39% | 24.79% | 24.09% | **23.82%** | 23.02% |

### Fine-grained interpretation

Strong/provisional keep evidence:

- pitcher season barrel rate
- batter 30-day barrel rate
- batter season average EV
- batter-vs-pitch-type HR rates
- dynamic top-pitch interaction pair
- recent batter/pitcher contact-shape features
- recent xwOBA features, despite mixed broader-percentile behavior

Hard-challenge candidates:

- the 9 raw pitcher pitch-usage proportions
- batter season/career xwOBA-on-contact features

The intent is not to select a feature because a single noisy bucket improves. Promotion/pruning requires a coherent pattern across shortlist metrics plus no material global calibration/discrimination damage.

## Targeted joint pruning

The two strongest passenger candidates were then removed alone and in combinations, still under the frozen 73-feature champion hyperparameters/rounds.

| Candidate | Features | Cal Brier | AUC | ECE | #1/day | Top-2/day | Top-4/day | Top-8/day | Daily top-5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_73 | 73 | 0.102500 | 0.6175 | 0.00626 | 26.46% | 25.77% | 24.23% | 22.67% | 21.69% |
| prune usage | 64 | 0.102502 | 0.6169 | 0.00633 | 26.46% | 24.93% | 24.65% | 23.08% | 21.67% |
| prune batter long xwOBA | 71 | 0.102512 | 0.6171 | 0.00628 | 27.30% | 25.63% | 24.44% | 23.22% | 21.78% |
| **prune usage + batter long xwOBA** | **62** | **0.102499** | **0.6175** | **0.00620** | **27.02%** | 24.65% | **25.07%** | **22.98%** | **21.89%** |
| prune usage + pitcher-vs-pitch HR | 55 | 0.102472 | 0.6180 | 0.00608 | 28.69% | 24.65% | 23.82% | 23.19% | 21.69% |
| prune pitcher-vs-pitch HR + batter long xwOBA | 62 | 0.102509 | 0.6173 | 0.00637 | 25.07% | 26.74% | 24.23% | 22.39% | 21.76% |
| prune usage + pitcher-vs-pitch HR + batter long xwOBA | 53 | 0.102474 | 0.6175 | 0.00634 | 27.58% | 26.04% | 24.23% | 22.98% | 22.29% |

### Challenger decision

The **62-feature `prune_usage_plus_batter_xwoba_long` candidate** is promoted to a *challenger only* because, without any retuning, it:

- improves top-4/day from 24.23% to 25.07%
- improves daily top-5% from 21.69% to 21.89%
- improves #1/day from 26.46% to 27.02%
- leaves AUC effectively unchanged
- leaves Brier effectively unchanged/slightly better
- improves ECE from 0.00626 to 0.00620

It is **not yet the champion**. It must receive the same 50-trial, three-fold chronological Optuna treatment as the full model before any architecture freeze.

The 55-feature candidate with usage + pitcher-vs-pitch HR removed has attractive global metrics and #1/day, but materially hurts top-4/day; it is not the preferred shortlist challenger.

## Bootstrap gate

The paired bootstrap proposed for the actionable tail is intentionally deferred until the feature architecture is frozen enough to avoid repeatedly testing moving targets.

Planned bootstrap design after the 62-feature challenger is retuned and a champion is selected:

1. Rank by raw XGBoost score within each daily slate.
2. Preserve all batters from sampled clusters rather than resampling batter rows independently.
3. Primary cluster unit: **game date / slate day**, because the deployed decision is a within-day ranking and all candidates on a slate share the cutoff competition.
4. Secondary sensitivity analysis: **game-level cluster bootstrap**, preserving the 18 batter rows per game.
5. Recompute each model's daily top-5% independently inside every bootstrap replicate.
6. Also recompute top-4/day, because that is the closest match to the actual betting workflow.
7. Use at least 10,000 paired replicates with a fixed RNG seed.
8. Report observed delta, bootstrap median/mean delta, 95% percentile CI, and Pr(delta > 0).
9. Report 2023, 2024, and combined 2023-2024 sensitivity results.
10. Do not touch 2025.

## Current status

- trusted data/target pipeline: green
- 73-feature aggressive model: frozen reference
- broad family ablations: complete
- fine-grained subfamily ablations: complete
- targeted pruning: complete
- 62-feature challenger: **eligible for full aggressive retune**
- paired tail bootstrap: **next after champion selection**
- 2025 holdout: **sealed**
