# Trusted v1.2 feature-family ablation checkpoint — 2026-09-04

## Scope

Development-only analysis on 2015-2024. **2025 was not read or evaluated and remains the sealed final holdout.**

Reference model: 73-feature trusted v1.2 aggressive XGBoost candidate with frozen Optuna-selected hyperparameters and 194 boosting rounds.

All family ablations were deliberately **not retuned**. Each removal used the same frozen XGBoost hyperparameters/round count, trained through 2021, fit isotonic calibration on 2022, and assessed on 2023-2024.

Ranking diagnostics use raw XGBoost score. Isotonic calibration is evaluated separately so probability ties cannot change ranking membership.

## Broad family ablation

| Model | Features | Cal Brier | Cal AUC | ECE | Pooled raw top-5 HR rate |
|---|---:|---:|---:|---:|---:|
| full_73 | 73 | 0.102500 | 0.6175 | 0.00626 | 22.22% |
| no_all_qoc | 53 | 0.102719 | 0.6114 | 0.00802 | 21.97% |
| no_barrel | 68 | 0.102547 | 0.6166 | 0.00639 | 22.22% |
| no_xwoba | 69 | 0.102478 | 0.6176 | 0.00635 | 22.16% |
| no_pitcher_qoc | 67 | 0.102507 | 0.6167 | 0.00646 | 22.20% |
| no_batter_qoc | 59 | 0.102739 | 0.6113 | 0.00805 | 21.95% |
| no_recent_qoc_30d | 59 | 0.102580 | 0.6163 | 0.00696 | 22.32% |
| no_long_horizon_qoc | 67 | 0.102617 | 0.6151 | 0.00723 | 21.78% |
| no_pitch_type_matchup | 44 | 0.102509 | 0.6167 | 0.00606 | 22.06% |
| no_park | 71 | 0.102553 | 0.6170 | 0.00645 | 21.39% |

Pooled percentiles can change their year composition at the cutoff. For example, removing the 30-day QoC block slightly improves the combined pooled top-5 while reducing top-5 hit rate in each year separately. Therefore pooled top-percentile results are secondary to the daily-slate diagnostics below.

## Daily-slate ranking diagnostics

Candidates are ranked independently on each game date by raw XGBoost score.

| Model | Daily top-5% | Daily top-2% | Top 1/day | Top 2/day | Top 4/day | Top 8/day | Top-4 retention vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_73 | 21.69% | 23.60% | 26.46% | 25.77% | 24.23% | 22.67% | — |
| no_barrel | 21.56% | 24.17% | 27.58% | 25.07% | 24.03% | 22.53% | 79.5% |
| no_all_qoc | 21.74% | 23.60% | 24.51% | 25.07% | 23.96% | 21.87% | 59.4% |
| no_xwoba | 21.91% | 23.39% | 26.46% | 25.63% | 23.89% | 22.91% | 82.6% |
| no_batter_qoc | 21.71% | 23.54% | 24.23% | 25.35% | 23.75% | 22.35% | 60.4% |
| no_pitcher_qoc | 22.00% | 23.54% | 27.30% | 26.04% | 23.47% | 22.91% | 82.8% |
| no_long_horizon_qoc | 21.67% | 23.91% | 26.46% | 23.96% | 23.47% | 22.67% | 71.9% |
| no_pitch_type_matchup | 21.49% | 23.70% | 26.74% | 25.63% | 23.33% | 22.46% | 80.9% |
| no_recent_qoc_30d | 22.03% | 23.39% | 27.30% | 24.09% | 22.49% | 22.77% | 77.0% |
| no_park | 20.97% | 22.40% | 23.68% | 24.79% | 22.35% | 22.11% | 70.5% |

Full-model daily-ranking stability:
- 2023: daily top-5% 22.20%, daily top-2% 23.12%, top-4/day 24.58%.
- 2024: daily top-5% 21.17%, daily top-2% 24.08%, top-4/day 23.88%.

## Findings at this checkpoint

### Strong keep: park

The two venue-aware park features are disproportionately valuable to the practical shortlist. Removing them reduces top-4/day from 24.23% to 22.35%, top-1/day from 26.46% to 23.68%, and daily top-5% from 21.69% to 20.97%. Only 70.5% of the full-model top-4 names survive without park.

### QoC is useful, especially batter QoC, but its subfeatures are mixed

Removing all 20 QoC features materially degrades AUC and ECE and changes 40.6% of top-4/day candidate identities. Removing batter QoC similarly degrades global discrimination/calibration and practical top-end rankings. The family should not be removed wholesale.

Long-horizon QoC appears more consistently useful than the 30-day block. The 30-day block moves signal between different shortlist depths rather than showing a uniform benefit or harm, so it requires finer decomposition before any cut.

### Barrel and xwOBA should not be judged by one aggregate number

Removing barrel mildly worsens global Brier/AUC and most broader ranking views, but some very narrow daily buckets improve. Removing xwOBA slightly improves aggregate Brier/AUC while slightly hurting the practical top-4/day shortlist. Both families remain plausible contributors with redundancy/interactions; horizon-level decomposition is required.

### Pitch-type block is mixed, not useless

Removing all 29 pitch-matchup features barely changes global Brier/ECE but reduces top-4/day from 24.23% to 23.33%. That suggests the block may contain useful ranking features mixed with redundant/noisy features. It should be decomposed into batter-vs-pitch HR rates, pitcher-vs-pitch HR rates, pitch usage, and top-pitch interaction before any pruning.

## Decision

No features are removed at this checkpoint. Proceed to fine-grained frozen-parameter subfamily ablations, then reassess candidate architecture. Bootstrap inference remains deferred until the ranking architecture is more stable.
