# v1.2 pre-2025 development reset review — 2026-09-04

## Purpose

Reset after the trusted rebuild, feature-family ablations, aggressive tuning, tail bootstrap, disagreement analysis, seasonality test, and yearly contextual-migration study.

This document distinguishes:

- **established development evidence**;
- **failed / demoted hypotheses**;
- **descriptive mechanism evidence**;
- **remaining decisions required before opening 2025**.

It is intentionally written before the sealed 2025 holdout is opened.

**2025 remains sealed.**

---

## 1. Frozen trustworthy data/model foundation

### Trusted target universe

- seasons: 2015–2024 only;
- authoritative MLB regular-season game identity / actual venue identity;
- 22,596 games;
- exactly 18 historical starter-lineup proxy rows per game;
- 406,728 batter-game rows;
- outcome = HR in game;
- 2025 never requested/read by trusted rebuild workflows.

### Frozen active architecture

**73 numeric features**:

- 53 trusted repaired core features;
- 20 QoC / Statcast contact-quality features.

Raw XGBoost score is the canonical ranking score. Isotonic calibration is evaluated separately and must not determine exact top-k membership.

### Frozen aggressive champion contract

- 50-trial TPE search over expanding chronological tuning folds;
- selected trial 47;
- frozen hyperparameters in `experiments/contracts/v12_full73_aggressive.json`;
- frozen boosting rounds: **194**.

Development 2023–24:

- calibrated Brier: **0.1025000**;
- calibrated AUC: **0.617512**;
- raw AUC: **0.617935**;
- AP: approximately **0.168** calibrated / **0.171** raw;
- ECE: **0.00626**;
- raw pooled top-5%: approximately **22.22%**;
- daily top-5%: **21.69%**;
- daily top-4: **24.23%**.

---

## 2. Feature-ablation verdict

### What survived

The full feature-rich architecture survived broad family destruction, fine-grained subfamily destruction, targeted pruning, and an equal-retune challenger.

Most important practical evidence:

- remove park → daily top-4 **24.23% → 22.35%**;
- remove all QoC → discrimination/calibration worsen and top-tail composition changes materially;
- remove pitcher season barrel → top-4 **24.23% → 23.33%**;
- remove batter-vs-pitch HR rates → top-4 **24.23% → 23.54%**;
- remove dynamic top-pitch pair → top-4 **24.23% → 23.68%**;
- remove 30d batter barrel → top-4 **24.23% → 23.82%**.

The apparent 62-feature pruning improvement under borrowed full73 hyperparameters did **not** survive fair retuning:

| Model | Daily top-5% | Daily top-4 | Cal Brier | Cal AUC | ECE |
|---|---:|---:|---:|---:|---:|
| full73 champion | **21.69%** | **24.23%** | **.102500** | **.61751** | **.00626** |
| tuned 62-feature challenger | 21.65% | 23.82% | .102554 | .61671 | .00643 |

### Decision

**Full73 remains champion.**

No feature family is removed before 2025.

The pruning false-positive disappearing after equal retuning is positive evidence that the development procedure is resisting feature-cut overfit rather than selecting whatever temporarily spikes the shortlist.

---

## 3. Strongest product-level development finding

The most robust contextual signal is not a special sleeper bucket. It is the broader **daily top-5% candidate board**.

Against the frozen long-horizon batter-only obvious-power proxy on 2023–24:

- obvious-power top-5 HR rate: **20.28%**;
- full73 top-5 HR rate: **21.69%**;
- lift: **+1.41 percentage points**;
- paired 10,000-slate bootstrap 95% CI: **+0.42 to +2.39 pp**;
- `P(lift > 0) ≈ 99.7%`.

This is the cleanest evidence that the extra contextual features buy ranking information beyond simply selecting the strongest long-horizon power bats.

### Important limitation

At exactly top-4/day, obvious-power was slightly better in the combined 2023–24 comparison. Therefore the evidence does **not** support a blanket claim that full73 is superior at every possible candidate-pool size.

The current product evidence is strongest around the broader daily top-5% board.

---

## 4. Disagreement / hidden-bucket work

### Initial signal

The first same-band disagreement analysis suggested large effects when full73 promoted hitters ranked farther down the obvious-power board, especially 17+.

### Adversarial correction

Nearest-rank same-slate matching reduced the 2023–24 17+ lift to:

- selected: **27.12%**;
- nearest-rank control: **18.08%**;
- observed lift: **+9.04 pp**;
- 95% CI: **-0.56 to +18.24 pp**.

It cleared the precommitted +5 pp magnitude floor but **failed the CI-above-zero requirement**.

### 2022 freshness replay

The exact same 17+ rule did not replicate:

- matched lift in 2022: **-1.08 pp**.

A different 5–8 band looked strong in 2022, but switching to it after seeing the result would be subgroup chasing.

### Decision

**No 5–8 / 9–16 / 17+ hidden-pick rule is promoted.**

The point estimate behavior is interesting, but bucket location is not stable enough to operationalize from <=2024 development evidence.

---

## 5. Season-phase test

Because monthly disagreement cells were underpowered, the predeclared seasonality test used the powered daily top-5% full73-vs-obvious comparison and only a FIRST_HALF / SECOND_HALF split.

Results:

| Phase | Obvious | Full73 | Contextual lift |
|---|---:|---:|---:|
| Opening Day–June 30 | 19.30% | **20.67%** | **+1.37 pp** |
| July 1–end | 21.28% | **22.73%** | **+1.45 pp** |

Half interaction:

- SECOND minus FIRST lift: **+0.07 pp**;
- 95% CI: **-1.88 to +2.04 pp**.

### Data-maturity audit

Rolling-history support is materially thinner early in the season, especially pitcher/top-pitch support, while the raw HR environment is stronger later.

Despite those changes, the contextual top-5 lift is essentially unchanged.

### Decision

No evidence that the broad contextual edge is meaningfully stronger in one half of the season.

No monthly outcome drill-down was triggered.

---

## 6. Yearly contextual-migration test

The user's original migration hypothesis was tested continuously rather than with fixed rank buckets.

For each 2019–2024 season, the frozen 73-feature / 194-round architecture was refit on 2015 through year-1 and scored raw on the next year. The obvious-power proxy used only prior-year-available medians.

### Observed behavior

| Year | Top-5 overlap | Full73-only mean obvious depth |
|---|---:|---:|
| 2019 | 52.52% | 14.68% |
| 2020 | 53.46% | 16.17% |
| 2021 | 58.90% | 11.85% |
| 2022 | 58.42% | 12.46% |
| 2023 | 61.32% | 11.03% |
| 2024 | 63.24% | 11.75% |

Depth trend:

- slope: about **-0.842 rank-percentile pp/year**;
- Spearman: **-0.829**;
- 10k slate-date bootstrap 95% CI: **-1.020 to -0.669 pp/year**.

Overlap trend:

- slope: about **+2.19 pp/year**;
- Spearman: **+0.943**;
- bootstrap 95% CI: **+1.76 to +2.64 pp/year**.

The result survives the predeclared sensitivity excluding 2020.

### Decision

The broad historical trend is **convergence toward obvious-power**, not progressive migration deeper into the obvious-power board.

Therefore do not carry a prospective 2025 hypothesis that the strongest contextual signal should continue moving toward 17+ or other increasingly deep ranks.

---

## 7. Mechanism: what tends to promote genuinely deeper hitters?

Among full73-only top-5 selections, association between contextual composite and obvious-rank depth:

- park: **+0.148**;
- pitcher vulnerability: **+0.085**;
- pitch matchup: approximately **0.000**;
- recent batter form: **-0.258**.

Interpretation:

- favorable **park** is the clearest marker of deeper contextual promotions;
- pitcher vulnerability contributes more weakly;
- broad pitch-matchup score does not determine how far down the obvious board the pick came from;
- recent batter form is associated with differentiated picks that are *closer* to obvious-power, not the deepest sleepers.

This is a mechanism explanation only, not an independent proof of edge.

---

## 8. What is established versus what is not

### Established / strongest development evidence

1. The trusted 73-feature architecture is technically and temporally much cleaner than the inherited M3 build.
2. The full73 architecture survived aggressive ablation and equal-retune pruning pressure.
3. Park, QoC/contact quality, and several opponent/pitch-context components materially affect the practical shortlist.
4. The full73 daily top-5% board has resolved contextual lift over the obvious-power proxy on 2023–24.
5. That broad lift is approximately stable across first versus second half despite large changes in HR environment and feature-history maturity.
6. No evidence supports replacing full73 with the leaner 62-feature challenger.

### Interesting but not operationally established

1. A fixed non-obvious disagreement bucket with materially higher HR rate.
2. A 17+ sleeper rule.
3. A month/season-phase regime selector.
4. A trend toward progressively deeper contextual selections.
5. A static mechanism saying one context family alone identifies hidden HR bets.

### Falsified/demoted development hypotheses

1. 'Prune usage + long batter xwOBA and the shortlist gets better' — disappeared after fair retuning.
2. '17+ is a stable hidden-pick bucket' — unresolved after matching and failed 2022 replication.
3. 'The broad contextual edge is mainly a second-half phenomenon' — no half interaction.
4. 'The contextual model has been migrating progressively deeper into the obvious-power board' — six-year continuous test shows the opposite.

---

## 9. Remaining caveats before 2025

### Development reuse

2023–24 has now been used for architecture, ablations, tail inference, disagreement diagnostics, and seasonality. It is development evidence, not a pristine final holdout.

2019–22 are also selection-touched in various ways through the tuning/calibration architecture. The yearly walk-forward migration test is therefore a structural retrospective diagnostic, not six independent pristine holdouts.

This makes the still-sealed 2025 one-shot especially important.

### Market profitability is not yet tested

HR hit-rate concentration does not prove positive expected value at sportsbook prices. The current model is a **candidate-ranking model**. Market price/EV should be a separate downstream layer after predictive holdout validity is established.

### Absolute top-4 versus top-5 behavior

The contextual advantage is strongest at the broader top-5% candidate board. Exact top-4 performance is noisier and can differ from the broader tail.

Therefore do not redefine the primary 2025 test around whichever exact-N bucket looks best after opening the holdout.

---

## 10. Proposed freeze before 2025

Before any 2025 file is read, explicitly freeze the following.

### Model architecture

- full trusted **73 features**;
- frozen champion hyperparameters;
- **194 rounds**;
- no further feature additions/removals before the one-shot.

### Final pre-2025 fit

Proposed production-like fit already established in the methodology plan:

- base XGBoost fit: **2015–2023**;
- imputation from 2015–2023 only;
- isotonic calibration: **2024 only**;
- raw score remains the ranking score;
- calibrated score used for probabilistic/calibration metrics.

No 2025 outcomes may influence fitting, imputation, calibration, or threshold choice.

### Proposed holdout evaluation hierarchy

#### Primary product question

**Does full73 retain useful daily top-5% HR concentration on sealed 2025?**

Report:

- full73 daily top-5% HR rate;
- obvious-power daily top-5% HR rate;
- paired slate-date lift and 95% CI;
- selected counts and slate counts.

Do not create a new rank-disagreement subgroup from 2025 to rescue or enhance the result.

#### Core model-quality metrics

Also report, without retuning:

- raw and calibrated Brier;
- raw and calibrated AUC;
- AP;
- log loss;
- ECE / calibration-in-the-large / slope if frozen harness supports them;
- raw daily top-10%, top-5%, top-2%, top-1%;
- raw top-1/day, top-2/day, top-4/day, top-8/day.

These are diagnostic/supporting metrics. The primary product interpretation should remain centered on the predeclared top-5% candidate board rather than whichever secondary tail number is highest in 2025.

### 2025 disagreement/migration readout

After the primary holdout metrics are locked, it is acceptable to **describe**:

- top-5 overlap with obvious-power;
- continuous obvious-rank depth of full73-only selections;
- whether 2025 continues the historical convergence direction.

This must remain secondary and cannot redefine success/failure of the primary 2025 holdout.

---

## 11. Readiness assessment

### Architecture readiness

**READY TO FREEZE.**

The full73 architecture has survived enough adversarial pressure. Further <=2024 feature/subgroup mining now has more risk of development overfit than expected information gain.

### Evaluation readiness

**NEARLY READY, but the exact 2025 execution/evaluation contract should be written and reviewed once before authorization.**

The remaining work should be administrative/statistical freeze work, not new feature discovery:

1. write exact one-shot 2025 executor/evaluation contract;
2. assert 2015–2023 fit + 2024 calibration;
3. assert raw-score ranking semantics;
4. assert primary daily top-5% comparator and paired bootstrap;
5. assert frozen secondary metrics;
6. fail closed if any code attempts post-hoc tuning or 2025-driven thresholding;
7. review the contract;
8. only then obtain explicit authorization to open 2025.

### Current verdict

> **V1.2_FULL73_DEVELOPMENT_COMPLETE_PENDING_2025_FREEZE_CONTRACT**

No more <=2024 signal-bucket mining is recommended before the sealed holdout.
