# Aggressive chronological tuning design

2025 remains sealed and unread.

The trusted 73-feature core+QoC model graduates from the 15-trial smoke tuner to a 50-trial Optuna TPE search. Every candidate hyperparameter set must be scored on three expanding-window folds:

- train 2015-2018 -> score 2019
- train 2015-2019 -> score 2020
- train 2015-2020 -> score 2021

Each fold fits preprocessing from that fold's training years only. XGBoost early stopping uses RMSE on binary probabilities, which is sqrt(Brier), so early stopping is aligned to the Brier objective. Optuna minimizes the mean of the three fold Brier scores. The final boosting-round count is the median of the winning trial's three fold-specific best rounds.

After tuning, the base model is fit through 2021, isotonic calibration is fit on 2022 only, and 2023-2024 are used as a development assessment. They are not described as a final holdout because their results are now being used for feature/model decisions. The eventual one-shot final holdout remains 2025.
