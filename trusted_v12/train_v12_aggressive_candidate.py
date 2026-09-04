"""Trusted v1.2 aggressive trainer.

No 2025 data is read.

Development design
------------------
Optuna hyperparameters must survive three chronological tuning folds:
  train 2015-2018 -> score 2019
  train 2015-2019 -> score 2020
  train 2015-2020 -> score 2021

Each fold fits its own imputation vector from prior years only. XGBoost early
stopping is aligned to Brier loss by using RMSE on binary probabilities
(minimizing RMSE is equivalent to minimizing Brier within a fold).

The best hyperparameter set minimizes mean fold Brier across 2019-2021. A
single boosting-round count is frozen as the median best round of the three
folds for the winning trial.

Then:
  2015-2021  base-model refit with frozen hyperparameters/rounds
  2022       isotonic calibration fit only
  2023-2024  development assessment (outside tuning/calibration)

2023-2024 are intentionally called development assessment, not a final
holdout: their results are now being used to make feature/model decisions.
The sealed 2025 holdout remains unread for the eventual one-shot final test.

After development assessment, a future-scoring artifact is refit:
  2015-2023  base model
  2024       isotonic calibration
No performance claim is made on 2024 as a production-calibration sample.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = "/workspace/hr_model"
FEAT_DIR = os.path.join(ROOT, "features/v1.2_trusted")
FEAT = os.path.join(FEAT_DIR, "game_features.parquet")
FLIST = os.path.join(FEAT_DIR, "feature_list.json")
OUT = os.path.join(ROOT, "models/v1.2_trusted")
REPORTS = os.path.join(ROOT, "reports")
TRIALS = os.path.join(OUT, "xgb_trials.csv")
SEED = 42
N_TRIALS = 50
TUNE_YEARS = (2019, 2020, 2021)


def log(msg: str) -> None:
    print(f"[train_v12_aggressive {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def top_k_hit_rate(y: np.ndarray, p: np.ndarray, frac: float = 0.05) -> float:
    n = max(1, int(np.ceil(len(p) * frac)))
    idx = np.argsort(-p)[:n]
    return float(np.mean(y[idx]))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "top5pct": top_k_hit_rate(y, p, 0.05),
    }


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        m = b == i
        if not m.any():
            continue
        pred = float(np.mean(p[m]))
        obs = float(np.mean(y[m]))
        n = int(m.sum())
        ece += (n / len(y)) * abs(pred - obs)
        rows.append({"bin": i, "n": n, "mean_pred": pred, "observed_rate": obs, "abs_gap": abs(pred - obs)})
    return pd.DataFrame(rows), float(ece)


@dataclass
class Matrix:
    X: np.ndarray
    y: np.ndarray


def to_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    means: np.ndarray | None = None,
) -> tuple[Matrix, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["hr_in_game"].to_numpy(dtype=np.int8, copy=True)
    if means is None:
        means = np.nanmean(X, axis=0).astype(np.float32)
    if np.isnan(means).any():
        bad = [feature_cols[i] for i in np.where(np.isnan(means))[0]]
        raise RuntimeError(f"all-NaN feature columns: {bad}")
    rr, cc = np.where(np.isnan(X))
    if len(rr):
        X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError("non-finite values after imputation")
    return Matrix(X=X, y=y), means


def fit_lr(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]):
    tr, means = to_matrix(train, feature_cols)
    te, _ = to_matrix(test, feature_cols, means)
    mu = tr.X.mean(axis=0).astype(np.float32)
    sd = tr.X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    tr.X = (tr.X - mu) / sd
    te.X = (te.X - mu) / sd
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    model.fit(tr.X, tr.y)
    p = model.predict_proba(te.X)[:, 1]
    joblib.dump(
        {"model": model, "col_means": means, "mean": mu, "std": sd, "feature_cols": feature_cols},
        os.path.join(OUT, "lr_evaluation.joblib"),
    )
    return p


def xgb_params(best: dict) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": -1,
        **best,
    }


def prepare_tune_folds(df: pd.DataFrame, feature_cols: list[str]):
    folds = []
    for score_year in TUNE_YEARS:
        tr_df = df[df.year < score_year].copy()
        va_df = df[df.year == score_year].copy()
        if tr_df.empty or va_df.empty:
            raise RuntimeError(f"empty tuning fold for {score_year}")
        tr, means = to_matrix(tr_df, feature_cols)
        va, _ = to_matrix(va_df, feature_cols, means)
        fold = {
            "year": score_year,
            "train_years": f"{int(tr_df.year.min())}-{int(tr_df.year.max())}",
            "n_train": int(len(tr_df)),
            "n_val": int(len(va_df)),
            "train": tr,
            "val": va,
            "dtrain": xgb.DMatrix(tr.X, label=tr.y, feature_names=feature_cols),
            "dval": xgb.DMatrix(va.X, label=va.y, feature_names=feature_cols),
        }
        folds.append(fold)
        log(
            f"tune fold {score_year}: train {fold['train_years']} "
            f"({fold['n_train']:,}) -> score {score_year} ({fold['n_val']:,})"
        )
    return folds


def tune_xgb_walkforward(df: pd.DataFrame, feature_cols: list[str], n_trials: int = N_TRIALS):
    folds = prepare_tune_folds(df, feature_cols)
    trial_rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        best = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }
        params = xgb_params(best)

        scores: list[float] = []
        rounds: list[int] = []
        row = {"trial": int(trial.number), **best}
        for fold in folds:
            year = int(fold["year"])
            bst = xgb.train(
                params,
                fold["dtrain"],
                num_boost_round=2000,
                evals=[(fold["dval"], f"{year}_tune")],
                early_stopping_rounds=60,
                verbose_eval=False,
            )
            br = int(bst.best_iteration + 1)
            p = bst.predict(fold["dval"], iteration_range=(0, br))
            score = float(brier_score_loss(fold["val"].y, p))
            scores.append(score)
            rounds.append(br)
            row[f"brier_{year}"] = score
            row[f"best_round_{year}"] = br

        mean_brier = float(np.mean(scores))
        std_brier = float(np.std(scores))
        median_round = int(np.median(rounds))
        row["mean_brier"] = mean_brier
        row["std_brier"] = std_brier
        row["worst_brier"] = float(np.max(scores))
        row["median_best_round"] = median_round
        trial_rows.append(row)

        trial.set_user_attr("fold_scores", {str(y): s for y, s in zip(TUNE_YEARS, scores)})
        trial.set_user_attr("fold_rounds", {str(y): r for y, r in zip(TUNE_YEARS, rounds)})
        trial.set_user_attr("median_best_round", median_round)
        return mean_brier

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = pd.DataFrame(trial_rows).sort_values(["mean_brier", "trial"]).reset_index(drop=True)
    trials_df.to_csv(TRIALS, index=False)

    best = dict(study.best_params)
    rounds = int(study.best_trial.user_attrs["median_best_round"])
    fold_scores = dict(study.best_trial.user_attrs["fold_scores"])
    fold_rounds = dict(study.best_trial.user_attrs["fold_rounds"])
    log(
        f"best {len(TUNE_YEARS)}-fold Optuna mean Brier={study.best_value:.6f}; "
        f"median round={rounds}; fold Brier={fold_scores}; fold rounds={fold_rounds}; params={best}"
    )
    return best, rounds, float(study.best_value), fold_scores, fold_rounds


def fit_base_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    best: dict,
    rounds: int,
):
    tr, means = to_matrix(train_df, feature_cols)
    pr, _ = to_matrix(pred_df, feature_cols, means)
    dtr = xgb.DMatrix(tr.X, label=tr.y, feature_names=feature_cols)
    dpr = xgb.DMatrix(pr.X, feature_names=feature_cols)
    bst = xgb.train(xgb_params(best), dtr, num_boost_round=rounds, verbose_eval=False)
    return bst, means, bst.predict(dpr), pr.y


def walk_forward(df: pd.DataFrame, feature_cols: list[str], best: dict, rounds: int) -> pd.DataFrame:
    rows = []
    for year in range(2019, 2025):
        tr_df = df[df.year < year]
        te_df = df[df.year == year]
        if tr_df.empty or te_df.empty:
            continue
        tr, means = to_matrix(tr_df, feature_cols)
        te, _ = to_matrix(te_df, feature_cols, means)
        bst = xgb.train(
            xgb_params(best),
            xgb.DMatrix(tr.X, label=tr.y, feature_names=feature_cols),
            num_boost_round=rounds,
            verbose_eval=False,
        )
        p = bst.predict(xgb.DMatrix(te.X, feature_names=feature_cols))
        m = metrics(te.y, p)
        m.update({"year": year, "n_train": len(tr_df)})
        rows.append(m)
        log(f"WF {year}: Brier={m['brier']:.5f} AUC={m['auc']:.4f} top5={m['top5pct']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    with open(FLIST) as fp:
        feature_cols = json.load(fp)

    needed = list(
        dict.fromkeys(
            feature_cols
            + ["hr_in_game", "game_pk", "game_date", "year", "split", "batter_id", "park_id"]
        )
    )
    df = pd.read_parquet(FEAT, columns=needed)
    if not df.year.between(2015, 2024).all():
        raise RuntimeError("feature years escaped 2015-2024")
    nonnumeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric:
        raise RuntimeError(f"active features must be numeric: {nonnumeric}")
    if df.duplicated(["game_pk", "batter_id"]).any():
        raise RuntimeError("duplicate batter-game rows")

    cal = df[df.year == 2022].copy()
    dev = df[df.year.isin([2023, 2024])].copy()
    if cal.empty or dev.empty:
        raise RuntimeError("empty calibration/development partition")
    log(f"active features: {len(feature_cols)}")
    log(f"calibration: {len(cal):,} rows, 2022")
    log(f"development assessment: {len(dev):,} rows, 2023-2024")
    log("sealed final holdout: 2025 (NOT READ)")

    p_lr = fit_lr(df[df.year <= 2022].copy(), dev, feature_cols)
    lr_m = metrics(dev.hr_in_game.to_numpy(int), p_lr)

    best, rounds, tune_mean_brier, tune_fold_scores, tune_fold_rounds = tune_xgb_walkforward(
        df[df.year <= 2021].copy(), feature_cols
    )

    base_eval, means_eval, p_cal_raw, y_cal = fit_base_predict(
        df[df.year <= 2021].copy(), cal, feature_cols, best, rounds
    )
    iso_eval = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal_raw, y_cal)
    dv, _ = to_matrix(dev, feature_cols, means_eval)
    p_dev_raw = base_eval.predict(xgb.DMatrix(dv.X, feature_names=feature_cols))
    p_dev_cal = iso_eval.predict(p_dev_raw)
    raw_m = metrics(dv.y, p_dev_raw)
    cal_m = metrics(dv.y, p_dev_cal)
    rel, ece = reliability(dv.y, p_dev_cal, 10)
    rel.to_csv(os.path.join(OUT, "reliability_test_2023_2024.csv"), index=False)

    joblib.dump(
        {
            "model": base_eval,
            "col_means": means_eval,
            "feature_cols": feature_cols,
            "best_params": best,
            "best_round": rounds,
            "tune_folds": list(TUNE_YEARS),
            "n_trials": N_TRIALS,
        },
        os.path.join(OUT, "xgb_evaluation.joblib"),
    )
    joblib.dump({"model": iso_eval}, os.path.join(OUT, "isotonic_evaluation.joblib"))

    per_year = []
    for year in [2023, 2024]:
        m = dev.year.to_numpy() == year
        mm = metrics(dv.y[m], p_dev_cal[m])
        mm["year"] = year
        per_year.append(mm)
    pd.DataFrame(per_year).to_csv(os.path.join(OUT, "test_metrics_by_year.csv"), index=False)

    wf = walk_forward(df, feature_cols, best, rounds)
    wf.to_csv(os.path.join(OUT, "walkforward_2019_2024.csv"), index=False)

    prod_train = df[df.year <= 2023].copy()
    prod_cal = df[df.year == 2024].copy()
    prod_bst, prod_means, p_prod_cal, y_prod_cal = fit_base_predict(
        prod_train, prod_cal, feature_cols, best, rounds
    )
    prod_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        p_prod_cal, y_prod_cal
    )
    joblib.dump(
        {
            "model": prod_bst,
            "col_means": prod_means,
            "feature_cols": feature_cols,
            "best_params": best,
            "best_round": rounds,
            "trained_through_year": 2023,
            "calibration_year": 2024,
            "tune_folds": list(TUNE_YEARS),
            "n_trials": N_TRIALS,
        },
        os.path.join(OUT, "xgb_production.joblib"),
    )
    joblib.dump(
        {"model": prod_iso, "calibration_year": 2024},
        os.path.join(OUT, "isotonic_production.joblib"),
    )

    prevalence = float(np.mean(dv.y))
    no_skill_brier = prevalence * (1.0 - prevalence)

    design = {
        "tune_selection": "walk-forward-2019-2021",
        "tune_folds": [
            {"train": "2015-2018", "score": "2019"},
            {"train": "2015-2019", "score": "2020"},
            {"train": "2015-2020", "score": "2021"},
        ],
        "tune_objective": "mean-fold-brier",
        "early_stopping_metric": "rmse=sqrt(brier)",
        "n_trials": N_TRIALS,
        "calibration_fit": "2022",
        "development_assessment": "2023-2024",
        "assessment_is_final_holdout": False,
        "sealed_final_holdout": "2025",
        "production_train": "2015-2023",
        "production_calibration": "2024",
        "holdout_2025_read": False,
    }
    result = {
        "design": design,
        "n_features": len(feature_cols),
        "best_params": best,
        "best_round": rounds,
        "tune_mean_brier": tune_mean_brier,
        "tune_fold_brier": tune_fold_scores,
        "tune_fold_best_rounds": tune_fold_rounds,
        "lr_test": lr_m,
        "xgb_raw_test": raw_m,
        "xgb_calibrated_test": cal_m,
        "test_ece": ece,
        "test_no_skill_brier": no_skill_brier,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fp:
        json.dump(result, fp, indent=2)
    with open(os.path.join(OUT, "feature_list.json"), "w") as fp:
        json.dump(feature_cols, fp, indent=2)

    report = [
        "# Trusted v1.2 aggressive training report",
        "",
        "**2025 was not read or evaluated. It remains the sealed final holdout.**",
        "",
        "## Chronological development design",
        "",
        "- Optuna: 50 TPE trials",
        "- Fold 1: train 2015-2018 -> score 2019",
        "- Fold 2: train 2015-2019 -> score 2020",
        "- Fold 3: train 2015-2020 -> score 2021",
        "- Objective: mean fold Brier",
        "- Early stopping: RMSE = sqrt(Brier), aligned to the probability objective",
        f"- Frozen rounds: median winning-fold best round = {rounds}",
        "- 2022: isotonic calibration fit only",
        "- 2023-2024: development assessment (not final holdout)",
        "- Production refit: base 2015-2023, calibration 2024",
        "",
        "## Winning Optuna configuration",
        "",
        f"- mean tuning Brier: {tune_mean_brier:.6f}",
        f"- fold Brier: {tune_fold_scores}",
        f"- fold best rounds: {tune_fold_rounds}",
        f"- params: `{best}`",
        "",
        "## 2023-2024 development assessment",
        "",
        f"- LR: {lr_m}",
        f"- XGB raw: {raw_m}",
        f"- XGB calibrated: {cal_m}",
        f"- calibrated ECE: {ece:.6f}",
        f"- no-skill prevalence Brier: {no_skill_brier:.6f}",
        "",
    ]
    with open(os.path.join(REPORTS, "model_trainer_v12_trusted_report.md"), "w") as fp:
        fp.write("\n".join(report))

    log(
        f"development calibrated: Brier={cal_m['brier']:.5f} "
        f"AUC={cal_m['auc']:.4f} top5={cal_m['top5pct']:.4f} ECE={ece:.5f}"
    )
    log("aggressive trusted training complete; 2025 remains sealed")


if __name__ == "__main__":
    main()
