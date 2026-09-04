"""Trusted v1.2 trainer with separated tuning, calibration, and assessment.

No 2025 data is read.

Chronological evaluation design:
  2015-2020  tuning model training
  2021       hyperparameter / early-stop selection
  2015-2021  base model refit with frozen parameters/rounds
  2022       isotonic calibration fit only
  2023-2024  final independent historical assessment only

After assessment, a separate deployable artifact is refit for future scoring:
  2015-2023  base model training with frozen parameters/rounds
  2024       isotonic calibration fit
No performance claim is made on the 2024 production-calibration rows.
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


def log(msg: str) -> None:
    print(f"[train_v12_trusted {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        pred = float(np.mean(p[m])); obs = float(np.mean(y[m])); n = int(m.sum())
        ece += (n / len(y)) * abs(pred - obs)
        rows.append({"bin": i, "n": n, "mean_pred": pred, "observed_rate": obs, "abs_gap": abs(pred-obs)})
    return pd.DataFrame(rows), float(ece)


@dataclass
class Matrix:
    X: np.ndarray
    y: np.ndarray


def to_matrix(df: pd.DataFrame, feature_cols: list[str], means: np.ndarray | None = None) -> tuple[Matrix, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["hr_in_game"].to_numpy(dtype=np.int8, copy=True)
    if means is None:
        means = np.nanmean(X, axis=0).astype(np.float32)
    if np.isnan(means).any():
        bad = [feature_cols[i] for i in np.where(np.isnan(means))[0]]
        raise RuntimeError(f"all-NaN feature columns: {bad}")
    rr, cc = np.where(np.isnan(X))
    if len(rr): X[rr, cc] = means[cc]
    if not np.isfinite(X).all():
        raise RuntimeError("non-finite values after imputation")
    return Matrix(X=X, y=y), means


def fit_lr(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]):
    tr, means = to_matrix(train, feature_cols)
    te, _ = to_matrix(test, feature_cols, means)
    mu = tr.X.mean(axis=0).astype(np.float32)
    sd = tr.X.std(axis=0).astype(np.float32); sd[sd == 0] = 1.0
    tr.X = (tr.X - mu) / sd
    te.X = (te.X - mu) / sd
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    model.fit(tr.X, tr.y)
    p = model.predict_proba(te.X)[:, 1]
    joblib.dump({"model": model, "col_means": means, "mean": mu, "std": sd, "feature_cols": feature_cols},
                os.path.join(OUT, "lr_evaluation.joblib"))
    return p


def tune_xgb(tune_train: pd.DataFrame, tune_val: pd.DataFrame, feature_cols: list[str], n_trials: int = 15):
    tr, means = to_matrix(tune_train, feature_cols)
    va, _ = to_matrix(tune_val, feature_cols, means)
    dtr = xgb.DMatrix(tr.X, label=tr.y, feature_names=feature_cols)
    dva = xgb.DMatrix(va.X, label=va.y, feature_names=feature_cols)

    trial_rows = []
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "eta": trial.suggest_float("eta", 0.02, 0.15, log=True),
            "seed": SEED,
        }
        bst = xgb.train(params, dtr, num_boost_round=1500, evals=[(dva, "2021_tune")],
                        early_stopping_rounds=50, verbose_eval=False)
        p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        score = float(brier_score_loss(va.y, p))
        trial.set_user_attr("best_round", int(bst.best_iteration + 1))
        trial_rows.append({"trial": trial.number, "brier": score, "best_round": int(bst.best_iteration+1), **trial.params})
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    pd.DataFrame(trial_rows).to_csv(TRIALS, index=False)
    best = dict(study.best_params)
    best_round = int(study.best_trial.user_attrs["best_round"])
    log(f"best 2021 tune Brier={study.best_value:.6f}; round={best_round}; params={best}")
    return best, best_round


def xgb_params(best: dict) -> dict:
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "seed": SEED, **best}


def fit_base_predict(train_df: pd.DataFrame, pred_df: pd.DataFrame, feature_cols: list[str], best: dict, rounds: int):
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
        if tr_df.empty or te_df.empty: continue
        tr, means = to_matrix(tr_df, feature_cols)
        te, _ = to_matrix(te_df, feature_cols, means)
        bst = xgb.train(xgb_params(best), xgb.DMatrix(tr.X, label=tr.y, feature_names=feature_cols),
                        num_boost_round=rounds, verbose_eval=False)
        p = bst.predict(xgb.DMatrix(te.X, feature_names=feature_cols))
        m = metrics(te.y, p); m.update({"year": year, "n_train": len(tr_df)})
        rows.append(m)
        log(f"WF {year}: Brier={m['brier']:.5f} AUC={m['auc']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True); os.makedirs(REPORTS, exist_ok=True)
    with open(FLIST) as fp: feature_cols = json.load(fp)
    needed = list(dict.fromkeys(feature_cols + ["hr_in_game", "game_pk", "game_date", "year", "split", "batter_id", "park_id"]))
    df = pd.read_parquet(FEAT, columns=needed)
    if not df.year.between(2015, 2024).all(): raise RuntimeError("feature years escaped 2015-2024")
    nonnumeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric: raise RuntimeError(f"active features must be numeric: {nonnumeric}")
    if df.duplicated(["game_pk", "batter_id"]).any(): raise RuntimeError("duplicate batter-game rows")

    tune_train = df[df.year <= 2020].copy()
    tune_val = df[df.year == 2021].copy()
    cal = df[df.year == 2022].copy()
    test = df[df.year.isin([2023, 2024])].copy()
    for name, part in [("tune_train", tune_train), ("tune_val", tune_val), ("cal", cal), ("test", test)]:
        if part.empty: raise RuntimeError(f"empty partition: {name}")
        log(f"{name}: {len(part):,} rows, years {part.year.min()}-{part.year.max()}")

    p_lr = fit_lr(df[df.year <= 2022].copy(), test, feature_cols)
    lr_m = metrics(test.hr_in_game.to_numpy(int), p_lr)

    best, rounds = tune_xgb(tune_train, tune_val, feature_cols)

    base_eval, means_eval, p_cal_raw, y_cal = fit_base_predict(
        df[df.year <= 2021].copy(), cal, feature_cols, best, rounds
    )
    iso_eval = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal_raw, y_cal)
    te, _ = to_matrix(test, feature_cols, means_eval)
    p_test_raw = base_eval.predict(xgb.DMatrix(te.X, feature_names=feature_cols))
    p_test_cal = iso_eval.predict(p_test_raw)
    raw_m = metrics(te.y, p_test_raw)
    cal_m = metrics(te.y, p_test_cal)
    rel, ece = reliability(te.y, p_test_cal, 10)
    rel.to_csv(os.path.join(OUT, "reliability_test_2023_2024.csv"), index=False)
    joblib.dump({"model": base_eval, "col_means": means_eval, "feature_cols": feature_cols,
                 "best_params": best, "best_round": rounds}, os.path.join(OUT, "xgb_evaluation.joblib"))
    joblib.dump({"model": iso_eval}, os.path.join(OUT, "isotonic_evaluation.joblib"))

    per_year = []
    for year in [2023, 2024]:
        m = test.year.to_numpy() == year
        mm = metrics(te.y[m], p_test_cal[m]); mm["year"] = year; per_year.append(mm)
    pd.DataFrame(per_year).to_csv(os.path.join(OUT, "test_metrics_by_year.csv"), index=False)

    wf = walk_forward(df, feature_cols, best, rounds)
    wf.to_csv(os.path.join(OUT, "walkforward_2019_2024.csv"), index=False)

    prod_train = df[df.year <= 2023].copy()
    prod_cal = df[df.year == 2024].copy()
    prod_bst, prod_means, p_prod_cal, y_prod_cal = fit_base_predict(prod_train, prod_cal, feature_cols, best, rounds)
    prod_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_prod_cal, y_prod_cal)
    joblib.dump({"model": prod_bst, "col_means": prod_means, "feature_cols": feature_cols,
                 "best_params": best, "best_round": rounds,
                 "trained_through_year": 2023, "calibration_year": 2024},
                os.path.join(OUT, "xgb_production.joblib"))
    joblib.dump({"model": prod_iso, "calibration_year": 2024}, os.path.join(OUT, "isotonic_production.joblib"))

    prevalence = float(np.mean(te.y))
    no_skill_brier = prevalence * (1.0 - prevalence)

    result = {
        "design": {
            "tune_train": "2015-2020", "tune_selection": "2021", "calibration_fit": "2022",
            "independent_assessment": "2023-2024", "production_train": "2015-2023",
            "production_calibration": "2024", "holdout_2025_read": False,
        },
        "n_features": len(feature_cols), "best_params": best, "best_round": rounds,
        "lr_test": lr_m, "xgb_raw_test": raw_m, "xgb_calibrated_test": cal_m,
        "test_ece": ece, "test_no_skill_brier": no_skill_brier,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fp: json.dump(result, fp, indent=2)
    with open(os.path.join(OUT, "feature_list.json"), "w") as fp: json.dump(feature_cols, fp, indent=2)

    report = [
        "# Trusted v1.2 training report", "",
        "2025 was not read or evaluated.", "",
        "## Chronological design", "",
        "- 2015-2020: tuning-model training", "- 2021: hyperparameter / early-stop selection",
        "- 2022: isotonic calibration fit", "- 2023-2024: independent assessment",
        "- Production refit: base 2015-2023, calibration 2024 (no metric claim on production calibration rows)", "",
        "## Independent 2023-2024 assessment", "",
        f"- LR: {lr_m}", f"- XGB raw: {raw_m}", f"- XGB calibrated: {cal_m}",
        f"- calibrated ECE: {ece:.6f}", f"- no-skill prevalence Brier: {no_skill_brier:.6f}", "",
        f"- frozen XGB rounds: {rounds}", f"- frozen XGB params: `{best}`", "",
    ]
    with open(os.path.join(REPORTS, "model_trainer_v12_trusted_report.md"), "w") as fp:
        fp.write("\n".join(report))
    log(f"independent test calibrated: Brier={cal_m['brier']:.5f} AUC={cal_m['auc']:.4f} top5={cal_m['top5pct']:.4f} ECE={ece:.5f}")
    log("trusted training complete")


if __name__ == "__main__":
    main()
