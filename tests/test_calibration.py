"""Test calibration on the val block."""
import os
import sys
import pandas as pd
import numpy as np
import joblib

ROOT = "/workspace/hr_model"
MODELS = os.path.join(ROOT, "models")
FEAT = os.path.join(ROOT, "features/v1/game_features.parquet")
FLIST = os.path.join(ROOT, "features/v1/feature_list.json")


def main() -> int:
    # Load model + calibrator
    xgb_pkg = joblib.load(os.path.join(MODELS, "xgb_v1.joblib"))
    iso_pkg = joblib.load(os.path.join(MODELS, "isotonic_v1.joblib"))
    bst = xgb_pkg["model"]
    iso = iso_pkg["model"]
    col_means = xgb_pkg["col_means"]
    feature_cols = xgb_pkg["feature_cols"]

    import xgboost as xgb_lib
    f = pd.read_parquet(FEAT)
    val = f[f["split"] == "val"].reset_index(drop=True)
    y_va = val["hr_in_game"].astype(int).values
    X_va = val[feature_cols].astype("float32").values
    X_va = np.where(np.isnan(X_va), col_means, X_va)
    dval = xgb_lib.DMatrix(X_va, feature_names=feature_cols)
    p_xgb = bst.predict(dval)
    p_cal = iso.predict(p_xgb)

    # ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n_total = len(p_cal)
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        if i == 9:
            mask = (p_cal >= lo) & (p_cal <= hi)
        else:
            mask = (p_cal >= lo) & (p_cal < hi)
        if mask.sum() == 0:
            continue
        ece += np.abs(p_cal[mask].mean() - y_va[mask].mean()) * mask.sum() / n_total
    print(f"ECE on val: {ece:.4f}")
    if ece > 0.02:
        print(f"FAIL: ECE {ece:.4f} > 0.02", file=sys.stderr)
        return 1
    print(f"PASS: ECE {ece:.4f} <= 0.02")

    # Reliability: each bin's predicted mean should be within ±2pp of observed
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        if i == 9:
            mask = (p_cal >= lo) & (p_cal <= hi)
        else:
            mask = (p_cal >= lo) & (p_cal < hi)
        if mask.sum() < 100:  # skip tiny bins
            continue
        gap = abs(p_cal[mask].mean() - y_va[mask].mean())
        if gap > 0.02:
            print(f"FAIL: bin {i} reliability gap {gap:.4f} > 0.02", file=sys.stderr)
            return 1
    print("PASS: reliability within ±2pp for all bins with n>=100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
