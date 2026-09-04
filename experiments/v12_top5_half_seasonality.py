"""Predeclared half-season analysis for the frozen full73 top-5% edge.

Primary question
----------------
Does the already-resolved daily top-5% full73-vs-obvious-power lift differ
between FIRST_HALF (through June 30) and SECOND_HALF (July 1 onward)?

This script intentionally does NOT emit calendar-month outcome tables and does
NOT analyze the sparse disagreement-rank bands. 2025 is rejected fail-closed.

Feature-maturity diagnostics are descriptive confound checks only. They do not
change the primary inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["game_pk", "batter_id"]
SEED = 20260904
TOP_FRAC = 0.05
GENUINE_SUPPORT_FEATURES = {
    "batter_pa_14d": 20.0,
    "batter_pa_30d": 20.0,
    "pitcher_pa_30d": 20.0,
    "top_pitch_total_pitches_30d": 100.0,
}


def phase_from_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s).dt.normalize()
    return pd.Series(
        np.where(d.dt.month.le(6), "FIRST_HALF", "SECOND_HALF"),
        index=s.index,
        dtype="object",
    )


def deterministic_top5_flags(frame: pd.DataFrame) -> pd.Series:
    """Canonical raw-score daily top-5%; mirrors v12_paired_tail_bootstrap."""
    out = pd.Series(False, index=frame.index, dtype=bool)
    for _, g in frame.groupby("game_date", sort=True):
        order = np.lexsort((
            g.batter_id.to_numpy(dtype=np.int64),
            g.game_pk.to_numpy(dtype=np.int64),
            -g.p_raw.to_numpy(dtype=np.float64),
        ))
        n = max(1, int(np.ceil(len(g) * TOP_FRAC)))
        out.loc[g.index[order[:n]]] = True
    return out


def validate_pair(obvious: pd.DataFrame, full73: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = {"game_pk", "batter_id", "game_date", "year", "hr_in_game", "p_raw"}
    for name, f in [("obvious", obvious), ("full73", full73)]:
        missing = need - set(f.columns)
        if missing:
            raise RuntimeError(f"{name} missing columns: {sorted(missing)}")
        if not f.year.between(2023, 2024).all():
            raise RuntimeError(f"{name} escaped frozen 2023-2024 development years")
        if 2025 in set(f.year.astype(int)):
            raise RuntimeError("2025 present; sealed holdout violation")
        if f.duplicated(KEYS).any():
            raise RuntimeError(f"{name} duplicate batter-game keys")
        f = f.copy()
        f["game_date"] = pd.to_datetime(f.game_date).dt.normalize()
        if name == "obvious":
            obvious = f
        else:
            full73 = f

    a = obvious.sort_values(KEYS).reset_index(drop=True)
    b = full73.sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(b[KEYS]):
        raise RuntimeError("paired inputs do not share identical batter-game keys")
    for col in ["hr_in_game", "year", "game_date"]:
        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            raise RuntimeError(f"paired inputs disagree on {col}")
    return a, b


def daily_selected_counts(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["selected_top5"] = deterministic_top5_flags(f)
    rows = []
    for day, g in f.groupby("game_date", sort=True):
        s = g[g.selected_top5]
        rows.append({
            "game_date": pd.Timestamp(day),
            "year": int(g.year.iloc[0]),
            "phase": str(g.phase.iloc[0]),
            "successes": int(s.hr_in_game.sum()),
            "n": int(len(s)),
        })
    return pd.DataFrame(rows)


def _rate(successes: np.ndarray, n: np.ndarray) -> float:
    return float(np.sum(successes) / np.sum(n))


def phase_bootstrap(
    obvious_daily: pd.DataFrame,
    full_daily: pd.DataFrame,
    phase: str,
    reps: int,
    seed: int,
) -> dict:
    a = obvious_daily[obvious_daily.phase.eq(phase)]
    b = full_daily[full_daily.phase.eq(phase)]
    m = a.merge(
        b, on=["game_date", "year", "phase"], suffixes=("_obvious", "_full73"),
        validate="one_to_one",
    )
    if len(m) != len(a) or len(m) != len(b):
        raise RuntimeError(f"paired slate dates differ in {phase}")
    if len(m) < 100:
        raise RuntimeError(f"too few combined slate dates for powered phase test: {phase} n={len(m)}")

    so = m.successes_obvious.to_numpy(float)
    no = m.n_obvious.to_numpy(float)
    sf = m.successes_full73.to_numpy(float)
    nf = m.n_full73.to_numpy(float)
    rng = np.random.default_rng(seed)
    lifts = np.empty(reps, float)
    ro = np.empty(reps, float)
    rf = np.empty(reps, float)
    for pos in range(0, reps, 1000):
        k = min(1000, reps - pos)
        idx = rng.integers(0, len(m), size=(k, len(m)), endpoint=False)
        ro[pos:pos+k] = so[idx].sum(1) / no[idx].sum(1)
        rf[pos:pos+k] = sf[idx].sum(1) / nf[idx].sum(1)
        lifts[pos:pos+k] = rf[pos:pos+k] - ro[pos:pos+k]

    obs_o = _rate(so, no)
    obs_f = _rate(sf, nf)
    return {
        "n_slate_dates": int(len(m)),
        "n_selected_obvious": int(no.sum()),
        "n_selected_full73": int(nf.sum()),
        "obvious_rate": obs_o,
        "full73_rate": obs_f,
        "observed_lift_full73_minus_obvious": obs_f - obs_o,
        "lift_bootstrap": {
            "mean": float(lifts.mean()),
            "median": float(np.median(lifts)),
            "ci95_low": float(np.quantile(lifts, 0.025)),
            "ci95_high": float(np.quantile(lifts, 0.975)),
            "prob_gt_0": float(np.mean(lifts > 0)),
        },
    }


def interaction_bootstrap(
    obvious_daily: pd.DataFrame,
    full_daily: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict:
    """Stratified by year x half, then SECOND lift minus FIRST lift."""
    m = obvious_daily.merge(
        full_daily,
        on=["game_date", "year", "phase"],
        suffixes=("_obvious", "_full73"),
        validate="one_to_one",
    )
    expected = {(2023, "FIRST_HALF"), (2023, "SECOND_HALF"),
                (2024, "FIRST_HALF"), (2024, "SECOND_HALF")}
    got = set((int(y), str(p)) for y, p in m[["year", "phase"]].drop_duplicates().itertuples(index=False, name=None))
    if got != expected:
        raise RuntimeError(f"missing year/phase strata: got={sorted(got)}")

    strata = {}
    for key, g in m.groupby(["year", "phase"], sort=True):
        strata[(int(key[0]), str(key[1]))] = {
            "so": g.successes_obvious.to_numpy(float),
            "no": g.n_obvious.to_numpy(float),
            "sf": g.successes_full73.to_numpy(float),
            "nf": g.n_full73.to_numpy(float),
        }

    rng = np.random.default_rng(seed)
    interaction = np.empty(reps, float)
    first_lift = np.empty(reps, float)
    second_lift = np.empty(reps, float)

    for r in range(reps):
        agg = {
            "FIRST_HALF": [0.0, 0.0, 0.0, 0.0],
            "SECOND_HALF": [0.0, 0.0, 0.0, 0.0],
        }
        for (_, phase), z in strata.items():
            n_dates = len(z["so"])
            idx = rng.integers(0, n_dates, size=n_dates, endpoint=False)
            vals = [z["so"][idx].sum(), z["no"][idx].sum(),
                    z["sf"][idx].sum(), z["nf"][idx].sum()]
            agg[phase] = [a + float(v) for a, v in zip(agg[phase], vals)]
        fl = agg["FIRST_HALF"][2] / agg["FIRST_HALF"][3] - agg["FIRST_HALF"][0] / agg["FIRST_HALF"][1]
        sl = agg["SECOND_HALF"][2] / agg["SECOND_HALF"][3] - agg["SECOND_HALF"][0] / agg["SECOND_HALF"][1]
        first_lift[r] = fl
        second_lift[r] = sl
        interaction[r] = sl - fl

    def observed_phase(phase: str) -> float:
        g = m[m.phase.eq(phase)]
        return (
            g.successes_full73.sum() / g.n_full73.sum()
            - g.successes_obvious.sum() / g.n_obvious.sum()
        )

    obs_first = float(observed_phase("FIRST_HALF"))
    obs_second = float(observed_phase("SECOND_HALF"))
    return {
        "definition": "SECOND_HALF lift minus FIRST_HALF lift",
        "cluster_scheme": "resample slate dates within each year x half stratum",
        "n_replicates": int(reps),
        "observed_first_half_lift": obs_first,
        "observed_second_half_lift": obs_second,
        "observed_interaction": obs_second - obs_first,
        "bootstrap_interaction": {
            "mean": float(interaction.mean()),
            "median": float(np.median(interaction)),
            "ci95_low": float(np.quantile(interaction, 0.025)),
            "ci95_high": float(np.quantile(interaction, 0.975)),
            "prob_gt_0": float(np.mean(interaction > 0)),
        },
    }


def support_feature_names(feature_names: list[str]) -> list[str]:
    """Return only genuine count/support features, never rate-valued *_per_pa_* fields."""
    return [c for c in GENUINE_SUPPORT_FEATURES if c in feature_names]


def support_threshold(col: str) -> float | None:
    return GENUINE_SUPPORT_FEATURES.get(col)


def maturity_audit(
    features: pd.DataFrame,
    feature_names: list[str],
    full73: pd.DataFrame,
) -> dict:
    need = {"game_pk", "batter_id", "game_date", "year", "hr_in_game"}
    missing = need - set(features.columns)
    if missing:
        raise RuntimeError(f"feature matrix missing ID/outcome columns: {sorted(missing)}")
    if not features.year.between(2015, 2024).all() or 2025 in set(features.year.astype(int)):
        raise RuntimeError("feature matrix year scope violated")

    f = features[features.year.between(2023, 2024)].copy()
    f["game_date"] = pd.to_datetime(f.game_date).dt.normalize()
    f["phase"] = phase_from_date(f.game_date)
    if f.duplicated(KEYS).any():
        raise RuntimeError("duplicate batter-game keys in feature matrix")

    ranks = full73[KEYS + ["game_date", "p_raw"]].copy()
    ranks["game_date"] = pd.to_datetime(ranks.game_date).dt.normalize()
    ranks["selected_full73_top5"] = deterministic_top5_flags(ranks)
    ranks = ranks[KEYS + ["selected_full73_top5"]]
    f = f.merge(ranks, on=KEYS, how="left", validate="one_to_one")
    if f.selected_full73_top5.isna().any():
        raise RuntimeError("feature matrix and full73 predictions have mismatched keys")
    f["selected_full73_top5"] = f.selected_full73_top5.astype(bool)

    recent = [c for c in feature_names if ("14d" in c or "30d" in c) and c in f.columns]
    support = support_feature_names(recent)

    out = {
        "recent_feature_count": len(recent),
        "recent_features": recent,
        "support_feature_count": len(support),
        "support_features": support,
        "by_half": {},
    }
    for phase in ["FIRST_HALF", "SECOND_HALF"]:
        g = f[f.phase.eq(phase)]
        gs = g[g.selected_full73_top5]
        phase_out = {
            "n_full_slate_rows": int(len(g)),
            "n_full73_top5_rows": int(len(gs)),
            "base_hr_rate": float(g.hr_in_game.mean()),
            "recent_missingness": {"full_slate": {}, "full73_top5": {}},
            "support": {"full_slate": {}, "full73_top5": {}},
        }
        for c in recent:
            phase_out["recent_missingness"]["full_slate"][c] = float(g[c].isna().mean())
            phase_out["recent_missingness"]["full73_top5"][c] = float(gs[c].isna().mean())
        for scope_name, x in [("full_slate", g), ("full73_top5", gs)]:
            for c in support:
                v = pd.to_numeric(x[c], errors="coerce").dropna().to_numpy(float)
                threshold = support_threshold(c)
                phase_out["support"][scope_name][c] = {
                    "n_nonmissing": int(len(v)),
                    "p10": None if len(v) == 0 else float(np.quantile(v, 0.10)),
                    "p25": None if len(v) == 0 else float(np.quantile(v, 0.25)),
                    "median": None if len(v) == 0 else float(np.median(v)),
                    "zero_fraction": None if len(v) == 0 else float(np.mean(v <= 0)),
                    "low_history_threshold": threshold,
                    "low_history_fraction": None if len(v) == 0 else float(np.mean(v < threshold)),
                }
        out["by_half"][phase] = phase_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obvious", required=True)
    ap.add_argument("--full73", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--feature-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maturity-csv", required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.reps < 1000:
        raise RuntimeError("at least 1000 bootstrap reps required")

    obvious = pd.read_parquet(args.obvious)
    full73 = pd.read_parquet(args.full73)
    obvious, full73 = validate_pair(obvious, full73)
    obvious["phase"] = phase_from_date(obvious.game_date)
    full73["phase"] = phase_from_date(full73.game_date)

    od = daily_selected_counts(obvious)
    fd = daily_selected_counts(full73)
    phases = {
        "FIRST_HALF": phase_bootstrap(od, fd, "FIRST_HALF", args.reps, args.seed),
        "SECOND_HALF": phase_bootstrap(od, fd, "SECOND_HALF", args.reps, args.seed + 1),
    }
    interaction = interaction_bootstrap(od, fd, args.reps, args.seed + 100)

    feature_names = json.loads(Path(args.feature_list).read_text())
    raw_features = pd.read_parquet(args.features)
    maturity = maturity_audit(raw_features, feature_names, full73)

    payload = {
        "design": {
            "primary_selector": "daily raw-score top 5%",
            "first_half": "Opening Day through June 30",
            "second_half": "July 1 through end regular season",
            "calendar_month_outcomes_emitted": False,
            "disagreement_buckets_analyzed": False,
            "development_years": [2023, 2024],
            "bootstrap_replicates": int(args.reps),
            "bootstrap_seed": int(args.seed),
            "sealed_final_holdout": "2025",
            "holdout_2025_read": False,
        },
        "top5_phase_results": phases,
        "half_interaction": interaction,
        "feature_maturity": maturity,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    rows = []
    for phase, z in maturity["by_half"].items():
        for scope, cols in z["recent_missingness"].items():
            for feature, miss in cols.items():
                rows.append({"phase": phase, "scope": scope, "feature": feature,
                             "metric": "nan_rate", "value": miss, "threshold": np.nan})
        for scope, cols in z["support"].items():
            for feature, stats in cols.items():
                for metric in ["p10", "p25", "median", "zero_fraction", "low_history_fraction"]:
                    rows.append({"phase": phase, "scope": scope, "feature": feature,
                                 "metric": metric, "value": stats[metric],
                                 "threshold": stats["low_history_threshold"]})
    pd.DataFrame(rows).to_csv(args.maturity_csv, index=False)

    print(json.dumps(payload, indent=2), flush=True)
    print("[top5-half-seasonality] 2025 NOT READ; no calendar-month outcomes emitted", flush=True)


if __name__ == "__main__":
    main()
