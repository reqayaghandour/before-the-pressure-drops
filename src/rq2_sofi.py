"""
RQ2 (supplementary): SOFI - Sparseness-Optimized Feature Importance (Grau & Napoles, 2024).

Usage (from the src folder, AFTER rq2_feature_importance.py):
    pip install sofi-tabular
    python rq2_sofi.py primary
    python rq2_sofi.py extended

What does SOFI do?
    SOFI directly searches for the feature ranking in which removing ("marginalising")
    the most important features degrades the model's response as quickly as possible
    (MoRF), while removing the least important features changes it as little as possible
    (LeRF). The search is hill climbing with swaps of two positions; the objective is to
    maximise the area between the LeRF and MoRF curves (degradation score, DS).

Design (same 10 x 5 outer folds as run_model.py and rq2_feature_importance.py):
    For each outer fold:
      1. Train LR (C = 0.1) on the training fold.
      2. Compute the SHAP ranking on the training fold (as in rq2_feature_importance.py).
      3. Search the SOFI ranking on the CORRECTLY classified training cases
         (select_reliable_instances), starting from the greedy ranking.
         -> the test fold is NOT used for the search (otherwise SOFI would be optimised
            on the same data it is evaluated on).
      4. Evaluate both rankings on the test fold with
         (a) the same ROC-AUC MORF/LERF procedure as in rq2_feature_importance.py
             (replacement by the training median / most frequent value) -> ABC
         (b) SOFI's own protocol (score_ranking) -> degradation score (DS)
    Marginalisation in SOFI: "constant" with the median (numerical) and mode (categorical),
    i.e. the same replacement protocol as the existing MORF/LERF analysis.

    In addition, a SOFI ranking is computed for the final model (all data), next to the
    SHAP and permutation rankings.

Output in results/rq2/:
    sofi_folds_<fs>.csv        per fold: ABC and DS for SHAP and SOFI, Spearman rho, top 3
    sofi_ranking_<fs>.csv      final model: SOFI rank next to SHAP and permutation rank
    sofi_curves_<fs>.csv       mean ROC-AUC MORF/LERF curves for SHAP and SOFI
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing import FEATURE_SETS, get_X_y_groups, load_data
from rq2_feature_importance import (BASE_SEED, N_OUTER, N_REPEATS, baseline_values, make_lr,
                                    removal_curve, shap_per_feature)

try:
    from sofi import SOFIExplainer, select_reliable_instances
except ImportError:
    sys.exit("Package missing. Install it with:  pip install sofi-tabular")

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "rq2"

SOFI_SETTINGS = dict(
    marginalization="constant",      # same protocol as the existing MORF/LERF analysis
    numeric_strategy="median",
    integer_strategy="median",
    categorical_strategy="mode",
    initialization="greedy",
    max_iterations=200,              # package default
    verbose=False,
)


def area(curve):
    """Area under a removal curve, with the proportion of removed features on the x-axis."""
    frac = np.linspace(0, 1, len(curve))
    return np.trapezoid(curve, frac)


def sofi_ranking(model, X_tr, y_tr, seed):
    """SOFI ranking, searched on the correctly classified training cases."""
    reliable = select_reliable_instances(model, X_tr, y_tr)
    explainer = SOFIExplainer(model, X_tr, random_state=seed, **SOFI_SETTINGS)
    expl = explainer.explain(reliable, mode="global")
    return explainer, expl


def main():
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "primary"
    if feature_set not in FEATURE_SETS:
        sys.exit(f"Choose a feature set from {FEATURE_SETS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, groups = get_X_y_groups(df, feature_set=feature_set)
    features = list(X.columns)
    n_folds = N_REPEATS * N_OUTER

    print(f"\nRQ2 - SOFI, LOGISTIC REGRESSION ({feature_set.upper()})")
    print("=" * 70)

    rows = []
    curves = {"SHAP_MORF": [], "SHAP_LERF": [], "SOFI_MORF": [], "SOFI_LERF": []}
    t0 = time.time()
    for repeat in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=BASE_SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups=groups), start=1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = make_lr(feature_set).fit(X_tr, y_tr)

            # SHAP ranking (same as rq2_feature_importance.py)
            shap_order = list(shap_per_feature(model, X_tr, X_tr).abs().mean()
                              .sort_values(ascending=False).index)

            # SOFI ranking (training data only)
            explainer, expl = sofi_ranking(model, X_tr, y_tr, seed=BASE_SEED + repeat * 10 + fold)
            sofi_order = list(expl.ranking)

            # (a) ROC-AUC MORF/LERF on the test fold
            base = baseline_values(X_tr)
            res = {}
            for name, order in [("SHAP", shap_order), ("SOFI", sofi_order)]:
                morf = removal_curve(model, X_te, y_te, order, base)
                lerf = removal_curve(model, X_te, y_te, order[::-1], base)
                curves[f"{name}_MORF"].append(morf)
                curves[f"{name}_LERF"].append(lerf)
                res[f"abc_{name.lower()}"] = area(lerf) - area(morf)
                res[f"auc_after3_{name.lower()}"] = morf[3]

            # (b) SOFI protocol on the test fold (correctly classified test cases)
            reliable_te = select_reliable_instances(model, X_te, y_te)
            res["ds_shap"] = explainer.score_ranking(shap_order, reliable_te).ds
            res["ds_sofi"] = explainer.score_ranking(sofi_order, reliable_te).ds

            shap_rank = pd.Series(range(len(shap_order)), index=shap_order).reindex(features)
            sofi_rank = pd.Series(range(len(sofi_order)), index=sofi_order).reindex(features)
            res["spearman_shap_sofi"] = spearmanr(shap_rank, sofi_rank)[0]
            res["top3_shap"] = " | ".join(shap_order[:3])
            res["top3_sofi"] = " | ".join(sofi_order[:3])
            rows.append({"repeat": repeat, "fold": fold, **res})
        print(f"  repetition {repeat + 1}/{N_REPEATS} done  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    folds = pd.DataFrame(rows)

    # ---------- Final model (all data) ----------
    final = make_lr(feature_set).fit(X, y)
    _, final_expl = sofi_ranking(final, X, y, seed=BASE_SEED)
    imp_path = OUT_DIR / f"importance_{feature_set}.csv"
    ranking = pd.DataFrame({"rank_sofi": pd.Series(range(1, len(features) + 1),
                                                   index=list(final_expl.ranking))})
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)
        ranking = ranking.join(imp[["rank_shap", "rank_perm"]])
    else:
        print(f"Note: {imp_path} not found; run rq2_feature_importance.py first "
              f"to compare with the SHAP and permutation rankings.")
    ranking = ranking.sort_values("rank_sofi")

    # ---------- Summary ----------
    print(f"\nPer fold (mean +- SD over {n_folds} test folds)")
    for col, label in [("abc_shap", "ABC (ROC-AUC), SHAP ranking"),
                       ("abc_sofi", "ABC (ROC-AUC), SOFI ranking"),
                       ("auc_after3_shap", "ROC-AUC after removing 3 features, SHAP"),
                       ("auc_after3_sofi", "ROC-AUC after removing 3 features, SOFI"),
                       ("ds_shap", "Degradation score (SOFI protocol), SHAP"),
                       ("ds_sofi", "Degradation score (SOFI protocol), SOFI"),
                       ("spearman_shap_sofi", "Spearman rho SHAP vs SOFI")]:
        print(f"  {label:<45} {folds[col].mean():.3f} +- {folds[col].std(ddof=1):.3f}")
    diff = folds["abc_sofi"] - folds["abc_shap"]
    print(f"\n  ABC SOFI - SHAP: {diff.mean():+.3f}; SOFI higher in {(diff > 0).sum()}/{len(diff)} folds "
          f"(descriptive; folds are not independent)")
    print("\n  Most frequent top 3 (SOFI):")
    print(folds["top3_sofi"].value_counts().head(3).to_string())

    print("\nFinal model: SOFI ranking (top 10) next to SHAP and permutation")
    print(ranking.head(10).to_string())
    if "rank_shap" in ranking:
        rho_final = spearmanr(ranking["rank_sofi"], ranking["rank_shap"])[0]
        print(f"  Spearman rho SOFI vs SHAP (final model): {rho_final:.2f}")

    # ---------- Save ----------
    folds.round(5).to_csv(OUT_DIR / f"sofi_folds_{feature_set}.csv", index=False)
    ranking.to_csv(OUT_DIR / f"sofi_ranking_{feature_set}.csv")
    k = np.arange(len(features) + 1)
    curve_df = pd.DataFrame({"n_removed": k})
    for name, c in curves.items():
        arr = np.vstack(c)
        curve_df[f"{name}_mean"] = arr.mean(axis=0)
        curve_df[f"{name}_sd"] = arr.std(axis=0, ddof=1)
    curve_df.round(5).to_csv(OUT_DIR / f"sofi_curves_{feature_set}.csv", index=False)
    print(f"\nSaved in: {OUT_DIR}  (total time {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
