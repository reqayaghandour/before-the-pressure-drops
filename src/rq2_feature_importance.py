"""
RQ2: which clinical features most strongly predict IH in the selected model (LR)?

Usage (from the src folder, after run_model.py lr primary/extended):
    python rq2_feature_importance.py primary
    python rq2_feature_importance.py extended

Three parts:

1. FEATURE PERMUTATION IMPORTANCE (model-agnostic)
   Already computed by run_model.py on every outer test fold; summarised here
   (mean ± SD over 50 folds).

2. SHAP (Lundberg & Lee, 2017) for the final model
   LR refitted on all data with C = 0.1 (the value selected in most folds).
   For a linear model, SHAP values can be computed exactly:
       phi_ij = coef_j * (x_ij - mean_j)        (log-odds, assuming independent features)
   This is what shap.LinearExplainer computes; here without an extra package.
   One-hot columns and missing indicators are summed per original feature.
   In addition: odds ratios per (standardised) model column.

3. MORF / LERF validation of the SHAP ranking (Samek et al., 2017)
   For each outer fold (the same 10 x 5 folds as run_model.py):
     - fit LR (C = 0.1) on the training fold and rank the features by SHAP on the training fold
     - on the test fold, progressively "remove" features (replace them by the training median
       or most frequent value) and measure the ROC-AUC after each step
       MORF: most relevant first   -> ROC-AUC should drop quickly
       LERF: least relevant first  -> ROC-AUC should stay high
       RANDOM: random order (mean of five permutations), as a reference
   The larger the area between LERF and MORF, the more faithful the ranking.

Output in results/rq2/:
    importance_<fs>.csv     feature permutation importance and SHAP per feature, with ranks
    odds_ratios_<fs>.csv    odds ratio per model column (final model)
    morf_lerf_<fs>.csv      mean ROC-AUC curves (MORF, LERF, random)
    shap_<fs>.png, morf_lerf_<fs>.png   quick-look figures
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from preprocessing import FEATURE_SETS, build_preprocessor, get_X_y_groups, load_data

warnings.filterwarnings("ignore", category=UserWarning)

BASE_SEED = 42
N_OUTER = 5
N_REPEATS = 10
FINAL_C = 0.1
N_RANDOM_ORDERS = 5
TOP_K_PLOT = 15

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "rq2"


def make_lr(feature_set):
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_set=feature_set)),
        ("model", LogisticRegression(C=FINAL_C, max_iter=1000, random_state=BASE_SEED)),
    ])


def column_to_feature(out_names, original_cols):
    """Map each model column (after encoding) to its original feature."""
    mapping = []
    for name in out_names:
        if name in original_cols:
            mapping.append(name)
        elif name.startswith("missingindicator_"):
            mapping.append(name.replace("missingindicator_", "", 1))
        else:
            matches = [c for c in original_cols if name.startswith(c + "_")]
            if not matches:
                raise ValueError(f"Cannot map column {name!r} to a feature.")
            mapping.append(max(matches, key=len))
    return np.array(mapping)


def shap_per_feature(fitted, X, X_background):
    """Exact linear SHAP values, summed per original feature (log-odds)."""
    prep = fitted.named_steps["preprocessor"]
    coef = fitted.named_steps["model"].coef_.ravel()
    Z = prep.transform(X)
    Z_bg = prep.transform(X_background)
    phi = (Z - Z_bg.mean(axis=0)) * coef
    feats = column_to_feature(prep.get_feature_names_out(), list(X.columns))
    phi_df = pd.DataFrame(phi, columns=feats)
    return phi_df.T.groupby(level=0).sum().T   # rows = cases, columns = features


def baseline_values(X_train):
    base = {}
    for c in X_train.columns:
        s = X_train[c]
        if pd.api.types.is_numeric_dtype(s):
            base[c] = s.median()
        else:
            mode = s.dropna().mode()
            base[c] = mode.iloc[0] if len(mode) else np.nan
    return base


def removal_curve(model, X_test, y_test, order, base):
    X_mod = X_test.copy()
    aucs = [roc_auc_score(y_test, model.predict_proba(X_mod)[:, 1])]
    for feat in order:
        X_mod[feat] = base[feat]
        proba = model.predict_proba(X_mod)[:, 1]
        aucs.append(roc_auc_score(y_test, proba) if np.unique(proba).size > 1 else 0.5)
    return np.array(aucs)


def main():
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "primary"
    if feature_set not in FEATURE_SETS:
        sys.exit(f"Choose a feature set from {FEATURE_SETS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, groups = get_X_y_groups(df, feature_set=feature_set)
    features = list(X.columns)
    p = len(features)

    print(f"\nRQ2 — FEATURE IMPORTANCE, LOGISTIC REGRESSION ({feature_set.upper()})")
    print("=" * 70)

    # ---------- 1. Feature permutation importance from run_model.py ----------
    perm_path = RESULTS_DIR / f"lr_{feature_set}" / "importances.csv"
    if not perm_path.exists():
        sys.exit(f"{perm_path} is missing. First run: python run_model.py lr {feature_set}")
    perm = (pd.read_csv(perm_path).groupby("feature")["importance"]
            .agg(perm_mean="mean", perm_sd="std"))

    # ---------- 2. SHAP for the final model ----------
    final = make_lr(feature_set).fit(X, y)
    phi = shap_per_feature(final, X, X)
    shap_imp = phi.abs().mean().rename("shap_mean_abs")
    # direction: correlation between feature value and SHAP value (numerical features only)
    direction = {}
    for f in features:
        if pd.api.types.is_numeric_dtype(X[f]):
            ok = X[f].notna()
            r = np.corrcoef(X.loc[ok, f], phi.loc[ok, f])[0, 1] if ok.sum() > 2 else np.nan
            direction[f] = "higher -> more IH" if r > 0 else "higher -> less IH"
        else:
            direction[f] = "categorical (see odds ratios)"

    imp = pd.concat([shap_imp, perm], axis=1).reindex(features)
    imp["direction"] = pd.Series(direction)
    imp["rank_shap"] = imp["shap_mean_abs"].rank(ascending=False).astype(int)
    imp["rank_perm"] = imp["perm_mean"].rank(ascending=False).astype(int)
    imp = imp.sort_values("shap_mean_abs", ascending=False)
    rho, rho_p = spearmanr(imp["shap_mean_abs"], imp["perm_mean"])

    prep = final.named_steps["preprocessor"]
    coef = final.named_steps["model"].coef_.ravel()
    odds = pd.DataFrame({
        "column": prep.get_feature_names_out(),
        "feature": column_to_feature(prep.get_feature_names_out(), features),
        "coef_log_odds": coef,
        "odds_ratio": np.exp(coef),
    }).sort_values("coef_log_odds", key=np.abs, ascending=False)

    print("\nTop 15 features (SHAP for the final model; permutation = mean over 50 test folds)")
    print(f"  {'feature':<22}{'mean|SHAP|':>11}{'perm. imp.':>12}{'rank SHAP':>11}{'rank perm':>11}  direction")
    for f, r in imp.head(15).iterrows():
        print(f"  {f:<22}{r['shap_mean_abs']:>11.3f}{r['perm_mean']:>12.4f}{r['rank_shap']:>11}"
              f"{r['rank_perm']:>11}  {r['direction']}")
    print(f"\n  Agreement of the rankings, SHAP vs permutation: Spearman rho = {rho:.2f} (p = {rho_p:.3g})")

    # ---------- 3. MORF / LERF ----------
    rng = np.random.default_rng(BASE_SEED)
    curves = {"MORF": [], "LERF": [], "RANDOM": []}
    for repeat in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=BASE_SEED + repeat)
        for train_idx, test_idx in outer.split(X, y, groups=groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = make_lr(feature_set).fit(X_tr, y_tr)
            ranking = shap_per_feature(model, X_tr, X_tr).abs().mean().sort_values(ascending=False)
            order = list(ranking.index)
            base = baseline_values(X_tr)
            curves["MORF"].append(removal_curve(model, X_te, y_te, order, base))
            curves["LERF"].append(removal_curve(model, X_te, y_te, order[::-1], base))
            rand = np.mean([removal_curve(model, X_te, y_te, list(rng.permutation(order)), base)
                            for _ in range(N_RANDOM_ORDERS)], axis=0)
            curves["RANDOM"].append(rand)
        print(f"  MORF/LERF: repetition {repeat + 1}/{N_REPEATS} done", flush=True)

    k = np.arange(p + 1)
    curve_df = pd.DataFrame({"n_removed": k})
    for name, c in curves.items():
        arr = np.vstack(c)
        curve_df[f"{name}_mean"] = arr.mean(axis=0)
        curve_df[f"{name}_sd"] = arr.std(axis=0, ddof=1)
    frac = k / p
    area = {name: np.trapezoid(curve_df[f"{name}_mean"], frac) for name in curves}
    abc = area["LERF"] - area["MORF"]

    print("\nMORF/LERF validation (mean over 50 test folds)")
    for n_removed in [0, 1, 2, 3, 5]:
        row = curve_df.iloc[n_removed]
        print(f"  {n_removed} feature(s) removed:  MORF {row['MORF_mean']:.3f} | "
              f"LERF {row['LERF_mean']:.3f} | random {row['RANDOM_mean']:.3f}")
    print(f"  Area under the curve: MORF {area['MORF']:.3f} | LERF {area['LERF']:.3f} | "
          f"random {area['RANDOM']:.3f}")
    print(f"  Area between LERF and MORF (ABC): {abc:.3f}  (larger = more faithful ranking)")

    # ---------- Save ----------
    imp.round(5).to_csv(OUT_DIR / f"importance_{feature_set}.csv")
    odds.round(5).to_csv(OUT_DIR / f"odds_ratios_{feature_set}.csv", index=False)
    curve_df.round(5).to_csv(OUT_DIR / f"morf_lerf_{feature_set}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = imp.head(TOP_K_PLOT).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(top.index, top["shap_mean_abs"], color="#4C72B0")
        ax.set_xlabel("Mean |SHAP value| (log-odds)")
        ax.set_title(f"Feature importance, logistic regression ({feature_set})")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"shap_{feature_set}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        styles = {"MORF": ("#C44E52", "Most relevant first"),
                  "LERF": ("#4C72B0", "Least relevant first"),
                  "RANDOM": ("#8C8C8C", "Random order")}
        for name, (color, label) in styles.items():
            m, s = curve_df[f"{name}_mean"], curve_df[f"{name}_sd"]
            ax.plot(k, m, color=color, label=label, linestyle="--" if name == "RANDOM" else "-")
            ax.fill_between(k, m - s, m + s, color=color, alpha=0.12)
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Number of features removed")
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(f"MORF / LERF validation of SHAP ranking ({feature_set})")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"morf_lerf_{feature_set}.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib not found; figures skipped.")

    print(f"\nSaved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
