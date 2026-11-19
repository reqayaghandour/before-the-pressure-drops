"""
Statistical comparison of the models (RQ1). Nothing is trained: the script only reads
results/<model>_<feature_set>/fold_scores.csv, created by run_model.py.

Usage (from the src folder):
    python compare_models.py            # on ROC-AUC (primary metric)
    python compare_models.py pr_auc     # optionally on another metric

Method:
    Corrected paired t-test for repeated k-fold cross-validation (Nadeau & Bengio, 2003;
    Bouckaert & Frank, 2004). All models use the same outer folds, so the difference between
    two models can be computed per fold (paired). Because the training sets of the folds
    overlap, the variance is corrected:
        var_corr = (1/n + n_test/n_train) * s^2      (n = number of folds = k * r)
        t = mean difference / sqrt(var_corr),        df = n - 1
    Effect size: mean difference with a corrected 95% confidence interval.
    Bonferroni correction within each family of tests:
        A. models against each other, per feature set (6 pairs per feature set)
        B. extended vs. primary, per model (4 tests)

Output:
    results/summary_table.csv          all metrics per model and feature set (mean ± SD)
    results/comparisons_<metric>.csv   all tests with difference, 95% CI, t, p, p (Bonferroni)
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "nn": "Neural Network"}
FEATURE_SETS = ["primary", "extended"]
METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall", "train_roc_auc"]
ALPHA = 0.05


def load_scores():
    scores = {}
    missing = []
    for m in MODELS:
        for fs in FEATURE_SETS:
            path = RESULTS_DIR / f"{m}_{fs}" / "fold_scores.csv"
            if not path.exists():
                missing.append(f"{m}_{fs}")
                continue
            df = pd.read_csv(path).sort_values(["repeat", "fold"]).set_index(["repeat", "fold"])
            scores[(m, fs)] = df
    if missing:
        print(f"Note: no results found for: {', '.join(missing)} (these are skipped)\n")
    if not scores:
        sys.exit(f"No results found in {RESULTS_DIR}.")
    return scores


def check_same_folds(scores):
    """A paired comparison requires identical folds: the same (repeat, fold) and n_test."""
    ref_key = next(iter(scores))
    ref = scores[ref_key]
    for key, df in scores.items():
        if not df.index.equals(ref.index) or not (df["n_test"] == ref["n_test"]).all():
            sys.exit(
                f"The folds of {key[0]}_{key[1]} do not match those of {ref_key[0]}_{ref_key[1]}.\n"
                "A paired comparison is then not valid. Check that all models were run with the same "
                "version of run_model.py and the same number of repetitions."
            )
    return ref


def corrected_ttest(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    var_corr = (1 / n + n_test / n_train) * var_d
    se = np.sqrt(var_corr)
    t_stat = mean_d / se if se > 0 else np.nan
    df = n - 1
    p = 2 * stats.t.sf(abs(t_stat), df) if se > 0 else np.nan
    half = stats.t.ppf(1 - ALPHA / 2, df) * se
    n_wins = int((d > 0).sum())
    return mean_d, mean_d - half, mean_d + half, t_stat, p, n_wins, n


def summary_table(scores):
    rows = []
    for (m, fs), df in scores.items():
        row = {"model": MODELS[m], "feature_set": fs, "n_folds": len(df)}
        for metric in METRICS:
            if metric in df:
                row[metric] = f"{df[metric].mean():.3f} ± {df[metric].std(ddof=1):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    metric = sys.argv[1] if len(sys.argv) > 1 else "roc_auc"
    if metric not in METRICS[:-1]:
        sys.exit(f"Unknown metric {metric!r}. Choose from: {', '.join(METRICS[:-1])}")

    scores = load_scores()
    ref = check_same_folds(scores)
    n_train = ref["n_train"].mean()
    n_test = ref["n_test"].mean()
    n_folds = len(ref)

    print("=" * 78)
    print(f"MODEL COMPARISON on {metric.upper()} - {n_folds} paired folds "
          f"(n_train ≈ {n_train:.0f}, n_test ≈ {n_test:.0f})")
    print("Corrected repeated-CV t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004)")
    print("=" * 78)

    summary = summary_table(scores)
    print("\nOVERVIEW (mean ± SD over folds)")
    print(summary.to_string(index=False))

    results = []

    # A. Models against each other, per feature set
    for fs in FEATURE_SETS:
        models_here = [m for m in MODELS if (m, fs) in scores]
        pairs = list(combinations(models_here, 2))
        for m1, m2 in pairs:
            diff, lo, hi, t_stat, p, wins, n = corrected_ttest(
                scores[(m1, fs)][metric], scores[(m2, fs)][metric], n_train, n_test
            )
            results.append({
                "family": f"A: models ({fs})", "comparison": f"{MODELS[m1]} vs {MODELS[m2]}",
                "feature_set": fs, "mean_1": scores[(m1, fs)][metric].mean(),
                "mean_2": scores[(m2, fs)][metric].mean(), "diff": diff, "ci_low": lo, "ci_high": hi,
                "t": t_stat, "p": p, "p_bonferroni": min(1.0, p * len(pairs)),
                "folds_1_better": f"{wins}/{n}",
            })

    # B. Extended vs. primary, per model
    models_both = [m for m in MODELS if (m, "primary") in scores and (m, "extended") in scores]
    for m in models_both:
        diff, lo, hi, t_stat, p, wins, n = corrected_ttest(
            scores[(m, "extended")][metric], scores[(m, "primary")][metric], n_train, n_test
        )
        results.append({
            "family": "B: extended vs primary", "comparison": f"{MODELS[m]}: extended vs primary",
            "feature_set": "both", "mean_1": scores[(m, "extended")][metric].mean(),
            "mean_2": scores[(m, "primary")][metric].mean(), "diff": diff, "ci_low": lo, "ci_high": hi,
            "t": t_stat, "p": p, "p_bonferroni": min(1.0, p * len(models_both)),
            "folds_1_better": f"{wins}/{n}",
        })

    res = pd.DataFrame(results)
    res["significant"] = np.where(res["p_bonferroni"] < ALPHA, "yes", "no")

    for family, block in res.groupby("family", sort=False):
        print(f"\n{family.upper()}")
        print(f"  {'comparison':<44}{'diff.':>9}  {'95% CI':<18}{'t':>7}{'p (Bonf.)':>11}  sign.  better in")
        for _, r in block.iterrows():
            ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            print(f"  {r['comparison']:<44}{r['diff']:>+9.3f}  {ci:<18}{r['t']:>7.2f}"
                  f"{r['p_bonferroni']:>11.4f}  {r['significant']:<5}  {r['folds_1_better']}")

    print("\nHow to read:")
    print("  diff.     = mean of (first - second) over all folds; positive = first is better")
    print("  95% CI    = corrected confidence interval of that difference (effect size)")
    print("  p (Bonf.) = p value after Bonferroni correction within the family; significant if < 0.05")
    print("  better in = in how many folds the first model scored higher")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    res.round(4).to_csv(RESULTS_DIR / f"comparisons_{metric}.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'summary_table.csv'}")
    print(f"       {RESULTS_DIR / f'comparisons_{metric}.csv'}")


if __name__ == "__main__":
    main()
