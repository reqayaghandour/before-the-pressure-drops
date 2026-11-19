"""
Repeated nested cross-validation for all models (RQ1; also the basis for RQ2 and RQ3).

Usage (from the src folder):
    python run_model.py lr primary
    python run_model.py rf extended
    python run_model.py lr primary 5        # optional: number of repetitions (default 10)
Models: lr, rf, xgb, nn. Feature sets: primary, extended.

Design:
    - N_REPEATS x (5 outer x 5 inner) StratifiedGroupKFold, grouped by patient (subjectid).
    - Repetition r uses outer seed 42 + r, so all models and feature sets get identical
      outer folds (needed for the paired comparison between models).
    - Hyperparameters are tuned in the inner loop on ROC-AUC (GridSearchCV).
    - Preprocessing from preprocessing.py, fitted on the training data of each fold only.

Saved per outer fold in results/<model>_<feature_set>/:
    fold_scores.csv   ROC-AUC, PR-AUC, F1, precision, recall, training ROC-AUC,
                      selected hyperparameters, computation time
    predictions.csv   per case: true outcome and predicted probability (for RQ1 and RQ3)
    importances.csv   feature permutation importance on the test fold, per original feature (RQ2)

If the script crashes or is stopped, simply restart it: completed folds are skipped.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from preprocessing import FEATURE_SETS, build_preprocessor, get_X_y_groups, load_data

warnings.filterwarnings("ignore", category=UserWarning)  # e.g. unknown categories in a test fold
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# Settings
# =========================================================

BASE_SEED = 42
N_OUTER = 5
N_INNER = 5
DEFAULT_REPEATS = 10
THRESHOLD = 0.5            # for F1, precision and recall
PERM_REPEATS = 5           # repetitions per feature for permutation importance

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# =========================================================
# Models and hyperparameter grids
# (each grid contains the software default; see Table 2 of the thesis)
# =========================================================

MODELS = {
    "lr": {
        "name": "Logistic Regression",
        "estimator": lambda: LogisticRegression(max_iter=1000, random_state=BASE_SEED),
        "grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
    },
    "rf": {
        "name": "Random Forest",
        "estimator": lambda: RandomForestClassifier(
            n_estimators=300, random_state=BASE_SEED, n_jobs=1
        ),
        "grid": {
            # 5 added: in an initial run, the selected value (10) was at the boundary of the grid
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__max_features": ["sqrt", 0.3],
        },
    },
    "xgb": {
        "name": "XGBoost",
        "estimator": lambda: _xgb_classifier(),
        # XGBoost defaults: n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1
        "grid": {
            # 500 and 1000 added: in an initial run, 300 (then the largest value) was
            # selected in most folds
            "model__n_estimators": [100, 300, 500, 1000],
            # 2 and 0.01 added: in an initial run, max_depth=3 (49/50) and
            # learning_rate=0.05 (50/50) were at the boundary of the grid
            "model__max_depth": [2, 3, 6],
            "model__learning_rate": [0.01, 0.05, 0.3],
            "model__subsample": [0.8, 1.0],
        },
    },
    "nn": {
        "name": "Neural Network (MLP)",
        "estimator": lambda: MLPClassifier(
            max_iter=1000,
            early_stopping=True,      # stops when the validation score no longer improves
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=BASE_SEED,
        ),
        # defaults: hidden_layer_sizes=(100,), alpha=0.0001
        "grid": {
            "model__hidden_layer_sizes": [(32,), (100,), (64, 32)],
            "model__alpha": [0.0001, 0.001, 0.01],
        },
    },
}


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        sys.exit("XGBoost is not installed. Install it with:  python -m pip install xgboost")
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=BASE_SEED,
        n_jobs=1,  # parallelisation already happens in GridSearchCV
    )


# =========================================================
# Arguments
# =========================================================

def parse_args():
    if len(sys.argv) < 3:
        sys.exit(
            "Usage: python run_model.py <model> <feature_set> [repetitions]\n"
            f"  model:       {', '.join(MODELS)}\n"
            f"  feature_set: {', '.join(FEATURE_SETS)}"
        )
    model_key, feature_set = sys.argv[1], sys.argv[2]
    n_repeats = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_REPEATS
    if model_key not in MODELS:
        sys.exit(f"Unknown model {model_key!r}. Choose from: {', '.join(MODELS)}")
    if feature_set not in FEATURE_SETS:
        sys.exit(f"Unknown feature set {feature_set!r}. Choose from: {', '.join(FEATURE_SETS)}")
    return model_key, feature_set, n_repeats


# =========================================================
# Saving / resuming
# =========================================================

def append_csv(df, path):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_folds(scores_path):
    if not scores_path.exists():
        return set()
    done = pd.read_csv(scores_path)
    return set(zip(done["repeat"], done["fold"]))


def drop_partial_rows(path, done):
    """Remove rows of a fold that are in predictions/importances but not in fold_scores
    (this can happen if the script stopped exactly while saving)."""
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = [(r, f) in done for r, f in zip(df["repeat"], df["fold"])]
    df[keep].to_csv(path, index=False)


# =========================================================
# Summary
# =========================================================

METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall"]


def print_summary(scores, model_name, feature_set, prevalence):
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} — {feature_set.upper()}")
    print(f"{scores['repeat'].nunique()} repetitions x {N_OUTER} folds = {len(scores)} estimates")
    print("=" * 60)
    for m in METRICS:
        print(f"  {m:<10} {scores[m].mean():.3f} ± {scores[m].std(ddof=1):.3f}")
    print(f"  (PR-AUC of a random model = prevalence = {prevalence:.3f})")

    if "train_roc_auc" in scores:
        gap = scores["train_roc_auc"] - scores["roc_auc"]
        print("\n  Training vs. test (generalisation):")
        print(f"    ROC-AUC training {scores['train_roc_auc'].mean():.3f} ± {scores['train_roc_auc'].std(ddof=1):.3f}"
              f" | test {scores['roc_auc'].mean():.3f} | gap {gap.mean():.3f} ± {gap.std(ddof=1):.3f}")

    # Stability: does the mean still change with more repetitions?
    n_rep = scores["repeat"].nunique()
    if n_rep > 1:
        print("\n  Mean ROC-AUC after k repetitions (stability of the estimate):")
        for k in sorted({1, 3, 5, n_rep} & set(range(1, n_rep + 1))):
            sub = scores[scores["repeat"] < k]
            print(f"    k = {k:>2}: {sub['roc_auc'].mean():.4f}")

    # Stability of the hyperparameters (how often each setting was selected)
    print("\n  Selected hyperparameters (how often, over all folds):")
    print(scores["best_params"].value_counts().to_string())

    imp_path = scores.attrs.get("imp_path")
    if imp_path and Path(imp_path).exists():
        imp = pd.read_csv(imp_path)
        agg = (
            imp.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
        print("\n  Top 10 permutation importance (decrease in ROC-AUC, mean ± SD over folds):")
        for feat, row in agg.head(10).iterrows():
            print(f"    {feat:<20} {row['mean']:.4f} ± {row['std']:.4f}")


# =========================================================
# Main program
# =========================================================

def main():
    model_key, feature_set, n_repeats = parse_args()
    spec = MODELS[model_key]

    out_dir = RESULTS_DIR / f"{model_key}_{feature_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "fold_scores.csv"
    preds_path = out_dir / "predictions.csv"
    imp_path = out_dir / "importances.csv"

    df = load_data()
    X, y, groups = get_X_y_groups(df, feature_set=feature_set)
    caseids = df["caseid"].to_numpy()

    # Protects against mixing runs on different datasets or grids (e.g. after a change
    # in the inclusion criteria): existing results must match the current setup.
    if scores_path.exists():
        old = pd.read_csv(scores_path)
        if not old.empty and "train_roc_auc" not in old.columns:
            sys.exit(
                f"The existing results in {out_dir} were created with an older version of this "
                "script (without training performance).\nMove or rename the results folder first, "
                "then the script starts again."
            )
        grid_now = str(sorted(spec["grid"].items(), key=lambda kv: kv[0]))
        if not old.empty and ("grid" not in old.columns or old["grid"].iloc[0] != grid_now):
            sys.exit(
                f"The existing results in {out_dir} were created with a different "
                "hyperparameter grid.\nMove or rename that folder first, then the "
                "script starts again."
            )
        if not old.empty and int((old["n_train"] + old["n_test"]).iloc[0]) != len(X):
            sys.exit(
                f"The existing results in {out_dir} were created on "
                f"{int((old['n_train'] + old['n_test']).iloc[0])} cases, the current data have "
                f"{len(X)} cases.\nMove or rename the results folder first, then the "
                "script starts again."
            )

    done = completed_folds(scores_path)
    drop_partial_rows(preds_path, done)
    drop_partial_rows(imp_path, done)

    total = n_repeats * N_OUTER
    todo = total - len({(r, f) for r, f in done if r < n_repeats})

    print(f"\n{spec['name']} — feature set: {feature_set}")
    print(f"{n_repeats} repetitions x {N_OUTER} outer x {N_INNER} inner folds")
    print(f"Grid: {len(list(_grid_iter(spec['grid'])))} combinations")
    print(f"Already done: {total - todo} of {total} folds | output: {out_dir}")
    print("-" * 60)

    run_start = time.time()
    n_fitted = 0

    for repeat in range(n_repeats):
        outer_cv = StratifiedGroupKFold(
            n_splits=N_OUTER, shuffle=True, random_state=BASE_SEED + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups=groups), start=1
        ):
            if (repeat, fold) in done:
                continue

            t0 = time.time()
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

            overlap = len(set(g_train) & set(g_test))
            if overlap:
                sys.exit(f"Patient overlap in repetition {repeat}, fold {fold}: {overlap}. Stopped.")

            inner_cv = StratifiedGroupKFold(
                n_splits=N_INNER, shuffle=True, random_state=1000 * repeat + fold
            )
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(feature_set=feature_set)),
                    ("model", spec["estimator"]()),
                ]
            )
            search = GridSearchCV(
                pipeline, spec["grid"], scoring="roc_auc", cv=inner_cv, n_jobs=-1, refit=True
            )
            search.fit(X_train, y_train, groups=g_train)
            best = search.best_estimator_

            proba = best.predict_proba(X_test)[:, 1]
            pred = (proba >= THRESHOLD).astype(int)
            # Performance on the training data of this fold (generalisation vs. training)
            proba_train = best.predict_proba(X_train)[:, 1]

            best_params = ", ".join(
                f"{k.replace('model__', '')}={v}" for k, v in sorted(search.best_params_.items())
            )

            # Feature permutation importance on the test fold, on the original columns
            # (so the one-hot columns of a categorical feature are permuted together).
            perm = permutation_importance(
                best, X_test, y_test, scoring="roc_auc",
                n_repeats=PERM_REPEATS, random_state=BASE_SEED, n_jobs=-1,
            )
            importances = pd.DataFrame({
                "repeat": repeat, "fold": fold,
                "feature": X.columns, "importance": perm.importances_mean,
            })

            predictions = pd.DataFrame({
                "repeat": repeat, "fold": fold,
                "caseid": caseids[test_idx], "subjectid": g_test.to_numpy(),
                "y_true": y_test.to_numpy(), "y_proba": proba,
            })

            seconds = time.time() - t0
            fold_scores = pd.DataFrame([{
                "repeat": repeat, "fold": fold,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
                "f1": f1_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "train_roc_auc": roc_auc_score(y_train, proba_train),
                "train_pr_auc": average_precision_score(y_train, proba_train),
                "best_params": best_params,
                "grid": str(sorted(spec["grid"].items(), key=lambda kv: kv[0])),
                "seconds": round(seconds, 1),
            }])

            # fold_scores last: only then does the fold count as completed
            append_csv(predictions, preds_path)
            append_csv(importances, imp_path)
            append_csv(fold_scores, scores_path)

            n_fitted += 1
            remaining = todo - n_fitted
            per_fold = (time.time() - run_start) / n_fitted
            eta_h = remaining * per_fold / 3600
            print(
                f"Repetition {repeat + 1:>2}/{n_repeats}, fold {fold}: "
                f"ROC-AUC {fold_scores['roc_auc'].iloc[0]:.3f} | {seconds / 60:5.1f} min | "
                f"{remaining} folds left, ~{eta_h:.1f} h",
                flush=True,
            )

    scores = pd.read_csv(scores_path)
    scores = scores[scores["repeat"] < n_repeats]
    scores.attrs["imp_path"] = str(imp_path)
    print_summary(scores, spec["name"], feature_set, y.mean())


def _grid_iter(grid):
    from sklearn.model_selection import ParameterGrid
    return ParameterGrid(grid)


if __name__ == "__main__":
    main()
