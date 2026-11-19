"""
RQ1 (supplementary): did the four models also make similar predictions for individual cases?

Nothing is trained: the stored outer-fold predictions of run_model.py are compared
descriptively (Section 4.8, last paragraph; reported in Section 5.1).
    - Spearman rank correlation between the predicted probabilities of each pair of models
    - proportion of cases to which all four models assign the same class (threshold 0.5)
    - proportion of cases each model classifies as IH, next to the observed prevalence

Usage (from the src folder, after run_model.py for all four models):
    python agreement.py

Output: results/agreement.csv (pairwise Spearman correlations)
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS = {"lr": "LR", "rf": "RF", "xgb": "XGBoost", "nn": "NN"}
FEATURE_SETS = ["primary", "extended"]
THRESHOLD = 0.5
KEY = ["repeat", "fold", "caseid"]


def load(feature_set):
    merged = None
    for m in MODELS:
        p = pd.read_csv(RESULTS_DIR / f"{m}_{feature_set}" / "predictions.csv")
        p = p[KEY + ["y_true", "y_proba"]].rename(columns={"y_proba": m})
        merged = p if merged is None else merged.merge(p.drop(columns="y_true"), on=KEY, validate="one_to_one")
    return merged


def main():
    rows = []
    for fs in FEATURE_SETS:
        d = load(fs)
        print(f"\nAGREEMENT BETWEEN THE MODELS ({fs.upper()}, {len(d)} predictions)")
        print("=" * 60)
        for a, b in combinations(MODELS, 2):
            rho = spearmanr(d[a], d[b])[0]
            rows.append({"feature_set": fs, "model_1": MODELS[a], "model_2": MODELS[b], "spearman_rho": round(rho, 3)})
            print(f"  Spearman rho {MODELS[a]:<8} vs {MODELS[b]:<8} {rho:.3f}")
        cls = d[list(MODELS)] >= THRESHOLD
        same = (cls.all(axis=1) | (~cls).all(axis=1)).mean()
        print(f"  All four models assign the same class: {same:.1%}")
        for m in MODELS:
            print(f"  Classified as IH by {MODELS[m]:<8} {cls[m].mean():.1%}")
        print(f"  Observed IH prevalence:          {d['y_true'].mean():.1%}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "agreement.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'agreement.csv'}")


if __name__ == "__main__":
    main()
