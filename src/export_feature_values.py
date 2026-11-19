"""
Export the feature values used for the SHAP beeswarm (Figure 7), so the figure can be
redrawn by figures/make_beeswarm.py. The rows are in the same order as
results/rq2/shap_values_<fs>.csv (the final model is fitted on all data in this order).

Usage (from the src folder, after rq2_feature_importance.py):
    python export_feature_values.py

Output:
    results/rq2/feature_values_primary.csv
    results/rq2/feature_values_extended.csv
Nothing is trained; only the input data are saved.
"""
from pathlib import Path

from preprocessing import get_X_y_groups, load_data

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "rq2"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    for fs in ["primary", "extended"]:
        X, _, _ = get_X_y_groups(df, fs)
        out = OUT_DIR / f"feature_values_{fs}.csv"
        X.to_csv(out, index=False)
        print(f"{fs}: {X.shape[0]} rows, {X.shape[1]} features -> {out}")


if __name__ == "__main__":
    main()
