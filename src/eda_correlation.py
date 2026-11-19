"""
Correlation matrix of the numerical features (Section 4.4, Figure 3) on the final cohort.

Usage (from the src folder):
    python eda_correlation.py

Creates in results/eda/:
    correlation_matrix.png / .pdf   heatmap (Pearson, lower triangle)
    correlation_pairs.csv           all pairs, sorted by |r| (Pearson and Spearman)
Nothing is trained and the data are not changed. The thesis version of the figure is
redrawn from correlation_pairs.csv by figures/make_correlation.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import EXTENDED_EXTRA_NUMERIC_FEATURES, ORDINAL_FEATURES, PRIMARY_NUMERIC_FEATURES, load_data

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "eda"

LABELS = {
    "age": "Age", "height": "Height", "weight": "Weight", "bmi": "BMI", "asa": "ASA class",
    "preop_hb": "Haemoglobin", "preop_plt": "Platelets", "preop_pt": "PT", "preop_aptt": "aPTT",
    "preop_na": "Sodium", "preop_k": "Potassium", "preop_gluc": "Glucose", "preop_alb": "Albumin",
    "preop_ast": "AST", "preop_alt": "ALT", "preop_bun": "BUN", "preop_cr": "Creatinine",
    "preop5_map_mean": "Pre-start MAP (mean)", "preop5_map_min60": "Pre-start MAP (lowest 60 s)",
    "preop5_pct_below65": "Pre-start % < 65 mmHg",
}
STRONG = 0.5
FULL_SQUARE = False  # True = full square; False = lower triangle only


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    cols = PRIMARY_NUMERIC_FEATURES + ORDINAL_FEATURES + EXTENDED_EXTRA_NUMERIC_FEATURES
    X = df[cols].apply(pd.to_numeric, errors="coerce").rename(columns=LABELS)

    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")

    iu = np.triu_indices(len(X.columns), k=1)
    pairs = pd.DataFrame({
        "feature_1": X.columns[iu[0]],
        "feature_2": X.columns[iu[1]],
        "pearson_r": pearson.values[iu].round(2),
        "spearman_rho": spearman.values[iu].round(2),
    })
    pairs = pairs.reindex(pairs["pearson_r"].abs().sort_values(ascending=False).index)
    pairs.to_csv(OUT_DIR / "correlation_pairs.csv", index=False)

    print(f"\nCORRELATIONS BETWEEN NUMERICAL FEATURES (n = {len(df)} cases)")
    print("=" * 78)
    print(f"Pairs with |r| >= {STRONG} (Pearson or Spearman):")
    strong = pairs[(pairs["pearson_r"].abs() >= STRONG) | (pairs["spearman_rho"].abs() >= STRONG)]
    print(strong.to_string(index=False) if len(strong) else "  none")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(pearson)
        if FULL_SQUARE:
            data = pearson.values
        else:
            data = np.ma.array(pearson.values, mask=np.triu(np.ones_like(pearson, dtype=bool)))

        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n if FULL_SQUARE else i):
                r = pearson.values[i, j]
                ax.text(j, i, f"{r:.2f}".replace("-", "\u2212"), ha="center", va="center",
                        fontsize=6.5, color="white" if abs(r) >= 0.6 else "#222222")
        # thin borders around each shown cell
        from matplotlib.patches import Rectangle
        for i in range(n):
            for j in range(n if FULL_SQUARE else i):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#555555", linewidth=0.4))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(pearson.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(pearson.columns, fontsize=8)
        if not FULL_SQUARE:
            ax.set_xlim(-0.5, n - 1.5)
            ax.set_ylim(n - 0.5, 0.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Pearson correlation", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "correlation_matrix.png", dpi=300, facecolor="white")
        fig.savefig(OUT_DIR / "correlation_matrix.pdf", facecolor="white")
        plt.close(fig)
    except ImportError:
        print("matplotlib not found; figure skipped.")

    print(f"\nSaved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
