"""
t-SNE and mutual information (Section 4.4, Figures 4 and 5), descriptive only.

Usage (from the src folder):
    python eda_tsne_mi.py

Nothing is trained for the models and nothing is used for feature selection, so there is
no leakage into the cross-validation results.

1. t-SNE: 2D representation of all cases after preprocessing, by IH status. Shows to what
   extent cases with and without IH are separable in the feature space.
2. Mutual information (MI) between each feature and IH: a model-free measure of how much
   information a feature carries about the outcome (including non-linear relationships).
   Averaged over 10 seeds, as the MI estimator for continuous features is random.
   Compared with the SHAP ranking of RQ2 (Spearman rho), if that is available.

Output in results/eda/:
    tsne.png / .pdf, tsne_embedding.csv, mutual_information.csv, mutual_information.png / .pdf
The thesis versions of the figures are redrawn from the CSV files by figures/make_tsne_mi.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE

from preprocessing import (EXTENDED_EXTRA_NUMERIC_FEATURES, ORDINAL_FEATURES, PRIMARY_CATEGORICAL_FEATURES,
                           PRIMARY_NUMERIC_FEATURES, build_preprocessor, get_X_y_groups, load_data)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "eda"
SEED = 42
N_MI_SEEDS = 10

LABELS = {
    "age": "Age", "height": "Height", "weight": "Weight", "bmi": "BMI", "asa": "ASA class",
    "preop_hb": "Haemoglobin", "preop_plt": "Platelets", "preop_pt": "PT", "preop_aptt": "aPTT",
    "preop_na": "Sodium", "preop_k": "Potassium", "preop_gluc": "Glucose", "preop_alb": "Albumin",
    "preop_ast": "AST", "preop_alt": "ALT", "preop_bun": "BUN", "preop_cr": "Creatinine",
    "sex": "Sex", "emop": "Emergency surgery", "department": "Department", "optype": "Operation type",
    "approach": "Surgical approach", "position": "Position", "preop_htn": "Hypertension",
    "preop_dm": "Diabetes", "preop_ecg_grouped": "ECG", "preop_pft": "Pulmonary function test",
    "preop5_map_mean": "Pre-start MAP (mean)", "preop5_map_min60": "Pre-start MAP (lowest 60 s)",
    "preop5_pct_below65": "Pre-start % < 65 mmHg",
}


def run_tsne(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Classes are distinguished by colour AND marker shape, so the figure also
    # works in greyscale and for colour-blind readers.
    styles = [(0, "#4C72B0", "o", "No IH"), (1, "#C44E52", "x", "IH")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    rows = []
    for ax, fs in zip(axes, ["primary", "extended"]):
        X, y, _ = get_X_y_groups(df, fs)
        Z = build_preprocessor(fs).fit_transform(X)  # for visualisation only
        emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                   random_state=SEED).fit_transform(Z)
        rows.append(pd.DataFrame({"feature_set": fs, "tsne_1": emb[:, 0], "tsne_2": emb[:, 1],
                                  "ih": y.values}))
        for val, color, marker, label in styles:
            m = (y == val).values
            if marker == "x":
                ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, color=color, marker=marker,
                           linewidths=0.6, label=label)
            else:
                ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.5, color=color, marker=marker,
                           linewidths=0, label=label)
        ax.set_title(f"{fs.capitalize()} feature set")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        print(f"t-SNE {fs}: done ({Z.shape[1]} columns after preprocessing)")
    axes[0].legend(frameon=False, markerscale=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tsne.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "tsne.pdf", facecolor="white")
    plt.close(fig)
    # Save the coordinates so the figure can be restyled later without rerunning t-SNE
    pd.concat(rows).to_csv(OUT_DIR / "tsne_embedding.csv", index=False)


def run_mi(df):
    numeric = PRIMARY_NUMERIC_FEATURES + EXTENDED_EXTRA_NUMERIC_FEATURES
    discrete = PRIMARY_CATEGORICAL_FEATURES + ORDINAL_FEATURES
    y = df["ih_target"].astype(int).values

    cols, is_discrete = [], []
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce")
        cols.append(v.fillna(v.median()).values)
        is_discrete.append(False)
    for c in discrete:
        v = df[c].astype(object).where(df[c].notna(), "Missing").astype(str)
        cols.append(pd.factorize(v)[0])
        is_discrete.append(True)
    X = np.column_stack(cols).astype(float)
    feats = numeric + discrete

    mi = np.array([mutual_info_classif(X, y, discrete_features=np.array(is_discrete), n_neighbors=3,
                                       random_state=s) for s in range(N_MI_SEEDS)])
    res = pd.DataFrame({"feature": feats, "label": [LABELS.get(f, f) for f in feats],
                        "mi_mean": mi.mean(0), "mi_sd": mi.std(0, ddof=1),
                        "extended_only": [f in EXTENDED_EXTRA_NUMERIC_FEATURES for f in feats]})
    res = res.sort_values("mi_mean", ascending=False).reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)

    print(f"\nMUTUAL INFORMATION WITH IH (nats; mean over {N_MI_SEEDS} seeds)")
    print("=" * 60)
    print(res[["rank", "label", "mi_mean", "mi_sd"]].round(4).to_string(index=False))

    for fs in ["primary", "extended"]:
        path = RESULTS_DIR / "rq2" / f"importance_{fs}.csv"
        if not path.exists():
            continue
        shap = pd.read_csv(path, index_col=0)["shap_mean_abs"]
        sub = res[res["feature"].isin(shap.index)].set_index("feature")
        rho, p = spearmanr(sub["mi_mean"], shap.reindex(sub.index))
        print(f"Agreement MI vs SHAP ranking ({fs}): Spearman rho = {rho:.2f} (p = {p:.3g})")

    res.round(5).to_csv(OUT_DIR / "mutual_information.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = np.where(plot["extended_only"], "#C44E52", "#4C72B0")
    ax.barh(plot["label"], plot["mi_mean"], xerr=plot["mi_sd"], color=colors, capsize=2)
    ax.set_xlabel("Mutual information with IH (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="Primary feature"),
                       Patch(color="#C44E52", label="Extended only (pre-start MAP)")],
              frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mutual_information.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "mutual_information.pdf", facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    run_mi(df)
    run_tsne(df)
    print(f"\nSaved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
