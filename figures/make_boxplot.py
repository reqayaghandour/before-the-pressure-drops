"""Figure 6: test ROC-AUC per model and feature set over the 50 outer folds (boxplot)."""
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from thesis_style import figure, save

Z, XGB_P, XGB_E = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else "roc_auc_boxplot"
BLUE, RED = "#4C72B0", "#C44E52"
names = {"lr": "LR", "rf": "RF", "xgb": "XGBoost", "nn": "NN"}
data = {}
for m in names:
    for fs in ["primary", "extended"]:
        if m == "xgb":
            d = pd.read_csv(XGB_P if fs == "primary" else XGB_E)
        else:
            d = pd.read_csv(f"{Z}/{m}_{fs}/fold_scores.csv")
        data[(m, fs)] = d.roc_auc.values

W, H = 126, 78
fig = figure(H)
ax = fig.add_axes([16 / W, 9 / H, 108 / W, 60 / H])
rng = np.random.default_rng(0)
for i, m in enumerate(names):
    for k, (fs, col) in enumerate([("primary", BLUE), ("extended", RED)]):
        x = i + (k - 0.5) * 0.36
        v = data[(m, fs)]
        ax.boxplot(v, positions=[x], widths=0.28, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col, lw=0.8),
                   medianprops=dict(color=col, lw=1.2), whiskerprops=dict(color=col, lw=0.8),
                   capprops=dict(color=col, lw=0.8))
        ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=col, alpha=0.6, lw=0, zorder=3)
        ax.scatter([x], [v.mean()], marker="D", s=14, color="white", edgecolor="black", lw=0.6, zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels(names.values())
ax.set_ylabel("Test ROC-AUC")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(handles=[Patch(facecolor=BLUE, alpha=0.5, label="Primary feature set"),
                   Patch(facecolor=RED, alpha=0.5, label="Extended feature set")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
save(fig, OUT)
print("saved", OUT)
