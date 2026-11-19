"""Figure 8: MORF/LERF curves for the SHAP ranking (solid), the SOFI ranking (dashed) and a random
order, at thesis text width with Palatino 11 pt."""
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from thesis_style import figure, save

ML_P, ML_E, SC_P, SC_E = sys.argv[1:5]
OUT = sys.argv[5] if len(sys.argv) > 5 else "morf_lerf"
RED, BLUE, GREY = "#C44E52", "#4C72B0", "#7f7f7f"

W, H = 126, 84
fig = figure(H)
for k, (fs, mlf, scf) in enumerate([("Primary", ML_P, SC_P), ("Extended", ML_E, SC_E)]):
    ax = fig.add_axes([(15 + k * 57) / W, 32 / H, 50 / W, 44 / H])
    ml, sc = pd.read_csv(mlf), pd.read_csv(scf)
    x = ml.n_removed / ml.n_removed.max()
    for pre, src, col, ls, band in [("MORF", ml, RED, "-", True), ("LERF", ml, BLUE, "-", True),
                                    ("SOFI_MORF", sc, RED, "--", False), ("SOFI_LERF", sc, BLUE, "--", False),
                                    ("RANDOM", ml, GREY, "-", True)]:
        m, s = src[f"{pre}_mean"], src[f"{pre}_sd"]
        ax.plot(x, m, color=col, ls=ls, lw=1.2)
        if band:
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.10, lw=0)
    ax.axhline(0.5, color="black", lw=0.6, ls=":")
    ax.set_title(f"{fs} feature set", pad=3)
    ax.set_xlabel("Proportion replaced")
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(0.35, 0.78)
    ax.spines[["top", "right"]].set_visible(False)
    if k == 0:
        ax.set_ylabel("Test ROC-AUC")
    else:
        ax.set_yticklabels([])
handles = [Line2D([], [], color=RED, lw=1.2, label="Most relevant first (MORF)"),
           Line2D([], [], color=BLUE, lw=1.2, label="Least relevant first (LERF)"),
           Line2D([], [], color=GREY, lw=1.2, label="Random order"),
           Line2D([], [], color="black", lw=1.2, label="SHAP ranking"),
           Line2D([], [], color="black", lw=1.2, ls="--", label="SOFI ranking")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.2, handlelength=1.8)
save(fig, OUT)
print("saved", OUT)
