"""Method flowchart (Figure 1), drawn at the exact text width of the thesis (126 mm)
in TeX Gyre Pagella (Palatino), so that the text matches the body text when included
with width=\\textwidth (no scaling)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, Rectangle

import thesis_style  # noqa: F401  (registers the Palatino font and the shared settings)
FS = float(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else "method_flowchart_colour"

W, H = 126, 188  # mm
MM = 1 / 25.4
STY = {
    "data": ("#d9d9d9", "#7f7f7f"),
    "process": ("#bdd0ee", "#5b7bbf"),
    "features": ("#f8e6a9", "#c9a33c"),
    "item": ("#e6e6e6", "#8c8c8c"),
    "decision": ("#b3dfd0", "#5a9c84"),
    "rq": ("#edc2dc", "#b04f8f"),
}
LW = 0.8

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # y downwards, in mm
ax.axis("off")


def label(x, y, s, **kw):
    ax.text(x, y, s, ha="center", va="center", fontsize=FS, linespacing=1.15, zorder=5, **kw)


def rbox(cx, cy, w, h, s, st):
    fc, ec = STY[st]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fc, ec=ec, lw=LW, zorder=3))
    label(cx, cy, s)


def cyl(cx, cy, w, h, s, st="data"):
    fc, ec = STY[st]
    e = h * 0.14
    top, bot = cy - h / 2 + e, cy + h / 2 - e
    ax.add_patch(Rectangle((cx - w / 2, top), w, bot - top, fc=fc, ec="none", zorder=3))
    ax.plot([cx - w / 2, cx - w / 2], [top, bot], color=ec, lw=LW, zorder=4)
    ax.plot([cx + w / 2, cx + w / 2], [top, bot], color=ec, lw=LW, zorder=4)
    ax.add_patch(Ellipse((cx, bot), w, 2 * e, fc=fc, ec=ec, lw=LW, zorder=3))
    ax.add_patch(Rectangle((cx - w / 2 + 0.05, top), w - 0.1, bot - top, fc=fc, ec="none", zorder=3.5))
    ax.add_patch(Ellipse((cx, top), w, 2 * e, fc=fc, ec=ec, lw=LW, zorder=4))
    label(cx, cy + e * 0.45, s)


def para(cx, cy, w, h, s, st="process", skew=4):
    fc, ec = STY[st]
    pts = [(cx - w / 2 + skew, cy - h / 2), (cx + w / 2, cy - h / 2),
           (cx + w / 2 - skew, cy + h / 2), (cx - w / 2, cy + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=LW, zorder=3))
    label(cx, cy, s)


def diamond(cx, cy, w, h, s, st="decision"):
    fc, ec = STY[st]
    pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=LW, zorder=3))
    label(cx, cy, s)


def ellipse(cx, cy, w, h, s, st="process"):
    fc, ec = STY[st]
    ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec=ec, lw=LW, zorder=3))
    label(cx, cy, s)


def seg(p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#333333", lw=LW, zorder=2, solid_capstyle="butt")


def arrow(p, q):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                color="#333333", lw=LW, shrinkA=0, shrinkB=0))


# ---- Row 1: data -> outcome/cohort -> final cohort
cyl(19, 15.5, 34, 30, "VitalDB\nclinical data and\narterial MAP\n6,388 cases")
para(63, 15, 44, 16, "Outcome definition\nand cohort selection")
cyl(107, 15, 32, 22, "Final cohort\n3,468 cases")
arrow((36, 15), (42.5, 15))
arrow((85, 15), (91, 15))

# ---- Row 2: EDA
rbox(107, 36, 32, 9, "EDA", "item")
arrow((107, 26), (107, 31.5))

# ---- Row 3: feature sets
seg((107, 40.5), (107, 45)); seg((34, 45), (107, 45))
arrow((34, 45), (34, 48)); arrow((96, 45), (96, 48))
rbox(34, 55, 60, 14, "Primary feature set\n27 preoperative features", "features")
rbox(96, 55, 52, 14, "Extended feature set\nprimary + pre-start MAP", "features")
seg((34, 62), (34, 66)); seg((96, 62), (96, 66)); seg((34, 66), (96, 66))
arrow((63, 66), (63, 70))

# ---- Row 4: nested CV
rbox(63, 79.5, 104, 19, "Repeated nested cross-validation\n10 repetitions × 5 outer folds, 5 inner folds\npatient-grouped and stratified", "process")

# ---- Row 5: models -> tuning -> evaluation
arrow((54, 89), (54, 92))
arrow((102, 89), (102, 92))
rbox(15, 105, 28, 18, "Four models\nLR, RF,\nXGBoost, NN", "item")
diamond(54, 105, 46, 26, "Hyperparameter\ntuning")
rbox(102, 105, 42, 26, "Model evaluation\n(outer test folds)\nROC-AUC, PR-AUC,\nF1, precision, recall", "process")
arrow((29, 105), (31, 105))
arrow((77, 105), (81, 105))

# ---- Row 6: RQ1 -> model selection
ellipse(102, 133, 46, 22, "RQ1: model\ncomparison (corrected\nt-test on ROC-AUC)")
arrow((102, 118), (102, 122))
diamond(54, 133, 36, 20, "Model\nselection")
arrow((79, 133), (72, 133))

# ---- Row 7: RQ2 / RQ3
seg((54, 143), (54, 147)); seg((31, 147), (94, 147))
arrow((31, 147), (31, 151)); arrow((94, 147), (94, 151))
rbox(31, 160, 58, 18, "RQ2: feature importance\nfeature permutation, SHAP\nMORF/LERF, SOFI", "rq")
rbox(94, 160, 58, 18, "RQ3: subgroup errors\nFNR and FPR per subgroup\nwithin-subgroup ROC-AUC", "rq")

# ---- Legend
items = [("data", "Data"), ("process", "Process"), ("features", "Feature sets"),
         ("item", "Other steps"), ("decision", "Decision"), ("rq", "Research questions")]
for i, (st, lab) in enumerate(items):
    r, c = divmod(i, 3)
    x, y = 2 + c * 40, 176 + r * 7
    fc, ec = STY[st]
    ax.add_patch(Rectangle((x, y - 2), 5, 4, fc=fc, ec=ec, lw=LW))
    ax.text(x + 7, y, lab, ha="left", va="center", fontsize=FS)

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("saved", OUT)
