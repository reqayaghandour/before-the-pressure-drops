"""Figure 7: SHAP beeswarm (left) and feature permutation importance (right) for LR, primary (top)
and extended (bottom) feature set, at thesis text width with Palatino 11 pt.
Args: shap_primary shap_extended values_primary values_extended importance_primary importance_extended"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Patch
from thesis_style import figure, save

SP, SE, VP, VE, IP, IE = sys.argv[1:7]
OUT = sys.argv[7] if len(sys.argv) > 7 else "feature_importance_beeswarm"
BLUE, RED, GREY = "#4C72B0", "#C44E52", "#8c8c8c"
CMAP = mpl.colors.LinearSegmentedColormap.from_list("shap", ["#008bfb", "#7c4dd8", "#ff0052"])  # shap default: blue - purple - red
CAT = {"sex", "emop", "department", "optype", "approach", "position", "preop_htn", "preop_dm",
       "preop_ecg_grouped", "preop_pft"}
LABEL = {"age": "Age", "height": "Height", "weight": "Weight", "bmi": "BMI", "preop_hb": "Haemoglobin",
         "preop_plt": "Platelets", "preop_pt": "PT", "preop_aptt": "aPTT", "preop_na": "Sodium",
         "preop_k": "Potassium", "preop_gluc": "Glucose", "preop_alb": "Albumin", "preop_ast": "AST",
         "preop_alt": "ALT", "preop_bun": "BUN", "preop_cr": "Creatinine", "sex": "Sex",
         "emop": "Emergency surgery", "department": "Department", "optype": "Operation type",
         "approach": "Surgical approach", "position": "Position", "preop_htn": "Hypertension",
         "preop_dm": "Diabetes", "preop_ecg_grouped": "ECG", "preop_pft": "Pulmonary function test",
         "asa": "ASA class", "preop5_map_mean": "Pre-start MAP (mean)",
         "preop5_map_min60": "Pre-start MAP (lowest 60 s)", "preop5_pct_below65": "Pre-start % < 65 mmHg"}
TOP = 15
ROW = 4.4  # mm per feature


def swarm_offsets(x, nbins=100, rng=np.random.default_rng(0)):
    """Beeswarm offsets in the style of shap.summary_plot."""
    n = len(x)
    rng_ = x.max() - x.min()
    q = np.round(nbins * (x - x.min()) / (rng_ + 1e-8))
    inds = np.argsort(q + rng.standard_normal(n) * 1e-6)
    ys = np.zeros(n); layer, last = 0, None
    for i in inds:
        if q[i] != last:
            layer = 0
        ys[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1; last = q[i]
    return ys * 0.4 / (np.abs(ys).max() + 1)


def colours(v):
    v = pd.to_numeric(v, errors="coerce").values.astype(float)
    lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    if hi <= lo:
        lo, hi = np.nanmin(v), np.nanmax(v)
    return CMAP(np.clip((v - lo) / (hi - lo + 1e-12), 0, 1))


W = 126
L, SW, GAP, PW = 50, 37, 4, 33          # label column, SHAP panel, gap, perm panel (mm)
PH = TOP * ROW
H = 12 + PH + 10 + 7 + PH + 36
fig = figure(H)
ys_top = [H - 12, H - 12 - PH - 10 - 7]   # top edge of each panel row (mm from bottom)
for k, (fs, sf, vf, imf) in enumerate([("Primary", SP, VP, IP), ("Extended", SE, VE, IE)]):
    sh, val = pd.read_csv(sf), pd.read_csv(vf)
    imp = pd.read_csv(imf, index_col=0)
    feats = list(imp.sort_values("rank_shap").index[:TOP])
    top = ys_top[k]
    a1 = fig.add_axes([L / W, (top - PH) / H, SW / W, PH / H])
    a2 = fig.add_axes([(L + SW + GAP) / W, (top - PH) / H, PW / W, PH / H])
    fig.text(1 / W, (top + 3.5) / H, f"{fs} feature set", style="italic", va="bottom")
    for r, f in enumerate(feats):
        x = sh[f].values
        y = r + swarm_offsets(x)
        c = [GREY] * len(x) if f in CAT else colours(val[f])
        a1.scatter(x, y, s=0.8, c=c, lw=0, rasterized=True, zorder=3)
        col = RED if f.startswith("preop5") else BLUE
        a2.barh(r, imp.loc[f, "perm_mean"], xerr=imp.loc[f, "perm_sd"], color=col, height=0.7,
                error_kw=dict(lw=0.6, capsize=1.2, capthick=0.6))
    for a in (a1, a2):
        a.set_ylim(TOP - 0.5, -0.5)
        a.axvline(0, color="#b0b0b0" if a is a1 else "black", lw=0.6, zorder=1)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="y", length=0)
    a1.set_yticks(range(TOP))
    a1.set_yticklabels([LABEL[f] for f in feats])
    for t, f in zip(a1.get_yticklabels(), feats):
        if f.startswith("preop5"):
            t.set_color(RED)
    a2.set_yticks([])
    a1.set_xlim(-2.65, 1.85)
    a1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    a2.set_xlim(-0.004, 0.064)
    a2.set_xticks([0, 0.02, 0.04, 0.06]); a2.set_xticklabels(["0", ".02", ".04", ".06"])
    if k == 0:
        a1.set_title("SHAP values", pad=4)
        a2.set_title("Feature permutation\nimportance", pad=4)
    else:
        a1.set_xlabel("SHAP value (log-odds)")
        a2.set_xlabel("Decrease in\ntest ROC-AUC")
# legend row
cax = fig.add_axes([4 / W, 7.5 / H, 40 / W, 2.5 / H])
cb = mpl.colorbar.ColorbarBase(cax, cmap=CMAP, orientation="horizontal")
cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.outline.set_linewidth(0.4)
fig.text(2 / W, 12 / H, "Feature value (grey = categorical)", va="bottom")
fig.legend(handles=[Patch(color=BLUE, label="Primary feature"),
                    Patch(color=RED, label="Pre-start MAP")],
           loc="lower right", bbox_to_anchor=(1.0, 0.01), frameon=False, ncol=2, columnspacing=1.0, handlelength=1.4)
save(fig, OUT)
print("saved", OUT, round(H, 1))
