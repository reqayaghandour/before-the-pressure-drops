"""Figure 9: difference in FNR between each subgroup and the rest of the cohort, primary vs extended,
at thesis text width with Palatino 11 pt."""
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from thesis_style import figure, save

P, E = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "fnr_differences"
BLUE, RED = "#4C72B0", "#C44E52"
NAMES = {"F": "Female", "M": "Male", "Missing": "ASA missing", "Gynecology": "Gynaecology"}
p, e = pd.read_csv(P), pd.read_csv(E)
p = p[p.fnr_diff_vs_rest.notna()].reset_index(drop=True)
e = e.set_index(["variable", "subgroup"])

rows, ypos, heads, y = [], [], [], 0.0
for var in ["Sex", "ASA class", "Department", "Operation type"]:
    heads.append((y, var)); y += 1
    for _, r in p[p.variable == var].iterrows():
        rows.append(r); ypos.append(y); y += 1
    y += 0.4
ymax = y

W = 126
H = 10 + ymax * 4.2 + 12
fig = figure(H)
ax = fig.add_axes([38 / W, 12 / H, 85 / W, (ymax * 4.2) / H])
for r, yy in zip(rows, ypos):
    re_ = e.loc[(r.variable, r.subgroup)]
    for src, col, mk, dy in [(r, BLUE, "o", -0.17), (re_, RED, "s", 0.17)]:
        light = src.n_folds_used < 25
        a = 0.35 if light else 1.0
        ax.plot([src.diff_ci_low, src.diff_ci_high], [yy + dy] * 2, color=col, lw=1.0, alpha=a)
        ax.plot(src.fnr_diff_vs_rest, yy + dy, marker=mk, ms=3.6, color=col, alpha=a, ls="")
labels = [NAMES.get(r.subgroup, r.subgroup) + ("$^\\dagger$" if r.n_folds_used < 25 else "") for r in rows]
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
for yy, var in heads:
    ax.text(-0.02, yy, var, transform=ax.get_yaxis_transform(), ha="right", va="center", style="italic")
ax.axvline(0, color="black", lw=0.6, ls=":")
ax.set_ylim(ymax - 0.2, -0.8)
ax.set_xlim(-0.55, 0.6)
ax.set_xlabel("FNR difference (subgroup $-$ rest of cohort)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=2)
handles = [Line2D([], [], color=BLUE, marker="o", ms=3.6, lw=1.0, label="Primary feature set"),
           Line2D([], [], color=RED, marker="s", ms=3.6, lw=1.0, label="Extended feature set")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0))
save(fig, OUT)
print("saved", OUT, H)
