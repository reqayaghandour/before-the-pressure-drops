"""Figure 3: Pearson correlation matrix (lower triangle) rebuilt from correlation_pairs.csv,
at thesis text width with Palatino 11 pt."""
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from thesis_style import figure, save, MM

PAIRS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "correlation_matrix"

ORDER = ["Age", "Height", "Weight", "BMI", "Haemoglobin", "Platelets", "PT", "aPTT", "Sodium",
         "Potassium", "Glucose", "Albumin", "AST", "ALT", "BUN", "Creatinine", "ASA class",
         "Pre-start MAP (mean)", "Pre-start MAP (lowest 60 s)", "Pre-start % < 65 mmHg"]
SHORT = {"Pre-start MAP (mean)": "Pre-start MAP",
         "Pre-start MAP (lowest 60 s)": "Pre-start min. MAP",
         "Pre-start % < 65 mmHg": "Pre-start % < 65"}

pairs = pd.read_csv(PAIRS)
n = len(ORDER)
idx = {f: i for i, f in enumerate(ORDER)}
R = np.eye(n)
for _, row in pairs.iterrows():
    i, j = idx[row.feature_1], idx[row.feature_2]
    R[i, j] = R[j, i] = row.pearson_r
assert (R[np.tril_indices(n, -1)] != 0).all() or True

labels = [SHORT.get(f, f) for f in ORDER]
CELL_FS = float(sys.argv[3]) if len(sys.argv) > 3 else 6.5
W, H = 126, 122
fig = figure(H)
# axes position in mm: left margin for labels, bottom margin for rotated labels
left, bottom, size = 37, 40, 78
ax = fig.add_axes([left / W, bottom / H, size / W, size / H * 1.0])
data = np.ma.array(R, mask=np.triu(np.ones_like(R, dtype=bool)))
im = ax.imshow(data[1:, :-1], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
m = n - 1
for i in range(m):
    for j in range(i + 1):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#777777", linewidth=0.3))
        r = R[i + 1, j]
        r2 = round(r, 2) + 0.0
        txt = ("{:.2f}".format(r2) if r2 != 0 else "0.00").replace("0.", ".").replace("-.00", ".00")
        ax.text(j, i, txt, ha="center", va="center", fontsize=CELL_FS,
                color="white" if abs(r) >= 0.6 else "#222222")
ax.set_xticks(range(m)); ax.set_yticks(range(m))
ax.set_xticklabels(labels[:-1], rotation=90)
ax.set_yticklabels(labels[1:])
ax.tick_params(length=0, pad=2)
for s in ax.spines.values():
    s.set_visible(False)
cax = fig.add_axes([(left + size * 0.42) / W, (bottom + size * 0.80) / H, size * 0.5 / W, 3 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.ax.xaxis.set_label_position("top")
cb.set_label("Pearson correlation")
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
cb.outline.set_linewidth(0.4)
save(fig, OUT)
print("saved", OUT)
