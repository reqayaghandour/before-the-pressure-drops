"""Figure 4 (t-SNE, from tsne_embedding.csv) and Figure 5 (mutual information, from
mutual_information.csv), at thesis text width with Palatino 11 pt."""
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from thesis_style import figure, save

BLUE, RED = "#4C72B0", "#C44E52"


def tsne(csv, out="tsne"):
    d = pd.read_csv(csv)
    W, H = 126, 72
    fig = figure(H)
    for k, fs in enumerate(["primary", "extended"]):
        ax = fig.add_axes([(1 + k * 63) / W, 1 / H, 61 / W, 58 / H])
        s = d[d.feature_set == fs]
        no, ih = s[s.ih == 0], s[s.ih == 1]
        ax.scatter(no.tsne_1, no.tsne_2, s=7, marker="o", color=BLUE, alpha=0.55, linewidths=0)
        ax.scatter(ih.tsne_1, ih.tsne_2, s=8, marker="x", color=RED, alpha=0.65, linewidths=0.6)
        ax.set_title(f"{fs.capitalize()} feature set", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=5, label="No IH"),
               Line2D([], [], marker="x", ls="", color=RED, markersize=6, markeredgewidth=1.2, label="IH")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.5)
    save(fig, out)


def mi(csv, out="mutual_information"):
    r = pd.read_csv(csv).iloc[::-1]
    n = len(r)
    W, H = 126, 8 + n * 4.6 + 14
    fig = figure(H)
    ax = fig.add_axes([53 / W, 14 / H, 70 / W, (n * 4.6) / H])
    colors = np.where(r.extended_only, RED, BLUE)
    y = np.arange(n)
    ax.barh(y, r.mi_mean, xerr=r.mi_sd, color=colors, height=0.7,
            error_kw=dict(lw=0.7, capsize=1.5, capthick=0.7))
    ax.set_yticks(y)
    ax.set_yticklabels(r.label)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, None)
    ax.set_xlabel("Mutual information with IH (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.legend(handles=[Patch(color=BLUE, label="Primary feature"),
                       Patch(color=RED, label="Pre-start MAP (extended set)")],
              frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    save(fig, out)


if __name__ == "__main__":
    what, csv = sys.argv[1], sys.argv[2]
    {"tsne": tsne, "mi": mi}[what](csv)
    print("saved", what)
