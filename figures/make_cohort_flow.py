"""Figure 2: cohort flow diagram at thesis text width, Palatino 11 pt."""
import sys
from matplotlib.patches import FancyBboxPatch
from thesis_style import figure, save, FONT_SIZE

OUT = sys.argv[1] if len(sys.argv) > 1 else "cohort_flow_diagram"
W, H = 126, 124
fig = figure(H)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
LW = 0.8
MAIN = ("#bdd0ee", "#5b7bbf")
EXCL = ("#e6e6e6", "#8c8c8c")
FINAL = ("#d9d9d9", "#7f7f7f")


def box(cx, cy, w, h, s, sty, ha="center"):
    fc, ec = sty
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fc, ec=ec, lw=LW, zorder=3))
    x = cx if ha == "center" else cx - w / 2 + 3
    ax.text(x, cy, s, ha=ha, va="center", fontsize=FONT_SIZE, linespacing=1.15, zorder=4)


def arrow(p, q):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color="#333333",
                                lw=LW, shrinkA=0, shrinkB=0))


MX, MW = 32, 58           # main column
EX, EW = 96, 58           # exclusion column
main = [
    (8, "VitalDB\n6,388 cases (6,090 patients)"),
    (42, "3,526 cases with\ninvasive MAP"),
    (76, "3,480 cases with a valid\noutcome, aged 18 or older"),
    (113, "Final cohort\n3,468 cases (3,353 patients)\nIH: 1,833 (52.9%)"),
]
excl = [
    (25, "Excluded: 2,862 cases\nwithout invasive MAP\n(ART_MBP)"),
    (59, "Excluded: 3 cases without\nMAP during surgery and\n43 cases aged under 18"),
    (94.5, "Excluded: 12 cases with\nASA class 6 (brain-dead\norgan donors)"),
]
heights = [12, 12, 12, 17]
for (cy, s), h in zip(main, heights):
    box(MX, cy, MW, h, s, FINAL if cy == main[-1][0] else MAIN)
for i in range(len(main) - 1):
    y0 = main[i][0] + heights[i] / 2
    y1 = main[i + 1][0] - heights[i + 1] / 2
    arrow((MX, y0), (MX, y1))
    ey = excl[i][0]
    arrow((MX, ey), (EX - EW / 2, ey))
for cy, s in excl:
    box(EX, cy, EW, 17, s, EXCL, ha="left")
save(fig, OUT)
print("saved", OUT)
