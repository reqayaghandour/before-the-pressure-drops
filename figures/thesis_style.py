"""Shared figure style: thesis text width (126 mm) and body font (Palatino, 11 pt)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

TEXTWIDTH_MM = 126.0
MM = 1 / 25.4
FONT_SIZE = 11

_PAGELLA = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
if os.path.isdir(_PAGELLA):
    for f in ["regular", "italic", "bold", "bolditalic"]:
        p = _PAGELLA + f"texgyrepagella-{f}.otf"
        if os.path.exists(p):
            fm.fontManager.addfont(p)

plt.rcParams.update({
    # TeX Gyre Pagella = Palatino (classicthesis); fallbacks for other machines
    "font.family": [f for f in ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "Book Antiqua"]
                    if f in {x.name for x in fm.fontManager.ttflist}] or ["DejaVu Serif"],
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def figure(height_mm, width_mm=TEXTWIDTH_MM, **kw):
    """Figure of exactly width_mm x height_mm (include in LaTeX with width=\\textwidth)."""
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kw)


def save(fig, path_without_ext):
    fig.savefig(path_without_ext + ".pdf")
    fig.savefig(path_without_ext + ".png", dpi=300)
    plt.close(fig)
