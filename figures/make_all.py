"""
Redraw all thesis figures (Figures 1-9) from the CSV files in results/, at the exact text
width of the thesis (126 mm) in Palatino 11 pt, so they can be included with width=\\textwidth.

Usage (from the figures folder, after running the analysis scripts in src/):
    python make_all.py

Output: figures/output/<name>.pdf and .png
Fonts: TeX Gyre Pagella or Palatino is used when installed; otherwise DejaVu Serif.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
OUT = HERE / "output"

JOBS = [
    # (script, arguments, output name)
    ("make_flowchart.py", ["10"], "method_flowchart_colour"),                       # Figure 1
    ("make_cohort_flow.py", [], "cohort_flow_diagram"),                              # Figure 2
    ("make_correlation.py", [R / "eda/correlation_pairs.csv"], "correlation_matrix"),  # Figure 3
    ("make_tsne_mi.py", ["tsne", R / "eda/tsne_embedding.csv"], "tsne"),              # Figure 4
    ("make_tsne_mi.py", ["mi", R / "eda/mutual_information.csv"], "mutual_information"),  # Figure 5
    ("make_boxplot.py", [R, R / "xgb_primary/fold_scores.csv", R / "xgb_extended/fold_scores.csv"],
     "roc_auc_boxplot"),                                                             # Figure 6
    ("make_beeswarm.py", [R / "rq2/shap_values_primary.csv", R / "rq2/shap_values_extended.csv",
                          R / "rq2/feature_values_primary.csv", R / "rq2/feature_values_extended.csv",
                          R / "rq2/importance_primary.csv", R / "rq2/importance_extended.csv"],
     "feature_importance_beeswarm"),                                                 # Figure 7
    ("make_morf_sofi.py", [R / "rq2/morf_lerf_primary.csv", R / "rq2/morf_lerf_extended.csv",
                           R / "rq2/sofi_curves_primary.csv", R / "rq2/sofi_curves_extended.csv"],
     "morf_lerf"),                                                                   # Figure 8
    ("make_fnr.py", [R / "rq3/subgroups_primary.csv", R / "rq3/subgroups_extended.csv"],
     "fnr_differences"),                                                             # Figure 9
]


def main():
    OUT.mkdir(exist_ok=True)
    for script, args, name in JOBS:
        args = [str(a) for a in args]
        cmd = [sys.executable, str(HERE / script), *args]
        if script != "make_tsne_mi.py":  # make_tsne_mi.py saves under its own name (in OUT)
            cmd.append(str(OUT / name))
        print(f"{name} ...", flush=True)
        subprocess.run(cmd, check=True, cwd=OUT)


if __name__ == "__main__":
    main()
