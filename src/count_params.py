"""
Count how often each hyperparameter value was selected in the inner loops (Appendix A,
Table A1). Reads results/<model>_<feature_set>/fold_scores.csv; nothing is trained.

Usage (from the src folder):
    python count_params.py
"""

from pathlib import Path
from collections import Counter
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
for model in ["lr", "rf", "xgb", "nn"]:
    for fs in ["primary", "extended"]:
        path = RESULTS / f"{model}_{fs}" / "fold_scores.csv"
        if not path.exists():
            print(f"\n{model} {fs}: not found")
            continue
        counts = Counter()
        for params in pd.read_csv(path)["best_params"].dropna():
            for p in str(params).split(", "):
                counts[p] += 1
        print(f"\n{model.upper()} {fs}  (50 folds)")
        for p, n in sorted(counts.items()):
            print(f"  {p:<40} {n}/50")