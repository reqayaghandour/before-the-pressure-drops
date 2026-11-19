"""
RQ3: how does the false negative rate (FNR) of the selected model (LR) differ across
subgroups defined by sex, ASA class and type of surgery?

Usage (from the src folder, after run_model.py lr primary/extended):
    python rq3_subgroups.py primary
    python rq3_subgroups.py extended

Nothing is trained: the stored test-fold predictions of run_model.py are used
(results/lr_<fs>/predictions.csv; each case occurs exactly once per repetition).

Definitions (threshold 0.5, as in the main analysis):
    FNR = FN / (FN + TP)   proportion of IH cases that the model misses (= 1 - recall)
    FPR = FP / (FP + TN)   proportion of cases without IH that receive a false alarm

Per subgroup:
    - FNR, FPR, recall, precision: mean ± SD over the 50 test folds
      (a fold counts only if the subgroup and the rest of the cohort each contain
      >= MIN_POSITIVES IH cases in that fold)
    - difference in FNR from the rest of the cohort, paired per fold, with a 95% interval
      over the folds (descriptive; the folds are not independent)
    - ROC-AUC within the subgroup (requires >= 10 IH and >= 10 non-IH cases per fold)
    - average confusion matrix per repetition (TN, FP, FN, TP)

Subgroups: sex, asa, department (type of surgery at department level) and optype
(more detailed; small groups get wider intervals).

Output in results/rq3/:
    subgroups_<fs>.csv, confusion_<fs>.csv, fnr_<fs>.png (quick-look figure)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from preprocessing import FEATURE_SETS, load_data

THRESHOLD = 0.5
MIN_POSITIVES = 10
SUBGROUP_VARS = {
    "sex": "Sex",
    "asa": "ASA class",
    "department": "Department",
    "optype": "Operation type",
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "rq3"


def rates(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fnr": fn / (fn + tp) if fn + tp else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
    }


def label_values(df):
    out = df.copy()
    out["asa"] = out["asa"].map(lambda v: f"ASA {int(v)}" if pd.notna(v) else "Missing")
    for c in ["sex", "department", "optype"]:
        out[c] = out[c].fillna("Missing").astype(str)
    return out


def main():
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "primary"
    if feature_set not in FEATURE_SETS:
        sys.exit(f"Choose a feature set from {FEATURE_SETS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = RESULTS_DIR / f"lr_{feature_set}" / "predictions.csv"
    if not pred_path.exists():
        sys.exit(f"{pred_path} is missing. First run: python run_model.py lr {feature_set}")
    preds = pd.read_csv(pred_path)

    data = label_values(load_data()[["caseid", *SUBGROUP_VARS]])
    preds = preds.merge(data, on="caseid", how="left", validate="many_to_one")
    if preds[list(SUBGROUP_VARS)].isna().any().any():
        sys.exit("Not all predictions could be linked to subgroups.")
    preds["pred"] = (preds["y_proba"] >= THRESHOLD).astype(int)

    n_repeats = preds["repeat"].nunique()
    overall_fold = (preds.groupby(["repeat", "fold"])
                    .apply(lambda g: pd.Series(rates(g["y_true"], g["pred"])), include_groups=False))

    print(f"\nRQ3 - FALSE NEGATIVE RATE PER SUBGROUP, LOGISTIC REGRESSION ({feature_set.upper()})")
    print("=" * 92)
    print(f"Threshold {THRESHOLD}; {len(overall_fold)} test folds; a subgroup counts in a fold with "
          f">= {MIN_POSITIVES} IH cases")
    overall_auc = preds.groupby(["repeat", "fold"]).apply(
        lambda g: roc_auc_score(g["y_true"], g["y_proba"]), include_groups=False)
    print(f"Whole cohort: FNR {overall_fold['fnr'].mean():.3f} ± {overall_fold['fnr'].std(ddof=1):.3f} | "
          f"FPR {overall_fold['fpr'].mean():.3f} ± {overall_fold['fpr'].std(ddof=1):.3f} | "
          f"ROC-AUC {overall_auc.mean():.3f} ± {overall_auc.std(ddof=1):.3f}")

    rows, conf_rows = [], []
    for var, var_label in SUBGROUP_VARS.items():
        print(f"\n{var_label.upper()}")
        print(f"  {'subgroup':<26}{'n/rep.':>8}{'IH %':>7}{'FNR':>16}{'FPR':>16}"
              f"{'FNR - rest [95% int.]':>28}{'folds':>7}{'ROC-AUC within':>18}")
        for level, sub in preds.groupby(var):
            fold_stats, diffs = [], []
            for (r, f), g in preds.groupby(["repeat", "fold"]):
                in_grp = g[var] == level
                if (g.loc[in_grp, "y_true"] == 1).sum() < MIN_POSITIVES:
                    continue
                if (g.loc[~in_grp, "y_true"] == 1).sum() < MIN_POSITIVES:
                    continue
                s_in = rates(g.loc[in_grp, "y_true"], g.loc[in_grp, "pred"])
                s_out = rates(g.loc[~in_grp, "y_true"], g.loc[~in_grp, "pred"])
                # ROC-AUC within the subgroup: threshold-independent discrimination
                n_neg = (g.loc[in_grp, "y_true"] == 0).sum()
                s_in["auc"] = (roc_auc_score(g.loc[in_grp, "y_true"], g.loc[in_grp, "y_proba"])
                               if n_neg >= MIN_POSITIVES else np.nan)
                fold_stats.append(s_in)
                diffs.append(s_in["fnr"] - s_out["fnr"])
            n_per_rep = len(sub) / n_repeats
            prev = sub["y_true"].mean() * 100

            counts = rates(sub["y_true"], sub["pred"])
            conf_rows.append({
                "variable": var_label, "subgroup": level, "n_per_repeat": round(n_per_rep, 1),
                **{k: round(counts[k] / n_repeats, 1) for k in ["tn", "fp", "fn", "tp"]},
            })

            if not fold_stats:
                print(f"  {str(level):<26}{n_per_rep:>8.0f}{prev:>7.1f}"
                      f"{'too small':>16}{'':>16}{'':>28}{0:>7}")
                rows.append({"variable": var_label, "subgroup": level, "n_per_repeat": n_per_rep,
                             "ih_pct": prev, "n_folds_used": 0})
                continue
            fs = pd.DataFrame(fold_stats)
            d = np.array(diffs)
            half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            row = {
                "variable": var_label, "subgroup": level, "n_per_repeat": n_per_rep, "ih_pct": prev,
                "fnr_mean": fs["fnr"].mean(), "fnr_sd": fs["fnr"].std(ddof=1),
                "fpr_mean": fs["fpr"].mean(), "fpr_sd": fs["fpr"].std(ddof=1),
                "recall_mean": fs["recall"].mean(), "precision_mean": fs["precision"].mean(),
                "auc_mean": fs["auc"].mean(), "auc_sd": fs["auc"].std(ddof=1),
                "auc_folds": int(fs["auc"].notna().sum()),
                "fnr_diff_vs_rest": d.mean(), "diff_ci_low": d.mean() - half,
                "diff_ci_high": d.mean() + half, "n_folds_used": len(d),
            }
            rows.append(row)
            interval = f"{d.mean():+.3f} [{d.mean() - half:+.3f}, {d.mean() + half:+.3f}]"
            auc_txt = (f"{row['auc_mean']:.3f} ± {row['auc_sd']:.3f}"
                       if row["auc_folds"] > 1 else "too small")
            print(f"  {str(level):<26}{n_per_rep:>8.0f}{prev:>7.1f}"
                  f"{row['fnr_mean']:>9.3f} ± {row['fnr_sd']:.3f}"
                  f"{row['fpr_mean']:>9.3f} ± {row['fpr_sd']:.3f}{interval:>28}{len(d):>7}"
                  f"{auc_txt:>18}")

    print("\nHow to read:")
    print("  n/rep.     = number of cases in the subgroup (each case occurs once per repetition)")
    print("  FNR        = proportion of IH cases the model misses; higher = more missed hypotension")
    print("  FNR - rest = difference from all other cases in the same fold; positive = missed more often")
    print("  95% int.   = descriptive interval over folds (folds are not independent; no formal test)")
    print("  ROC-AUC within = how well the model separates IH from no IH WITHIN the subgroup,")
    print("                   independent of the threshold. Similar to the cohort -> the FNR difference")
    print("                   mainly reflects baseline risk + threshold; lower -> weaker discrimination.")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT_DIR / f"subgroups_{feature_set}.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT_DIR / f"confusion_{feature_set}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_vars = [v for v in SUBGROUP_VARS.values() if v != "Operation type"]
        fig, axes = plt.subplots(1, len(plot_vars), figsize=(12, 4.2), sharey=True,
                                 gridspec_kw={"width_ratios": [max(2, (res["variable"] == v).sum())
                                                               for v in plot_vars]})
        overall = overall_fold["fnr"].mean()
        for ax, v in zip(axes, plot_vars):
            block = res[(res["variable"] == v) & res["fnr_mean"].notna()]
            ax.bar(block["subgroup"].astype(str), block["fnr_mean"], yerr=block["fnr_sd"],
                   color="#4C72B0", alpha=0.85, capsize=3)
            ax.axhline(overall, color="#C44E52", linestyle="--", linewidth=1,
                       label=f"whole cohort ({overall:.2f})")
            ax.set_title(v)
            ax.tick_params(axis="x", rotation=35)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("False negative rate (mean ± SD over folds)")
        axes[0].legend(frameon=False, loc="upper left")
        fig.suptitle(f"Missed hypotension by subgroup, logistic regression ({feature_set})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fnr_{feature_set}.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib not found; figure skipped.")

    print(f"\nSaved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
