"""
Exploratory data analysis (Section 4.4) on the final cohort (adults, no ASA 6).

Usage (from the src folder):
    python eda_table.py

Creates in results/eda/:
    table1.csv        patient characteristics by IH status (Table 1)
    missingness.csv   % missing per feature (primary + extended)
Nothing is trained and the data are not changed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import TARGET, GROUP, EXTENDED_EXTRA_NUMERIC_FEATURES, get_all_features, load_data

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "eda"

# Variables in Table 1 (label, column)
CONTINUOUS = [
    ("Age (years)", "age"),
    ("BMI (kg/m²)", "bmi"),
    ("Haemoglobin (g/dL)", "preop_hb"),
    ("Albumin (g/dL)", "preop_alb"),
    ("Creatinine (mg/dL)", "preop_cr"),
    ("Pre-start MAP, mean (mmHg)", "preop5_map_mean"),
    ("Pre-start MAP, lowest 60-s (mmHg)", "preop5_map_min60"),
    ("Pre-start time MAP < 65 mmHg (%)", "preop5_pct_below65"),
]
CATEGORICAL = [
    ("Sex", "sex"),
    ("ASA class", "asa"),
    ("Emergency surgery", "emop"),
    ("Department", "department"),
    ("Surgical approach", "approach"),
    ("Hypertension", "preop_htn"),
    ("Diabetes", "preop_dm"),
]


def med_iqr(s):
    s = s.dropna()
    if s.empty:
        return "–"
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{q2:.1f} [{q1:.1f}–{q3:.1f}]"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    groups = {"All": df, "IH": df[df[TARGET] == 1], "No IH": df[df[TARGET] == 0]}

    print("\nCLASS DISTRIBUTION (final cohort)")
    print("=" * 60)
    print(f"Cases: {len(df)} | patients: {df[GROUP].nunique()} | "
          f"repeated operations: {len(df) - df[GROUP].nunique()}")
    n_ih = int(df[TARGET].sum())
    print(f"IH: {n_ih} ({n_ih / len(df):.1%}) | no IH: {len(df) - n_ih} ({1 - n_ih / len(df):.1%})")

    rows = [{"Variable": "n", **{g: str(len(d)) for g, d in groups.items()}}]
    for label, col in CONTINUOUS:
        if col not in df:
            continue
        rows.append({"Variable": f"{label}, median [IQR]", **{g: med_iqr(d[col]) for g, d in groups.items()}})
    for label, col in CATEGORICAL:
        if col not in df:
            continue
        values = df[col].astype(object).where(df[col].notna(), "Missing")
        if col == "asa":
            values = values.map(lambda v: v if v == "Missing" else f"ASA {int(float(v))}")
        rows.append({"Variable": f"{label}, n (%)", **{g: "" for g in groups}})
        for level in sorted(values.unique(), key=lambda v: (v == "Missing", str(v))):
            row = {"Variable": f"   {level}"}
            for g, d in groups.items():
                k = int((values.loc[d.index] == level).sum())
                row[g] = f"{k} ({k / len(d):.1%})"
            rows.append(row)
    table1 = pd.DataFrame(rows)

    print("\nTABLE 1 - PATIENT CHARACTERISTICS BY IH STATUS")
    print("=" * 60)
    print(table1.to_string(index=False))
    table1.to_csv(OUT_DIR / "table1.csv", index=False)

    feats = get_all_features("extended")
    miss = (df[feats].isna().mean() * 100).round(1).sort_values(ascending=False)
    miss = miss.rename("missing_pct").reset_index().rename(columns={"index": "feature"})
    miss["feature_set"] = np.where(miss["feature"].isin(EXTENDED_EXTRA_NUMERIC_FEATURES), "extended only", "primary")

    print("\nMISSING PER FEATURE (%)")
    print("=" * 60)
    print(miss.to_string(index=False))
    miss.to_csv(OUT_DIR / "missingness.csv", index=False)

    print(f"\nSaved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
