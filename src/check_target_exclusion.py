"""
Check (Section 4.3): why do some cases have no valid target? Lists the cases without
surgical start/end times and the cases without valid ART_MBP during surgery.

Usage (from the src folder, after build_ih_target.py):
    python check_target_exclusion.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLINICAL_FILE = BASE_DIR / "data" / "raw" / "clinical_data.csv"
TARGET_FILE = BASE_DIR / "data" / "processed" / "ih_target.csv"

clinical_data = pd.read_csv(CLINICAL_FILE)
target_df = pd.read_csv(TARGET_FILE)

missing_target = target_df[
    target_df["ih_target"].isna()
].copy()

print("\nTOTAL NaN TARGETS")
print("=" * 50)
print(len(missing_target))

# Check missing surgery timing
merged = missing_target.merge(
    clinical_data[
        [
            "caseid",
            "opstart",
            "opend"
        ]
    ],
    on="caseid",
    how="left"
)

missing_timing = merged[
    merged["opstart"].isna() |
    merged["opend"].isna()
]

no_map_in_window = merged[
    (
        merged["opstart"].notna() &
        merged["opend"].notna()
    ) &
    (
        merged["valid_map_measurements"] == 0
    )
]

print("\nMISSING OPSTART/OPEND")
print("=" * 50)
print(missing_timing[
    [
        "caseid",
        "opstart",
        "opend"
    ]
])

print("\nNO VALID ART_MBP DURING OPSTART-OPEND")
print("=" * 50)
print(no_map_in_window[
    [
        "caseid",
        "opstart",
        "opend",
        "valid_map_measurements"
    ]
])