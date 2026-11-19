"""
Step 2: merge the outcome with the clinical data and keep the cases with a valid target.

Usage (from the src folder):
    python merge_target_clinical.py

Input:  data/raw/clinical_data.csv, data/processed/ih_target.csv
Output: data/processed/analysis_dataset.csv
(The exclusion of patients younger than 18 and of ASA 6 cases happens in preprocessing.py.)
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLINICAL_FILE = BASE_DIR / "data" / "raw" / "clinical_data.csv"
TARGET_FILE = BASE_DIR / "data" / "processed" / "ih_target.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "analysis_dataset.csv"

clinical_data = pd.read_csv(CLINICAL_FILE)
target_data = pd.read_csv(TARGET_FILE)

merged_data = clinical_data.merge(
    target_data,
    on="caseid",
    how="left"
)

print("\nMERGE CHECK")
print("=" * 50)

print("Clinical rows:", len(clinical_data))
print("Target rows:", len(target_data))
print("Merged rows:", len(merged_data))

# Keep only cases with a valid binary target
analysis_data = merged_data[
    merged_data["ih_target"].notna()
].copy()

print("\nANALYSIS DATASET")
print("=" * 50)

print("Cases with valid target:", len(analysis_data))

print("\nTarget distribution:")
print(
    analysis_data["ih_target"].value_counts()
)

print("\nTarget percentages:")
print(
    analysis_data["ih_target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
)

analysis_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)