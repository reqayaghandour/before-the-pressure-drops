"""
Step 0: download the clinical information of all VitalDB cases.

The VitalDB open dataset (Lee et al., 2022) is publicly available under the Creative Commons
Attribution 4.0 International licence. The clinical table is downloaded from the VitalDB web
API; the arterial pressure tracks are downloaded later, case by case, with the vitaldb
Python package (build_ih_target.py and check_preopstart_and_target.py).

Usage (from the src folder):
    python download_data.py

Output: data/raw/clinical_data.csv
"""

from pathlib import Path

import pandas as pd

CASES_URL = "https://api.vitaldb.net/cases"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(CASES_URL)
    out = RAW_DIR / "clinical_data.csv"
    cases.to_csv(out, index=False)
    print(f"{len(cases)} cases saved to {out}")


if __name__ == "__main__":
    main()
