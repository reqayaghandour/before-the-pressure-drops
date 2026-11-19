"""
Step 1: build the outcome (IH) for every VitalDB case.

IH = 1 if the invasive arterial MAP (Solar8000/ART_MBP) stays below 65 mmHg for at least
60 seconds between surgical start (opstart) and end (opend), and IH = 0 otherwise.
Values below 20 or above 200 mmHg are treated as artefacts (set to missing), and gaps of up
to 4 seconds are allowed within an episode. Cases without valid ART_MBP during surgery, or
without surgical start/end times, receive no target (NaN).

Usage (from the src folder; downloads the MAP track of every case, which takes several hours):
    python build_ih_target.py

Input:  data/raw/clinical_data.csv
Output: data/processed/ih_target.csv
"""

import vitaldb
import numpy as np
import pandas as pd
from pathlib import Path

MAP_TRACK = "Solar8000/ART_MBP"

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLINICAL_FILE = RAW_DIR / "clinical_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "ih_target.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

clinical_data = pd.read_csv(CLINICAL_FILE)


def classify_hypotension(
    map_values,
    opstart,
    opend,
    threshold=65,
    min_duration=60,
    min_map=20,
    max_map=200,
    max_gap=4
):
    # Only use the surgical period
    surgical_map = map_values[
        int(opstart):int(opend)
    ]

    clean_map = surgical_map.copy()

    # Remove implausible values
    clean_map[
        (clean_map < min_map) |
        (clean_map > max_map)
    ] = np.nan

    valid_indices = np.where(
        ~np.isnan(clean_map)
    )[0]

    valid_values = clean_map[
        valid_indices
    ]

    valid_measurements = len(valid_values)

    # No usable ART_MBP data
    if valid_measurements == 0:
        return np.nan, 0, 0

    max_episode_duration = 0
    current_start = None
    previous_index = None

    for idx, value in zip(
        valid_indices,
        valid_values
    ):
        if value < threshold:

            if current_start is None:
                current_start = idx

            elif (
                previous_index is not None
                and idx - previous_index > max_gap
            ):
                current_start = idx

            previous_index = idx

            duration = idx - current_start

            if duration > max_episode_duration:
                max_episode_duration = duration

        else:
            current_start = None
            previous_index = None

    ih_target = int(
        max_episode_duration >= min_duration
    )

    return (
        ih_target,
        max_episode_duration,
        valid_measurements
    )


results = []

print("\nBUILDING IH TARGET FOR ALL CASES")
print("=" * 50)

for caseid in range(1, 6389):

    row = clinical_data.loc[
        clinical_data["caseid"] == caseid
    ]

    if row.empty:
        continue

    opstart = row["opstart"].iloc[0]
    opend = row["opend"].iloc[0]

    # Missing surgical timing
    if pd.isna(opstart) or pd.isna(opend):

        results.append({
            "caseid": caseid,
            "ih_target": np.nan,
            "longest_episode_sec": np.nan,
            "valid_map_measurements": 0
        })

        print(
            f"Processed case {caseid} "
            f"- missing opstart/opend"
        )

        continue

    data = vitaldb.load_case(
        caseid,
        [MAP_TRACK],
        interval=1
    )

    map_values = data[:, 0]

    (
        ih_target,
        longest_episode,
        valid_measurements
    ) = classify_hypotension(
        map_values,
        opstart,
        opend
    )

    results.append({
        "caseid": caseid,
        "ih_target": ih_target,
        "longest_episode_sec": longest_episode,
        "valid_map_measurements": valid_measurements
    })

    print(f"Processed case {caseid}")


target_df = pd.DataFrame(results)

target_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nTARGET SUMMARY")
print("=" * 50)

print(
    target_df["ih_target"]
    .value_counts(
        dropna=False
    )
)

valid_targets = target_df[
    "ih_target"
].dropna()

print(
    "\nPercentage IH = 1:",
    round(
        (valid_targets == 1).mean()
        * 100,
        1
    ),
    "%"
)

print(
    "Percentage IH = 0:",
    round(
        (valid_targets == 0).mean()
        * 100,
        1
    ),
    "%"
)

print(
    "\nSaved to:",
    OUTPUT_FILE
)