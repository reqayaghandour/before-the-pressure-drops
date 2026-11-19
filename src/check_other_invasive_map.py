"""
Check (Section 4.2): do other invasive MAP tracks add cases without Solar8000/ART_MBP?
Run on a sample of the first 499 cases. Result: only one additional case, so the other
tracks were not used.

Usage (from the src folder):
    python check_other_invasive_map.py
"""

import vitaldb
import numpy as np

PRIMARY_TRACK = "Solar8000/ART_MBP"

OTHER_INVASIVE_MAP_TRACKS = [
    "Solar8000/FEM_MBP",
    "EV1000/ART_MBP"
]

print("\nCHECK OTHER INVASIVE MAP TRACKS")
print("=" * 50)

primary_available = 0
other_invasive_available = 0
still_missing = 0

for caseid in range(1, 500):

    primary_data = vitaldb.load_case(
        caseid,
        [PRIMARY_TRACK],
        interval=1
    )

    primary_map = primary_data[:, 0]

    primary_valid = primary_map[
        (~np.isnan(primary_map)) &
        (primary_map >= 20) &
        (primary_map <= 200)
    ]

    if len(primary_valid) > 0:
        primary_available += 1
        print(f"Processed case {caseid}")
        continue

    found_other_invasive = False

    for track in OTHER_INVASIVE_MAP_TRACKS:

        data = vitaldb.load_case(
            caseid,
            [track],
            interval=1
        )

        map_values = data[:, 0]

        valid_values = map_values[
            (~np.isnan(map_values)) &
            (map_values >= 20) &
            (map_values <= 200)
        ]

        if len(valid_values) > 0:
            found_other_invasive = True
            break

    if found_other_invasive:
        other_invasive_available += 1
    else:
        still_missing += 1

    print(f"Processed case {caseid}")


print("\nRESULT")
print("=" * 50)

print(
    "Cases with Solar8000/ART_MBP:",
    primary_available
)

print(
    "Additional cases with another invasive MAP track:",
    other_invasive_available
)

print(
    "Cases still without invasive MAP:",
    still_missing
)

total_invasive = (
    primary_available +
    other_invasive_available
)

print(
    "Total cases with usable invasive MAP:",
    total_invasive
)

print(
    "Percentage with usable invasive MAP:",
    round(total_invasive / 500 * 100, 1),
    "%"
)