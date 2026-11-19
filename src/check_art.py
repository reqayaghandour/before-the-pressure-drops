"""
Check (Section 4.3): how many VitalDB cases have a usable invasive MAP (Solar8000/ART_MBP)?

Usage (from the src folder; downloads the MAP track of every case):
    python check_art.py
"""

import vitaldb
import numpy as np

MAP_TRACK = "Solar8000/ART_MBP"

print("\nCHECK ART_MBP AVAILABILITY - ALL CASES")
print("=" * 50)

art_available = 0
art_missing = 0

for caseid in range(1, 6389):

    data = vitaldb.load_case(
        caseid,
        [MAP_TRACK],
        interval=1
    )

    map_values = data[:, 0]

    clean_map = map_values.copy()

    clean_map[
        (clean_map < 20) |
        (clean_map > 200)
    ] = np.nan

    valid_measurements = np.sum(
        ~np.isnan(clean_map)
    )

    if valid_measurements > 0:
        art_available += 1
    else:
        art_missing += 1

    print(f"Processed case {caseid}")

print("\nRESULT")
print("=" * 50)

print("Cases with ART_MBP:", art_available)
print("Cases without ART_MBP:", art_missing)
print("Total cases checked:", art_available + art_missing)

print(
    "Percentage with ART_MBP:",
    round(art_available / 6388 * 100, 1),
    "%"
)