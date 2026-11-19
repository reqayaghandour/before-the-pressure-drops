"""
One run over all cases with a valid IH target. For each case, Solar8000/ART_MBP is downloaded
once and used for three things:

  A. PRE-START FEATURES AND COVERAGE
     How many valid arterial MAP measurements are available in the last 5, 10 and 15 minutes
     before surgical start, and the MAP summaries per window: mean, lowest 60-second rolling
     mean and % of time below 65 mmHg. The 5-minute summaries are the pre-start MAP features
     of the extended feature set (see preprocessing.py).

  B. ARTEFACTS IN THE TARGET
     The same episode rules as build_ih_target.py (MAP < 65 for >= 60 s, 20-200 mmHg,
     gaps <= 4 s). For each IH episode, the median MAP is computed. An episode with a median
     below 40 mmHg is suspicious (likely an artefact such as flushing, zeroing or blood
     sampling). The script counts how many IH cases have ONLY such suspicious episodes, and
     also recomputes the target to check that it reproduces ih_target.csv.

  C. ALREADY HYPOTENSIVE AT SURGICAL START
     How many IH cases have their first episode within the first 5 minutes after surgical
     start, and how many of these already had a mean MAP < 65 in the last minute before start.

Output: data/processed/preopstart_target_check.csv (saved periodically; after a crash,
rerunning the script continues where it stopped).
"""

import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

# =========================================================
# Settings
# =========================================================

MAP_TRACK = "Solar8000/ART_MBP"

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLINICAL_FILE = RAW_DIR / "clinical_data.csv"
TARGET_FILE = PROCESSED_DIR / "ih_target.csv"
OUTPUT_FILE = PROCESSED_DIR / "preopstart_target_check.csv"

# Same rules as build_ih_target.py
THRESHOLD = 65
MIN_DURATION = 60
MIN_MAP = 20
MAX_MAP = 200
MAX_GAP = 4

# Checks
WINDOWS_MIN = [5, 10, 15]
EXPECTED_INTERVAL_SEC = 2      # ART_MBP is usually recorded every 2 s
SUSPICIOUS_MEDIAN = 40         # episode median below 40 mmHg = suspicious
EARLY_EPISODE_SEC = 5 * 60     # "early" episode = starts within 5 min after surgical start
ROLLING_SEC = 60
ROLLING_MIN_VALID = 20         # at least 20 valid measurements (~40 s) per 60 s

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5
SAVE_EVERY = 25

# Without a time-out, a single download can hang indefinitely.
socket.setdefaulttimeout(60)


# =========================================================
# Helpers
# =========================================================

def load_case_with_retry(caseid):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return vitaldb.load_case(caseid, [MAP_TRACK], interval=1)
        except Exception as e:  # network errors, 502, time-outs
            print(f"Download error case {caseid}: {e} (attempt {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)
    return None


def clean(values):
    values = values.astype(float).copy()
    values[(values < MIN_MAP) | (values > MAX_MAP)] = np.nan
    return values


def find_episodes(surgical_map):
    """All episodes of MAP < 65, with the same logic as classify_hypotension() in
    build_ih_target.py. Returns a list of (start, end) in seconds from surgical start."""
    valid_idx = np.where(~np.isnan(surgical_map))[0]
    episodes = []
    cur_start, prev = None, None
    for idx in valid_idx:
        if surgical_map[idx] < THRESHOLD:
            if cur_start is None:
                cur_start = idx
            elif idx - prev > MAX_GAP:
                episodes.append((cur_start, prev))
                cur_start = idx
            prev = idx
        else:
            if cur_start is not None:
                episodes.append((cur_start, prev))
            cur_start, prev = None, None
    if cur_start is not None:
        episodes.append((cur_start, prev))
    return episodes


def window_features(map_values, opstart, minutes):
    """Coverage and MAP summary in [opstart - minutes, opstart)."""
    start = max(0, opstart - minutes * 60)
    window = clean(map_values[start:opstart]) if opstart > 0 else np.array([])
    expected = minutes * 60 / EXPECTED_INTERVAL_SEC
    n_valid = int(np.sum(~np.isnan(window))) if len(window) else 0

    out = {
        f"w{minutes}_n_valid": n_valid,
        f"w{minutes}_coverage": min(n_valid / expected, 1.0),
        f"w{minutes}_map_mean": np.nan,
        f"w{minutes}_map_min_roll60": np.nan,
        f"w{minutes}_pct_below65": np.nan,
    }
    if n_valid == 0:
        return out

    valid = window[~np.isnan(window)]
    roll = (
        pd.Series(window)
        .rolling(ROLLING_SEC, min_periods=ROLLING_MIN_VALID)
        .mean()
    )
    out[f"w{minutes}_map_mean"] = float(np.mean(valid))
    out[f"w{minutes}_map_min_roll60"] = float(roll.min()) if roll.notna().any() else np.nan
    out[f"w{minutes}_pct_below65"] = float(np.mean(valid < THRESHOLD) * 100)
    return out


def analyse_case(caseid, map_values, opstart, opend, ih_target_file):
    opstart, opend = int(opstart), int(opend)
    row = {"caseid": caseid, "ih_target_file": ih_target_file}

    # --- A. coverage and features before surgical start ---
    for w in WINDOWS_MIN:
        row.update(window_features(map_values, opstart, w))

    # --- B + C. episodes during surgery ---
    surgical = clean(map_values[opstart:opend])
    episodes = find_episodes(surgical)
    qualifying = [(s, e) for s, e in episodes if e - s >= MIN_DURATION]

    medians = [float(np.nanmedian(surgical[s:e + 1])) for s, e in qualifying]
    n_suspicious = sum(m < SUSPICIOUS_MEDIAN for m in medians)

    row["ih_recomputed"] = int(len(qualifying) > 0) if np.any(~np.isnan(surgical)) else np.nan
    row["n_ih_episodes"] = len(qualifying)
    row["n_suspicious_episodes"] = n_suspicious
    row["all_episodes_suspicious"] = int(len(qualifying) > 0 and n_suspicious == len(qualifying))
    row["min_episode_median"] = min(medians) if medians else np.nan
    row["first_episode_start_sec"] = qualifying[0][0] if qualifying else np.nan

    # last minute before surgical start
    last_min = clean(map_values[max(0, opstart - 60):opstart]) if opstart > 0 else np.array([])
    row["last60_map_mean"] = float(np.nanmean(last_min)) if np.any(~np.isnan(last_min)) else np.nan
    return row


# =========================================================
# Data
# =========================================================

clinical = pd.read_csv(CLINICAL_FILE)
target = pd.read_csv(TARGET_FILE)
target = target[target["ih_target"].notna()]

cases = clinical.merge(target[["caseid", "ih_target"]], on="caseid", how="inner")
cases = cases[cases["opstart"].notna() & cases["opend"].notna()]

results, done = [], set()
if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE)
    if not existing.empty:
        results = existing.to_dict("records")
        done = set(existing["caseid"].astype(int))
        print(f"Existing output found: {len(done)} cases already processed.")

# =========================================================
# Main loop
# =========================================================

print("\nCHECK PRE-OPSTART COVERAGE + TARGET")
print("=" * 50)

failed, new = [], 0
for _, r in cases.iterrows():
    caseid = int(r["caseid"])
    if caseid in done:
        continue

    data = load_case_with_retry(caseid)
    if data is None or data.ndim != 2 or data.shape[1] == 0:
        failed.append(caseid)
        continue

    results.append(analyse_case(caseid, data[:, 0], r["opstart"], r["opend"], int(r["ih_target"])))
    done.add(caseid)
    new += 1

    if new % SAVE_EVERY == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print(f"Processed in this run: {new} | total saved: {len(results)}")

df = pd.DataFrame(results).drop_duplicates("caseid", keep="last").sort_values("caseid")
df.to_csv(OUTPUT_FILE, index=False)

# =========================================================
# Summary
# =========================================================

n = len(df)
ih = df[df["ih_target_file"] == 1]

print("\n" + "=" * 60)
print(f"SUMMARY ({n} cases, {len(ih)} with IH = 1)")
print("=" * 60)

print("\nSANITY CHECK - does the recomputed target match ih_target.csv?")
match = (df["ih_recomputed"] == df["ih_target_file"]).mean() * 100
print(f"  Agreement: {match:.1f}% (should be 100%)")

print("\nA. COVERAGE BEFORE SURGICAL START")
print(f"  (coverage = valid measurements / expected measurements at 1 per {EXPECTED_INTERVAL_SEC} s)")
for w in WINDOWS_MIN:
    cov = df[f"w{w}_coverage"]
    print(
        f"  Last {w:>2} min:  "
        f"any data {np.mean(cov > 0) * 100:5.1f}%  |  "
        f">= 50% {np.mean(cov >= 0.5) * 100:5.1f}%  |  "
        f">= 80% {np.mean(cov >= 0.8) * 100:5.1f}%  |  "
        f"median coverage {cov.median() * 100:5.1f}%"
    )
for w in WINDOWS_MIN:
    print(f"\n  MAP summary, last {w} min (cases with data):")
    cols = [f"w{w}_map_mean", f"w{w}_map_min_roll60", f"w{w}_pct_below65"]
    print(df[cols].describe().round(1).to_string())

print("\nB. ARTEFACTS IN THE TARGET")
print(f"  IH = 1 cases with at least one suspicious episode (median < {SUSPICIOUS_MEDIAN}): "
      f"{int((ih['n_suspicious_episodes'] > 0).sum())} "
      f"({(ih['n_suspicious_episodes'] > 0).mean() * 100:.1f}%)")
print(f"  IH = 1 cases with ONLY suspicious episodes:                  "
      f"{int(ih['all_episodes_suspicious'].sum())} "
      f"({ih['all_episodes_suspicious'].mean() * 100:.1f}%)")
print("  -> the last number matters most: without these episodes the case would be IH = 0.")
print("\n  Distribution of the lowest episode median for IH = 1 (mmHg):")
print(ih["min_episode_median"].describe().round(1).to_string())

print("\nC. ALREADY HYPOTENSIVE AT SURGICAL START")
early = ih["first_episode_start_sec"] < EARLY_EPISODE_SEC
early_and_low = early & (ih["last60_map_mean"] < THRESHOLD)
print(f"  IH = 1 cases with the first episode in the first 5 min after start: "
      f"{int(early.sum())} ({early.mean() * 100:.1f}%)")
print(f"  ...of which MAP in the last minute before start was already < 65:   "
      f"{int(early_and_low.sum())} ({early_and_low.mean() * 100:.1f}% of all IH = 1)")

print(f"\nFailed downloads in this run: {len(failed)}")
if failed:
    print(failed[:50])
print("\nSaved in:", OUTPUT_FILE)
