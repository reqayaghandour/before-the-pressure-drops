"""
Data loading and preprocessing for IH prediction (VitalDB; prediction time point = surgical start).

Two feature sets:
  - "primary":  preoperative information (27 features)
  - "extended": primary + invasive arterial pressure in the last 5 minutes before surgical
                start (mean MAP, lowest 60-s rolling mean, % of time < 65 mmHg).
                These come from data/processed/preopstart_target_check.csv
                (created by check_preopstart_and_target.py).

Usage in the model scripts:

    from preprocessing import load_data, get_X_y_groups, build_preprocessor

    df = load_data()
    X, y, groups = get_X_y_groups(df, feature_set="extended")
    model = Pipeline([("prep", build_preprocessor("extended")), ("clf", ...)])
    # cross-validation with StratifiedGroupKFold(...).split(X, y, groups)

The preprocessor is NOT fitted here: it is fitted per fold inside the pipeline, so that
imputation, scaling and the merging of rare categories are learned from training data only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "analysis_dataset.csv"
PREOPSTART_FILE = BASE_DIR / "data" / "processed" / "preopstart_target_check.csv"

# Only these three columns are taken from PREOPSTART_FILE. The other columns in that file
# (episodes, recomputed target, ...) describe the period AFTER surgical start and must
# never be used as features.
PREOPSTART_COLUMNS = {
    "w5_map_mean": "preop5_map_mean",          # mean MAP, last 5 min before surgical start
    "w5_map_min_roll60": "preop5_map_min60",   # lowest 60-s rolling mean MAP in that window
    "w5_pct_below65": "preop5_pct_below65",    # % of valid measurements < 65 mmHg in that window
}

TARGET = "ih_target"
GROUP = "subjectid"  # patient ID: needed for patient-grouped folds

FEATURE_SETS = ("primary", "extended")

# ----------------------------
# Primary: preoperative
# ----------------------------

PRIMARY_NUMERIC_FEATURES = [
    "age",
    "height",
    "weight",
    "bmi",
    "preop_hb",
    "preop_plt",
    "preop_pt",
    "preop_aptt",
    "preop_na",
    "preop_k",
    "preop_gluc",
    "preop_alb",
    "preop_ast",
    "preop_alt",
    "preop_bun",
    "preop_cr",
]

PRIMARY_CATEGORICAL_FEATURES = [
    "sex",
    "emop",
    "department",
    "optype",
    "approach",
    "position",
    "preop_htn",
    "preop_dm",
    "preop_ecg_grouped",
    "preop_pft",
]

ORDINAL_FEATURES = ["asa"]

# ----------------------------
# Extended: additional information available at surgical start
# ----------------------------

# Arterial MAP in the last 5 minutes before surgical start. About 1-2% of the cases have
# too little data; these receive median imputation plus a "was missing" indicator.
EXTENDED_EXTRA_NUMERIC_FEATURES = list(PREOPSTART_COLUMNS.values())

ALL_CATEGORICAL_FEATURES = PRIMARY_CATEGORICAL_FEATURES

# Categories occurring in fewer than 1% of the training rows (per fold) are merged
# into a single 'infrequent' column.
MIN_CATEGORY_FREQUENCY = 0.01

# Inclusion: patients aged 18 years or older only.
MIN_AGE = 18


def get_feature_lists(feature_set="primary"):
    """Return (numeric, numeric-with-indicator, categorical, ordinal) feature lists."""
    if feature_set == "primary":
        return (
            PRIMARY_NUMERIC_FEATURES,
            [],
            PRIMARY_CATEGORICAL_FEATURES,
            ORDINAL_FEATURES,
        )
    if feature_set == "extended":
        return (
            PRIMARY_NUMERIC_FEATURES,
            EXTENDED_EXTRA_NUMERIC_FEATURES,
            PRIMARY_CATEGORICAL_FEATURES,
            ORDINAL_FEATURES,
        )
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}, not {feature_set!r}")


def get_all_features(feature_set="primary"):
    numeric, numeric_ind, categorical, ordinal = get_feature_lists(feature_set)
    return numeric + numeric_ind + categorical + ordinal


def load_data(path=DATA_FILE):
    """Read the dataset and apply only deterministic, row-by-row operations
    (no statistics over the whole dataset, so no leakage)."""
    df = pd.read_csv(path)

    # Empty strings count as missing.
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Age: '>89' -> 90; anything that is still not a number is a data error.
    age_raw = df["age"].replace(">89", "90")
    age_num = pd.to_numeric(age_raw, errors="coerce")
    bad_age = age_raw.notna() & age_num.isna()
    if bad_age.any():
        raise ValueError(
            f"Unexpected values in 'age': {age_raw[bad_age].unique()[:10]}"
        )
    df["age"] = age_num

    # Inclusion: adults only. The IH threshold of MAP < 65 mmHg is derived for adults;
    # children have a physiologically lower blood pressure.
    too_young = df["age"] < MIN_AGE
    if too_young.any():
        print(f"[load_data] {int(too_young.sum())} cases younger than {MIN_AGE} years excluded "
              f"({int((~too_young).sum())} cases remaining).")
    df = df[~too_young].reset_index(drop=True)

    # Exclusion: ASA 6 = brain-dead organ donors. Their blood pressure is determined by
    # donor management rather than by the physiology studied here.
    donors = pd.to_numeric(df["asa"], errors="coerce") == 6
    if donors.any():
        print(f"[load_data] {int(donors.sum())} cases with ASA 6 (organ donors) excluded "
              f"({int((~donors).sum())} cases remaining).")
    df = df[~donors].reset_index(drop=True)

    # Group the ECG into normal/abnormal. Missing stays missing (otherwise NaN would become 'Abnormal').
    ecg = df["preop_ecg"]
    df["preop_ecg_grouped"] = np.nan
    df["preop_ecg_grouped"] = df["preop_ecg_grouped"].astype(object)
    df.loc[ecg.notna(), "preop_ecg_grouped"] = np.where(
        ecg[ecg.notna()] == "Normal Sinus Rhythm", "Normal", "Abnormal"
    )

    # Add the pre-start MAP features (only the three allowed columns).
    if PREOPSTART_FILE.exists():
        preop = pd.read_csv(PREOPSTART_FILE, usecols=["caseid", *PREOPSTART_COLUMNS])
        preop = preop.drop_duplicates("caseid").rename(columns=PREOPSTART_COLUMNS)
        n_before = len(df)
        df = df.merge(preop, on="caseid", how="left", validate="one_to_one")
        if len(df) != n_before:
            raise ValueError("Number of rows changed when merging the pre-start features.")

    # All categorical columns as text; NaN stays NaN, so that the imputer can turn it
    # into 'Missing' without type conflicts.
    for col in ALL_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
            df[col] = df[col].astype(object)

    return df


def get_X_y_groups(df, feature_set="primary"):
    """Split into features, target and patient groups."""
    all_features = get_all_features(feature_set)

    missing_cols = [c for c in all_features + [TARGET, GROUP] if c not in df.columns]
    if missing_cols:
        hint = ""
        if any(c in EXTENDED_EXTRA_NUMERIC_FEATURES for c in missing_cols):
            hint = (f"\nThe extended features come from {PREOPSTART_FILE}; "
                    "run check_preopstart_and_target.py first.")
        raise KeyError(f"Columns missing from the dataset: {missing_cols}{hint}")

    if df[TARGET].isna().any():
        raise ValueError(f"{df[TARGET].isna().sum()} rows without a target")

    X = df[all_features]
    y = df[TARGET].astype(int)
    groups = df[GROUP]
    return X, y, groups


def build_preprocessor(feature_set="primary"):
    """Return a new, unfitted preprocessor (the same for LR, RF, XGBoost and the NN)."""
    numeric, numeric_ind, categorical, ordinal = get_feature_lists(feature_set)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_with_indicator_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),  # harmless for trees, needed for LR and the NN
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric)]
    if numeric_ind:
        transformers.append(("numeric_ind", numeric_with_indicator_pipeline, numeric_ind))
    transformers += [
        ("categorical", categorical_pipeline, categorical),
        ("ordinal", ordinal_pipeline, ordinal),
    ]

    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)


if __name__ == "__main__":
    # Smoke test only: fitting on all data is fine here to check shapes,
    # but must NOT be done in the model scripts.
    df = load_data()

    for feature_set in FEATURE_SETS:
        X, y, groups = get_X_y_groups(df, feature_set=feature_set)

        print(f"\nPREPROCESSING TEST — FEATURE SET: {feature_set.upper()}")
        print("=" * 50)
        print("Rows (cases):", len(df))
        print("Unique patients:", groups.nunique())
        print("IH prevalence:", round(y.mean(), 3))

        print("\nMissing per feature (%):")
        print((X.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())

        prep = build_preprocessor(feature_set)
        X_processed = prep.fit_transform(X)
        print("\nOriginal shape:", X.shape)
        print("Processed shape:", X_processed.shape)
        print("\nFeature names after encoding:")
        print(list(prep.get_feature_names_out()))

    print("\nDISTRIBUTION OF THE EXTENDED FEATURES")
    print("=" * 50)
    print(df[EXTENDED_EXTRA_NUMERIC_FEATURES].describe().round(1).to_string())
    print("\nMissing (%):")
    print((df[EXTENDED_EXTRA_NUMERIC_FEATURES].isna().mean() * 100).round(1).to_string())

    print("\nAGE")
    print("=" * 50)
    print(df["age"].describe().round(1).to_string())
    print("Number of cases < 18 years:", int((df["age"] < 18).sum()))
