"""

Feature scaling and train/test splitting.

Fits StandardScaler on features and MinMaxScaler on Irradiance target **on the

training split only** (no test-set leakage into scaler statistics).

Supports chronological hold-out for time-series honesty (recommended for solar).

"""

from __future__ import annotations

import json

import joblib

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.feature_engineering import FEATURE_COLS

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, timer

np.random.seed(42)

FEATURE_COLUMNS = FEATURE_COLS

TARGET_COL = "Irradiance"

CLASS_COL = "irradiance_class"

USE_CHRONOLOGICAL_SPLIT = True

TEST_SIZE = 0.2

def fit_scalers(

    X_train: np.ndarray, y_train: np.ndarray

) -> tuple[StandardScaler, MinMaxScaler, np.ndarray, np.ndarray, list[str]]:

    """Fit scalers on training arrays only; return scaled X_train, y_train."""

    feature_scaler = StandardScaler()

    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)

    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    return feature_scaler, target_scaler, X_train_scaled, y_train_scaled, FEATURE_COLUMNS

def load_scalers() -> tuple[StandardScaler, MinMaxScaler]:

    """Load previously saved feature and target scalers."""

    root = get_project_root()

    models_dir = root / "results" / "models"

    feature_scaler = joblib.load(models_dir / "feature_scaler.joblib")

    target_scaler = joblib.load(models_dir / "target_scaler.joblib")

    return feature_scaler, target_scaler

def get_train_test_split(

    X: np.ndarray,

    y: np.ndarray,

    y_class: np.ndarray | None = None,

    test_size: float = 0.2,

    stratify: bool = True,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """Return train/test split; stratified on class labels when provided."""

    strat = y_class if stratify and y_class is not None else None

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)

@timer

def run_scaling() -> dict:

    """Fit scalers on train only, split data, persist arrays for downstream modules."""

    ensure_dirs()

    root = get_project_root()

    df = load_processed("engineered_solar_data")

    df = df.sort_values("datetime").reset_index(drop=True)

    X_raw = df[FEATURE_COLUMNS].values.astype(float)

    y_raw = df[TARGET_COL].values.astype(float).reshape(-1, 1)

    y_class = df[CLASS_COL].values

    if USE_CHRONOLOGICAL_SPLIT:

        n = len(df)

        split_idx = max(1, int(n * (1 - TEST_SIZE)))

        X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]

        y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

        y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]

        print(
            f"  Split: chronological  train={len(X_train_raw):,}  test={len(X_test_raw):,}"
        )

    else:

        X_train_raw, X_test_raw, y_train_raw, y_test_raw, y_class_train, y_class_test = (

            train_test_split(

                X_raw,

                y_raw,

                y_class,

                test_size=TEST_SIZE,

                random_state=42,

                stratify=y_class,

            )

        )

        print(
            f"  Split: stratified  train={len(X_train_raw):,}  test={len(X_test_raw):,}"
        )

    models_dir = root / "results" / "models"

    feature_scaler, target_scaler, X_train, y_train, feature_names = fit_scalers(

        X_train_raw, y_train_raw.ravel()

    )

    X_test = feature_scaler.transform(X_test_raw)

    y_test = target_scaler.transform(y_test_raw).ravel()

    joblib.dump(feature_scaler, models_dir / "feature_scaler.joblib")

    joblib.dump(target_scaler, models_dir / "target_scaler.joblib")

    processed_dir = root / "data" / "processed"

    np.save(processed_dir / "X_train.npy", X_train)

    np.save(processed_dir / "X_test.npy", X_test)

    np.save(processed_dir / "y_train.npy", y_train)

    np.save(processed_dir / "y_test.npy", y_test)

    np.save(processed_dir / "y_class_train.npy", y_class_train)

    np.save(processed_dir / "y_class_test.npy", y_class_test)

    meta = {

        "feature_names": feature_names,

        "n_train": int(len(X_train)),

        "n_test": int(len(X_test)),

        "chronological_split": USE_CHRONOLOGICAL_SPLIT,

        "test_size": TEST_SIZE,

    }

    meta_path = root / "data" / "processed" / "scaling_meta.json"

    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  Scaled {len(feature_names)} features  train={X_train.shape[0]:,}  test={X_test.shape[0]:,}")

    return {

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "y_class_train": y_class_train,

        "y_class_test": y_class_test,

        "feature_names": feature_names,

        "df": df,

    }

def load_split_data() -> dict:

    """Load persisted train/test arrays and metadata."""

    root = get_project_root()

    processed = root / "data" / "processed"

    meta = json.loads((processed / "scaling_meta.json").read_text())

    return {

        "X_train": np.load(processed / "X_train.npy"),

        "X_test": np.load(processed / "X_test.npy"),

        "y_train": np.load(processed / "y_train.npy"),

        "y_test": np.load(processed / "y_test.npy"),

        "y_class_train": np.load(processed / "y_class_train.npy"),

        "y_class_test": np.load(processed / "y_class_test.npy"),

        "feature_names": meta["feature_names"],

    }

def inverse_transform_target(y_scaled: np.ndarray) -> np.ndarray:

    """Convert scaled target back to original irradiance units."""

    _, target_scaler = load_scalers()

    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

def main() -> dict:

    return run_scaling()

if __name__ == "__main__":

    main()
