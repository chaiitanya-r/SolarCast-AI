"""
Data cleaning for the Solar Iridescence tabular dataset.

Input:  data/raw/SolarIridescenceDataset.csv
Output: data/processed/cleaned_solar_data.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dirs, get_project_root, timer

np.random.seed(42)


def _clip_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Clip values to [Q1 - factor*IQR, Q3 + factor*IQR] without dropping rows."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


@timer
def run_cleaning() -> pd.DataFrame:
    """Execute the full cleaning pipeline and save processed CSV."""
    ensure_dirs()
    root = get_project_root()
    raw_path = root / "data" / "raw" / "SolarIridescenceDataset.csv"
    out_path = root / "data" / "processed" / "cleaned_solar_data.csv"

    df = pd.read_csv(raw_path)
    n_start = len(df)
    print(f"Loaded raw data: shape={df.shape}")
    print(df.head())

    df["datetime"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Hour"])
    )

    n_before_dup = len(df)
    df = df.drop_duplicates()
    n_dup_removed = n_before_dup - len(df)

    n_before_neg = len(df)
    df = df.dropna(subset=["Temperature"])
    df = df[df["Irradiance"] >= 0]
    n_invalid_removed = n_before_neg - len(df)

    n_before_night = len(df)
    df = df[(df["Hour"] >= 6) & (df["Hour"] <= 19)]
    n_night_removed = n_before_night - len(df)

    temp_clipped = (df["Temperature"] != _clip_iqr(df["Temperature"])).sum()
    irrad_clipped = (df["Irradiance"] != _clip_iqr(df["Irradiance"])).sum()
    df["Temperature"] = _clip_iqr(df["Temperature"])
    df["Irradiance"] = _clip_iqr(df["Irradiance"])

    n_end = len(df)
    print("\n=== Cleaning Summary ===")
    print(f"  Starting rows:           {n_start}")
    print(f"  Duplicates removed:      {n_dup_removed}")
    print(f"  Invalid/NaN removed:     {n_invalid_removed}")
    print(f"  Nighttime rows removed:  {n_night_removed}")
    print(f"  Temperature clipped:     {temp_clipped}")
    print(f"  Irradiance clipped:      {irrad_clipped}")
    print(f"  Final rows:              {n_end} (removed {n_start - n_end} total)")
    print(f"  Nulls remaining:         {df.isnull().sum().sum()}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved cleaned data to {out_path}")
    print(df.head())
    print(f"Shape: {df.shape}")
    return df


def main() -> pd.DataFrame:
    return run_cleaning()


if __name__ == "__main__":
    main()
