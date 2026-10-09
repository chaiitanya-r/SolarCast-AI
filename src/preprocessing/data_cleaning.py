"""
Data cleaning for the Solar Iridescence tabular dataset.

Input:  data/raw/SolarIridescenceDataset.csv
Output: data/processed/cleaned_solar_data.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.helpers import ensure_dirs, get_project_root, save_plot, timer

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
    df_before = df.copy()
    n_start = len(df)
    n_raw = len(df)

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
    print(
        f"  Cleaned: {n_raw:,} → {n_end:,} rows "
        f"(dup={n_dup_removed}, invalid={n_invalid_removed}, night={n_night_removed})"
    )

    df.to_csv(out_path, index=False)
    print(f"  Saved: cleaned_solar_data.csv")
    print(f"  Output: {len(df):,} rows × {df.columns.size} cols")

    try:
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].hist(df_before["Irradiance"].dropna(), bins=30, color="tab:blue", alpha=0.8)
        axes[0, 0].set_title("Irradiance Before Cleaning")
        axes[0, 0].set_xlabel("Irradiance")
        axes[0, 0].set_ylabel("Frequency")
        axes[1, 0].hist(df["Irradiance"].dropna(), bins=30, color="tab:green", alpha=0.8)
        axes[1, 0].set_title("Irradiance After Cleaning")
        axes[1, 0].set_xlabel("Irradiance")
        axes[1, 0].set_ylabel("Frequency")
        axes[0, 1].hist(df_before["Temperature"].dropna(), bins=30, color="tab:orange", alpha=0.8)
        axes[0, 1].set_title("Temperature Before Cleaning")
        axes[0, 1].set_xlabel("Temperature")
        axes[0, 1].set_ylabel("Frequency")
        axes[1, 1].hist(df["Temperature"].dropna(), bins=30, color="tab:red", alpha=0.8)
        axes[1, 1].set_title("Temperature After Cleaning")
        axes[1, 1].set_xlabel("Temperature")
        axes[1, 1].set_ylabel("Frequency")
        fig.suptitle("Data Distribution Before vs After Cleaning")
        plt.tight_layout()
        save_plot(fig, "cleaning_distribution.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate cleaning distribution plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.boxplot(y=df["Temperature"], ax=axes[0], color="tab:orange")
        axes[0].set_title("Temperature Outliers After Cleaning")
        axes[0].set_xlabel("Temperature")
        axes[0].set_ylabel("Temperature")
        sns.boxplot(y=df["Irradiance"], ax=axes[1], color="tab:blue")
        axes[1].set_title("Irradiance Outliers After Cleaning")
        axes[1].set_xlabel("Irradiance")
        axes[1].set_ylabel("Irradiance")
        plt.tight_layout()
        save_plot(fig, "cleaning_boxplots.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate cleaning boxplots: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        heat_df = (
            df.groupby(["Hour", "Month"], as_index=False)["Irradiance"]
            .mean()
            .pivot(index="Hour", columns="Month", values="Irradiance")
        )
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(heat_df, annot=False, cmap="YlOrRd", ax=ax)
        ax.set_title("Hourly Irradiance Heatmap by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour")
        plt.tight_layout()
        save_plot(fig, "irradiance_heatmap_hour_month.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate irradiance heatmap: {exc}")
    return df

def main() -> pd.DataFrame:
    return run_cleaning()

if __name__ == "__main__":
    main()
