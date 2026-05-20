"""
Feature engineering for cleaned solar irradiance data.

Input:  data/processed/cleaned_solar_data.csv
Output: data/processed/engineered_solar_data.csv

Leakage controls: features that use irradiance history use only *past* values
relative to each row (lags, shifted rolling means, expanding max). Labels
(irradiance_class) still reflect the current hour's irradiance for supervised
learning — models must not see current irradiance in X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, save_plot, timer

np.random.seed(42)

FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "Temperature",
    "lag_1h_irradiance",
    "rolling_mean_3h",
    "clearness_index",
]
# Leakage note: rolling_mean_3h and clearness_index MUST use only past irradiance
# relative to the row timestamp. The label irradiance_class uses *current* Irradiance.


def _solar_elevation_approx(hour: pd.Series) -> pd.Series:
    """Approximate solar elevation factor from hour (0 at night edges, 1 at noon)."""
    return np.sin(np.pi * (hour - 6) / 13).clip(lower=0.05)


@timer
def run_feature_engineering() -> pd.DataFrame:
    """Create cyclical, lag, rolling, clearness, DGR, and class features."""
    ensure_dirs()
    root = get_project_root()
    out_path = root / "data" / "processed" / "engineered_solar_data.csv"

    df = load_processed("cleaned_solar_data")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["lag_1h_irradiance"] = df["Irradiance"].shift(1)
    df["lag_2h_irradiance"] = df["Irradiance"].shift(2)
    # Past-only: mean of hours t-1, t-2, t-3 (no current-hour irradiance)
    df["rolling_mean_3h"] = (
        df["Irradiance"].shift(1).rolling(window=3, min_periods=1).mean()
    )

    elevation = _solar_elevation_approx(df["Hour"])
    # Past-only clearness: use lagged irradiance and max irradiance seen *before* this row
    max_irr_past = df["Irradiance"].expanding().max().shift(1).clip(lower=1.0)
    df["clearness_index"] = df["lag_1h_irradiance"] / (max_irr_past * elevation + 1e-6)

    df["date"] = df["datetime"].dt.date
    # Past-only within-day cumulative mean (excludes current hour; avoids future-day leak)
    df["DGR"] = df.groupby("date", group_keys=False)["Irradiance"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df["irradiance_class"] = pd.cut(
        df["Irradiance"],
        bins=[-np.inf, 200, 600, np.inf],
        labels=[0, 1, 2],
    ).astype(int)
    class_map = {0: "Low", 1: "Medium", 2: "High"}
    df["irradiance_class_label"] = df["irradiance_class"].map(class_map)

    df = df.dropna(subset=["lag_2h_irradiance"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"  Saved: engineered_solar_data.csv")
    print(f"  Output: {len(df):,} rows × {df.columns.size} cols")

    try:
        plt.style.use("seaborn-v0_8")
        class_counts = (
            df["irradiance_class_label"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(class_counts.index, class_counts.values, color=["tab:blue", "tab:orange", "tab:green"])
        for container in [bars]:
            ax.bar_label(container, fmt="%.4f", padding=3)
        ax.set_title("Irradiance Class Distribution")
        ax.set_xlabel("Irradiance Class")
        ax.set_ylabel("Count")
        plt.tight_layout()
        save_plot(fig, "class_distribution.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate class distribution plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc1 = axes[0].scatter(df["hour_sin"], df["hour_cos"], c=df["Hour"], cmap="viridis", alpha=0.6, s=12)
        axes[0].set_title("Hour Cyclical Encoding")
        axes[0].set_xlabel("hour_sin")
        axes[0].set_ylabel("hour_cos")
        axes[0].grid(True, alpha=0.3)
        fig.colorbar(sc1, ax=axes[0], label="Hour")
        sc2 = axes[1].scatter(df["month_sin"], df["month_cos"], c=df["Month"], cmap="viridis", alpha=0.6, s=12)
        axes[1].set_title("Month Cyclical Encoding")
        axes[1].set_xlabel("month_sin")
        axes[1].set_ylabel("month_cos")
        axes[1].grid(True, alpha=0.3)
        fig.colorbar(sc2, ax=axes[1], label="Month")
        plt.tight_layout()
        save_plot(fig, "cyclical_features.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate cyclical feature plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        numeric_df = df.select_dtypes(include=["number"])
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Engineered Feature Correlation Heatmap")
        ax.set_xlabel("Features")
        ax.set_ylabel("Features")
        plt.tight_layout()
        save_plot(fig, "feature_correlation_heatmap.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate feature correlation heatmap: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        preview = df.head(min(200, len(df)))
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(preview.index, preview["Irradiance"], label="Irradiance")
        ax.plot(preview.index, preview["lag_1h_irradiance"], label="lag_1h_irradiance")
        ax.plot(preview.index, preview["lag_2h_irradiance"], label="lag_2h_irradiance")
        ax.plot(preview.index, preview["rolling_mean_3h"], label="rolling_mean_3h")
        ax.set_title("Lag Features Preview (First 200 Rows)")
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Irradiance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "lag_features_preview.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate lag features preview: {exc}")
    return df


def main() -> pd.DataFrame:
    return run_feature_engineering()


if __name__ == "__main__":
    main()
