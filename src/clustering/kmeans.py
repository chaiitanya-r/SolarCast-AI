"""
KMeans clustering with elbow and silhouette analysis.

Input:  engineered solar data (Temperature, Irradiance, Hour)
Output: clustered_data.csv, elbow/silhouette/scatter plots
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, save_plot, timer

np.random.seed(42)
plt.style.use("seaborn-v0_8")

CLUSTER_FEATURES = ["Temperature", "Irradiance"]
EXTRA_3D = ["Hour", "Temperature", "Irradiance"]
MAX_ROWS = 15000


def _find_best_k(X: np.ndarray, k_range: range = range(2, 11)) -> int:
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    root = get_project_root()
    plots_dir = root / "results" / "plots"

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(list(k_range), inertias, "bo-")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans Elbow Method")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "kmeans_elbow.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(list(k_range), silhouettes, "go-")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("KMeans Silhouette Scores")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "kmeans_silhouette.png")
    plt.close(fig)

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    if silhouettes[int(np.argmax(silhouettes))] - silhouettes[2] < 0.02:
        best_k = 3
    return best_k


@timer
def run_kmeans(k: int | None = None) -> pd.DataFrame:
    """Fit KMeans, save cluster assignments and diagnostic plots."""
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"

    df = load_processed("engineered_solar_data")
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"  Sampled {MAX_ROWS} rows for speed")
    X = df[CLUSTER_FEATURES].values

    best_k = k or _find_best_k(X)
    print(f"  Selected k={best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        df["Temperature"],
        df["Irradiance"],
        c=df["kmeans_cluster"],
        cmap="viridis",
        alpha=0.4,
        s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Irradiance")
    ax.set_title(f"KMeans Clusters (k={best_k})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "kmeans_2d_scatter.png")
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.scatter(
        df["Hour"],
        df["Temperature"],
        df["Irradiance"],
        c=df["kmeans_cluster"],
        cmap="viridis",
        alpha=0.3,
        s=5,
    )
    ax3d.set_xlabel("Hour")
    ax3d.set_ylabel("Temperature")
    ax3d.set_zlabel("Irradiance")
    ax3d.set_title(f"KMeans 3D Clusters (k={best_k})")
    plt.tight_layout()
    save_plot(fig, "kmeans_3d_scatter.png")
    plt.close(fig)

    try:
        plt.style.use("seaborn-v0_8")
        radar_features = ["Temperature", "Irradiance", "hour_sin", "month_sin", "clearness_index"]
        available = [c for c in radar_features if c in df.columns]
        if len(available) >= 3:
            means = df.groupby("kmeans_cluster")[available].mean()
            vals = means.values
            vals_min = vals.min(axis=0)
            vals_max = vals.max(axis=0)
            denom = np.where((vals_max - vals_min) == 0, 1, (vals_max - vals_min))
            norm_vals = (vals - vals_min) / denom
            categories = list(means.columns)
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, polar=True)
            for cluster_id, row in zip(means.index, norm_vals):
                row_closed = np.concatenate([row, row[:1]])
                ax.plot(angles, row_closed, linewidth=2, label=f"Cluster {cluster_id}")
                ax.fill(angles, row_closed, alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title("KMeans Cluster Profile Radar (Normalized Means)")
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            plt.tight_layout()
            save_plot(fig, "kmeans_cluster_radar.png")
            plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate KMeans radar plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        counts = df["kmeans_cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.pie(
            counts.values,
            labels=[f"Cluster {c}" for c in counts.index],
            autopct="%1.1f%%",
            colors=plt.cm.tab10(np.linspace(0, 1, len(counts))),
            startangle=90,
        )
        ax.set_title("KMeans Cluster Size Distribution")
        plt.tight_layout()
        save_plot(fig, "kmeans_cluster_sizes.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate cluster size pie chart: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(12, 6))
        for cluster_id in sorted(df["kmeans_cluster"].dropna().unique()):
            sns.kdeplot(
                data=df[df["kmeans_cluster"] == cluster_id],
                x="Irradiance",
                fill=False,
                common_norm=False,
                linewidth=2,
                ax=ax,
                label=f"Cluster {int(cluster_id)}",
            )
        ax.set_title("Irradiance Distribution by KMeans Cluster")
        ax.set_xlabel("Irradiance")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, "kmeans_irradiance_kde.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate irradiance KDE plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        k_values = list(range(2, 11))
        sil_values = []
        for k_val in k_values:
            model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            sil_values.append(silhouette_score(X, labels))
        colors = ["tab:orange" if k_val == best_k else "tab:blue" for k_val in k_values]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(k_values, sil_values, color=colors)
        for container in [bars]:
            ax.bar_label(container, fmt="%.4f", padding=3)
        ax.set_title("Silhouette Scores for KMeans (k=2 to 10)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        plt.tight_layout()
        save_plot(fig, "kmeans_silhouette_bars.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate silhouette bar chart: {exc}")

    out_path = root / "data" / "processed" / "clustered_data.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: clustered_data.csv")
    print(f"  Output: {len(df):,} rows × {df.columns.size} cols")
    return df


def main() -> pd.DataFrame:
    return run_kmeans()


if __name__ == "__main__":
    main()
