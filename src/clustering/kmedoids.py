"""
kmedoids.py — KMedoids clustering using a pure scipy/numpy implementation.
No sklearn_extra dependency (ABI-incompatible on many setups).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.helpers import get_project_root, load_processed, save_plot, timer

np.random.seed(42)


def kmedoids_fit(X: np.ndarray, k: int = 3, max_iter: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    PAM-style KMedoids using scipy cdist for distance matrix.
    Returns (labels, medoid_indices).
    """
    n = len(X)
    medoid_idx = np.random.choice(n, k, replace=False)
    dist_matrix = cdist(X, X, metric="euclidean")

    for _ in range(max_iter):
        labels = np.argmin(dist_matrix[:, medoid_idx], axis=1)
        new_medoids = medoid_idx.copy()
        for c in range(k):
            cluster_pts = np.where(labels == c)[0]
            if len(cluster_pts) == 0:
                continue
            sub = dist_matrix[np.ix_(cluster_pts, cluster_pts)]
            new_medoids[c] = cluster_pts[np.argmin(sub.sum(axis=1))]

        if np.array_equal(sorted(new_medoids), sorted(medoid_idx)):
            break
        medoid_idx = new_medoids

    labels = np.argmin(dist_matrix[:, medoid_idx], axis=1)
    return labels, medoid_idx


@timer
def run_kmedoids() -> pd.DataFrame:
    df = load_processed("engineered_solar_data.csv")
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    X = sample[["Temperature", "Irradiance"]].values

    km_labels, km_medoids = kmedoids_fit(X, k=3)
    medoid_centers = X[km_medoids]

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kms_labels = kmeans.fit_predict(X)
    kms_centers = kmeans.cluster_centers_

    print("  KMedoids vs KMeans centers compared (k=3)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use("seaborn-v0_8")
    for ax, labels, centers, title in [
        (axes[0], km_labels, medoid_centers, "KMedoids (k=3)"),
        (axes[1], kms_labels, kms_centers, "KMeans (k=3)"),
    ]:
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.5, s=10)
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, zorder=5, label="Centers")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Irradiance (W/m²)")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    save_plot(fig, "kmedoids_comparison.png")
    plt.close(fig)

    try:
        plt.style.use("seaborn-v0_8")
        order = np.argsort(medoid_centers[:, 0])
        kms_order = np.argsort(kms_centers[:, 0])
        med = medoid_centers[order]
        kmn = kms_centers[kms_order]
        clusters = np.arange(3)
        width = 0.2
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, idx, feat in [(axes[0], 0, "Temperature"), (axes[1], 1, "Irradiance")]:
            bars1 = ax.bar(clusters - width / 2, med[:, idx], width=width, color="tab:blue", label="KMedoids")
            bars2 = ax.bar(clusters + width / 2, kmn[:, idx], width=width, color="tab:orange", label="KMeans")
            ax.bar_label(bars1, fmt="%.4f", padding=3)
            ax.bar_label(bars2, fmt="%.4f", padding=3)
            ax.set_title(f"{feat} Centers by Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel(feat)
            ax.set_xticks(clusters)
            ax.set_xticklabels([f"C{i}" for i in clusters])
            ax.legend()
        fig.suptitle("KMedoids vs KMeans Cluster Center Comparison")
        plt.tight_layout()
        save_plot(fig, "kmedoids_center_comparison.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate center comparison chart: {exc}")
    print("  Saved: kmedoids_comparison.png")
    return sample


if __name__ == "__main__":
    run_kmedoids()
