"""
Gaussian Mixture Model clustering with BIC/AIC model selection.

Fits GMM with n_components=3 and plots BIC/AIC for n=2..8.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.utils.helpers import ensure_dirs, get_project_root, load_processed, save_plot, timer

np.random.seed(42)
plt.style.use("seaborn-v0_8")

FEATURES = ["Temperature", "Irradiance"]
N_COMPONENTS = 3
MAX_ROWS = 12000


@timer
def run_gmm() -> pd.DataFrame:
    """Fit GMM, plot BIC/AIC curves, scatter clusters, print covariances."""
    ensure_dirs()
    root = get_project_root()
    plots_dir = root / "results" / "plots"

    df = load_processed("engineered_solar_data")
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"  Sampled {MAX_ROWS} rows for speed")
    X = df[FEATURES].values

    n_range = range(2, 9)
    bics, aics = [], []
    for n in n_range:
        gmm_tmp = GaussianMixture(n_components=n, random_state=42)
        gmm_tmp.fit(X)
        bics.append(gmm_tmp.bic(X))
        aics.append(gmm_tmp.aic(X))

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(list(n_range), bics, "b-o", label="BIC")
    ax.plot(list(n_range), aics, "r-o", label="AIC")
    ax.axvline(N_COMPONENTS, color="black", linestyle="--", linewidth=1.5, label=f"Selected k={N_COMPONENTS}")
    ax.set_xlabel("n_components")
    ax.set_ylabel("Score")
    ax.set_title("GMM Model Selection — BIC / AIC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "gmm_bic_aic.png")
    plt.close(fig)

    gmm = GaussianMixture(n_components=N_COMPONENTS, random_state=42)
    df["gmm_cluster"] = gmm.fit_predict(X)

    print(f"  GMM fitted with k={N_COMPONENTS} components")

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        df["Temperature"],
        df["Irradiance"],
        c=df["gmm_cluster"],
        cmap="coolwarm",
        alpha=0.4,
        s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Irradiance")
    ax.set_title(f"GMM Clusters (n={N_COMPONENTS})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "gmm_clusters.png")
    plt.close(fig)

    try:
        plt.style.use("seaborn-v0_8")
        x_min, x_max = df["Temperature"].min(), df["Temperature"].max()
        y_min, y_max = df["Irradiance"].min(), df["Irradiance"].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = np.exp(gmm.score_samples(grid)).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(12, 6))
        contour = ax.contourf(xx, yy, zz, levels=25, cmap="YlOrRd", alpha=0.8)
        ax.scatter(df["Temperature"], df["Irradiance"], c=df["gmm_cluster"], cmap="coolwarm", s=8, alpha=0.35)
        plt.colorbar(contour, ax=ax, label="Density")
        ax.set_title("GMM Probability Contours with Cluster Overlay")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Irradiance")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "gmm_contour.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate GMM contour plot: {exc}")

    try:
        plt.style.use("seaborn-v0_8")
        confidence = gmm.predict_proba(X).max(axis=1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(confidence, bins=30, color="tab:blue", alpha=0.8)
        ax.set_title("GMM Cluster Membership Confidence")
        ax.set_xlabel("Max Posterior Probability")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        save_plot(fig, "gmm_confidence.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Could not generate GMM confidence histogram: {exc}")

    print("  Saved: gmm_bic_aic.png, gmm_clusters.png")
    return df


def main() -> pd.DataFrame:
    return run_gmm()


if __name__ == "__main__":
    main()
