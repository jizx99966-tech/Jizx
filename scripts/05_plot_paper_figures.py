from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from salt_ml.data_utils import load_config, read_csv
from salt_ml.plotting import save_fig, set_paper_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style figures")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_paper_style()

    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    processed = read_csv(cfg["paths"]["processed_csv"])
    cluster_path = Path(cfg["paths"]["cluster_csv"])

    # Figure 1: trajectory envelope of VolShrinkRatio
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=processed, x="year", y="VolShrinkRatio", estimator="mean", errorbar=("ci", 95))
    plt.title("Mean trajectory of VolShrinkRatio (95% CI)")
    plt.xlabel("Year")
    plt.ylabel("VolShrinkRatio")
    save_fig(figures_dir / "fig1_vsr_trajectory.png")

    # Figure 2: clustered trajectories (if available)
    if cluster_path.exists():
        cluster_df = read_csv(cluster_path)
        merged = processed.merge(cluster_df, on="case_id", how="left")

        plt.figure(figsize=(7, 4.5))
        sns.lineplot(
            data=merged,
            x="year",
            y="VolShrinkRatio",
            hue="cluster",
            estimator="mean",
            errorbar=None,
            palette="tab10",
        )
        plt.title("Cluster-wise mean trajectory")
        plt.xlabel("Year")
        plt.ylabel("VolShrinkRatio")
        save_fig(figures_dir / "fig2_cluster_trajectory.png")

    # Figure 3: SHAP bar chart (if available)
    shap_path = Path(cfg["paths"]["shap_csv"])
    if shap_path.exists():
        shap_df = read_csv(shap_path).head(20)
        plt.figure(figsize=(6, 6))
        sns.barplot(data=shap_df, y="feature", x="mean_abs_shap", color="#4C78A8")
        plt.title("Top-20 global feature importance")
        plt.xlabel("mean(|SHAP|)")
        plt.ylabel("")
        save_fig(figures_dir / "fig3_shap_top20.png")

    print(f"Figures saved in: {figures_dir}")


if __name__ == "__main__":
    main()
