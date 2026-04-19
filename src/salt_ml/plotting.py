from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = "colorblind"


def set_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
        }
    )


def save_fig(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
