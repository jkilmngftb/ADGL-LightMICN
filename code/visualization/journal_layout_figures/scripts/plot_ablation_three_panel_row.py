"""One-row, three-panel ablation figure for ADGL-LightMICN.

Numerical values are read from the frozen experiment CSV files in
``source_data``.  The grouped-bar positioning and value-label pattern adapts
Matplotlib's official grouped bar example:
https://github.com/matplotlib/matplotlib/blob/main/galleries/examples/lines_bars_and_markers/barchart.py
Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
FIG = ROOT / "figures"

BLUE = "#2F6FA3"
LIGHT_BLUE = "#76A5C8"
ORANGE = "#D55E00"
BLACK = "#1A1A1A"
GRID = "#D9D9D9"


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.07, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", clip_on=False)


def add_percent_panel(ax: plt.Axes, values: pd.DataFrame, labels: list[str],
                      title: str, ymax: float) -> None:
    x = np.arange(len(values))
    bars = ax.bar(x, values["delta_pct"], width=0.62, color=BLUE,
                  edgecolor="white", linewidth=0.45)
    ax.bar_label(bars, labels=[f"+{v:.2f}%" for v in values["delta_pct"]],
                 padding=2.5, fontsize=6.3, color=BLACK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("RMSE increase vs. complete model (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0.02, 0.95, "Reference RMSE = 1.7617 °C",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            color=ORANGE)


def main() -> None:
    set_style()
    graph = pd.read_csv(DATA / "dynamic_graph_component_ablation.csv")
    temporal = pd.read_csv(DATA / "lightmicn_internal_ablation.csv")
    matching = pd.read_csv(DATA / "architecture_matching_3x2.csv")

    full = float(graph.loc[graph["configuration"] == "ADGL-LightMICN", "rmse"].iloc[0])

    graph_order = ["No Top-k", "No L1", "Fixed graph", "Node embedding"]
    graph_plot = graph.set_index("configuration").loc[graph_order].reset_index()
    graph_plot["delta_pct"] = (graph_plot["rmse"] / full - 1.0) * 100.0

    adgl_tcn = float(matching.loc[
        (matching["spatial_representation"] == "ADGL-GCN") &
        (matching["temporal_encoder"] == "TCN"), "rmse"].iloc[0])
    temporal_plot = pd.concat([
        pd.DataFrame({"configuration": ["ADGL-TCN"], "rmse": [adgl_tcn]}),
        temporal.set_index("configuration").loc[
            ["Single scale (2)", "No decomposition"]].reset_index()[
                ["configuration", "rmse"]],
    ], ignore_index=True)
    temporal_plot["delta_pct"] = (temporal_plot["rmse"] / full - 1.0) * 100.0

    # Fixed axes positions keep all three journal panels equal in width and
    # preserve identical horizontal gutters despite different y-label lengths.
    fig = plt.figure(figsize=(7.35, 2.75))
    axes = np.array([
        fig.add_axes([0.075, 0.20, 0.255, 0.70]),
        fig.add_axes([0.390, 0.20, 0.255, 0.70]),
        fig.add_axes([0.705, 0.20, 0.255, 0.70]),
    ])

    add_percent_panel(
        axes[0], graph_plot,
        ["No\nTop-k", "No\nL1", "Fixed\ngraph", "Node\nembedding"],
        "Dynamic-graph components", 4.05)
    panel_label(axes[0], "a")

    add_percent_panel(
        axes[1], temporal_plot,
        ["ADGL-\nTCN", "Single\nscale", "No\ndecomposition"],
        "Temporal encoder", 4.05)
    panel_label(axes[1], "b")

    ax = axes[2]
    spatial_order = ["No spatial propagation", "Fixed-graph GCN", "ADGL-GCN"]
    spatial_labels = ["No spatial\npropagation", "Fixed-graph\nGCN", "ADGL-GCN"]
    x = np.arange(len(spatial_order))
    width = 0.34
    containers = []
    for offset, encoder, color in [(-width / 2, "TCN", LIGHT_BLUE),
                                   (width / 2, "LightMICN", ORANGE)]:
        vals = (matching[matching["temporal_encoder"] == encoder]
                .set_index("spatial_representation").loc[spatial_order, "rmse"]
                .to_numpy())
        bars = ax.bar(x + offset, vals, width, label=encoder, color=color,
                      edgecolor="none", linewidth=0)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals],
                     label_type="center", padding=0,
                     fontsize=6.1, rotation=90, rotation_mode="anchor",
                     color="white", fontweight="bold")
        containers.append(bars)
    ax.set_xticks(x, spatial_labels)
    ax.set_ylim(0, 2.34)
    ax.set_ylabel("Test RMSE (°C)")
    ax.set_title("Spatial × temporal matching", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncols=2,
              handlelength=1.0, columnspacing=0.8,
              bbox_to_anchor=(0.5, 0.99))
    panel_label(ax, "c")

    FIG.mkdir(parents=True, exist_ok=True)
    stem = "Fig_ablation_three_panel_row"
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    source_rows = []
    for _, row in graph_plot.iterrows():
        source_rows.append({"panel": "a", "category": row["configuration"],
                            "series": "Ablation", "rmse": row["rmse"],
                            "reference_rmse": full,
                            "relative_change_pct": row["delta_pct"]})
    for _, row in temporal_plot.iterrows():
        source_rows.append({"panel": "b", "category": row["configuration"],
                            "series": "Alternative", "rmse": row["rmse"],
                            "reference_rmse": full,
                            "relative_change_pct": row["delta_pct"]})
    for _, row in matching.iterrows():
        source_rows.append({"panel": "c", "category": row["spatial_representation"],
                            "series": row["temporal_encoder"], "rmse": row["rmse"],
                            "reference_rmse": np.nan,
                            "relative_change_pct": np.nan})
    pd.DataFrame(source_rows).to_csv(
        DATA / "ablation_three_panel_plot_data.csv", index=False,
        encoding="utf-8-sig")


if __name__ == "__main__":
    main()
