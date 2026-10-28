"""Reference-shaped three-panel ablation figure using frozen experiment data.

The grouped-bar positioning and bar labels adapt Matplotlib's official example:
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
DATA, FIG = ROOT / "source_data", ROOT / "figures"

BLUE = "#4DA6D8"
CORAL = "#EF6A5B"
GREEN = "#49C779"
GOLD = "#E6A23C"
ORANGE = "#D55E00"
BLACK = "#1A1A1A"
GRID = "#E5E5E5"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 6.7,
        "axes.titlesize": 7.3,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.65,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def grouped_bars(ax, x, data, colors, labels, width, offsets):
    containers = []
    for series_index, (values, color, label, offset) in enumerate(
            zip(data, colors, labels, offsets)):
        bars = ax.bar(x + offset, values, width=width, color=color,
                      edgecolor="none", label=label, zorder=3)
        containers.append(bars)
        for rect, value in zip(bars, values):
            if abs(value) < 0.005:
                continue
            va = "bottom" if value >= 0 else "top"
            dy = 0.10 if value >= 0 else (-0.10 - 0.18 * series_index)
            x_shift = 0.0
            if len(data) == 2:
                x_shift = -0.08 if series_index == 0 else 0.08
            ax.text(rect.get_x() + rect.get_width() / 2 + x_shift, value + dy,
                    f"{value:.2f}%", ha="center", va=va,
                    fontsize=5.8, fontweight="bold", color=BLACK,
                    clip_on=False)
    return containers


def format_panel(ax, title, xticks, xlabels, show_ylabel=False):
    ax.set_title(title, pad=4, fontweight="bold")
    ax.set_xticks(xticks, xlabels)
    ax.set_ylim(-4.6, 3.05)
    ax.set_yticks(np.arange(-4, 3, 2))
    ax.axhline(0, color=BLACK, linewidth=0.75, zorder=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#555555")
    if show_ylabel:
        ax.set_ylabel("RMSE change relative to reference (%)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel("Experiment group", labelpad=2)


def main() -> None:
    style()
    graph = pd.read_csv(DATA / "dynamic_graph_component_ablation.csv")
    temporal = pd.read_csv(DATA / "lightmicn_internal_ablation.csv")
    matching = pd.read_csv(DATA / "architecture_matching_3x2.csv")
    full = float(graph.loc[graph.configuration == "ADGL-LightMICN", "rmse"].iloc[0])

    # a-b: negative values mean that the complete model improves upon the
    # corresponding ablated/alternative configuration, matching the reference.
    graph_names = ["Fixed graph", "No Top-k", "No L1", "Node embedding"]
    graph_rmse = graph.set_index("configuration").loc[graph_names, "rmse"].to_numpy()
    graph_change = (full - graph_rmse) / graph_rmse * 100.0

    adgl_tcn = float(matching.loc[
        (matching.spatial_representation == "ADGL-GCN") &
        (matching.temporal_encoder == "TCN"), "rmse"].iloc[0])
    temporal_names = ["ADGL-TCN", "Single scale", "No decomposition"]
    temporal_rmse = np.array([
        adgl_tcn,
        float(temporal.loc[temporal.configuration == "Single scale (2)", "rmse"].iloc[0]),
        float(temporal.loc[temporal.configuration == "No decomposition", "rmse"].iloc[0]),
    ])
    temporal_change = (full - temporal_rmse) / temporal_rmse * 100.0

    # c: all six combinations are expressed relative to No spatial + TCN.
    spatial_order = ["No spatial propagation", "Fixed-graph GCN", "ADGL-GCN"]
    reference = float(matching.loc[
        (matching.spatial_representation == "No spatial propagation") &
        (matching.temporal_encoder == "TCN"), "rmse"].iloc[0])
    match_values = {}
    match_rmse = {}
    for encoder in ["TCN", "LightMICN"]:
        vals = (matching[matching.temporal_encoder == encoder]
                .set_index("spatial_representation").loc[spatial_order, "rmse"]
                .to_numpy())
        match_rmse[encoder] = vals
        match_values[encoder] = (vals - reference) / reference * 100.0

    fig = plt.figure(figsize=(7.35, 2.65))
    axes = np.array([
        fig.add_axes([0.070, 0.20, 0.270, 0.61]),
        fig.add_axes([0.365, 0.20, 0.270, 0.61]),
        fig.add_axes([0.660, 0.20, 0.270, 0.61]),
    ])
    fig.suptitle("Ablation analysis of ADGL-LightMICN", y=0.965,
                 fontsize=8.3, fontweight="bold")
    fig.text(0.5, 0.905,
             "RMSE change relative to the panel reference; negative values indicate improvement",
             ha="center", va="center", fontsize=6.2, fontweight="bold")

    x0 = np.array([0.0])
    offsets_a = np.linspace(-0.255, 0.255, 4)
    grouped_bars(axes[0], x0, [np.array([v]) for v in graph_change],
                 [BLUE, CORAL, GREEN, GOLD], graph_names, 0.15, offsets_a)
    axes[0].set_xlim(-0.55, 0.55)
    format_panel(axes[0], "Dynamic-graph components", x0, ["Complete model vs. ablations"], True)
    axes[0].legend(loc="upper left", frameon=True, facecolor="white",
                   edgecolor="#AAAAAA", ncols=1, borderpad=0.35,
                   handlelength=1.1, handletextpad=0.35)

    offsets_b = np.linspace(-0.17, 0.17, 3)
    grouped_bars(axes[1], x0, [np.array([v]) for v in temporal_change],
                 [BLUE, CORAL, GREEN], temporal_names, 0.15, offsets_b)
    axes[1].set_xlim(-0.55, 0.55)
    format_panel(axes[1], "Temporal encoder", x0, ["Complete model vs. alternatives"])
    axes[1].legend(loc="upper left", frameon=True, facecolor="white",
                   edgecolor="#AAAAAA", ncols=1, borderpad=0.35,
                   handlelength=1.1, handletextpad=0.35)

    x = np.arange(3, dtype=float)
    grouped_bars(axes[2], x,
                 [match_values["TCN"], match_values["LightMICN"]],
                 [BLUE, CORAL], ["TCN", "LightMICN"], 0.42,
                 [-0.21, 0.21])
    axes[2].set_xlim(-0.55, 2.55)
    format_panel(axes[2], "Spatial × temporal matching", x,
                 ["No spatial", "Fixed GCN", "ADGL-GCN"])
    axes[2].set_xlabel("Spatial representation", labelpad=2)
    axes[2].legend(loc="upper right", frameon=True, facecolor="white",
                   edgecolor="#AAAAAA", ncols=2, borderpad=0.35,
                   handlelength=1.1, handletextpad=0.35, columnspacing=0.7)

    for label, ax in zip("abc", axes):
        ax.text(-0.11, 1.04, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.4, fontweight="bold",
                clip_on=False)

    FIG.mkdir(parents=True, exist_ok=True)
    stem = "Fig_ablation_three_panel_reference_layout_uniform_barwidth"
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, rmse, change in zip(graph_names, graph_rmse, graph_change):
        rows.append(["a", name, rmse, full, change])
    for name, rmse, change in zip(temporal_names, temporal_rmse, temporal_change):
        rows.append(["b", name, rmse, full, change])
    for encoder in ["TCN", "LightMICN"]:
        for spatial, rmse, change in zip(spatial_order, match_rmse[encoder], match_values[encoder]):
            rows.append(["c", f"{spatial} + {encoder}", rmse, reference, change])
    pd.DataFrame(rows, columns=["panel", "configuration", "rmse",
                                "reference_rmse", "relative_change_pct"]).to_csv(
        DATA / "ablation_reference_layout_plot_data.csv",
        index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
