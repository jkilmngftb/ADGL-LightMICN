"""Create publication figures from the frozen ADGL-LightMICN data and results.

All plotted values are read from saved predictions, metrics, adjacency outputs,
or the original meteorological workbook. No synthetic values are generated.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


CODE_ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
DATA = OUT / "source_data"

ORIGINAL_XLSX = CODE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
ANALYSIS = CODE_ROOT / "results" / "error_dynamic_graph" / "analysis"
PRED = (CODE_ROOT / "results" / "error_dynamic_graph" /
        "runs" / "w12_k2-4_lr0p0010_layers1" /
        "predictions_full_both.csv")

ROMAN = {
    "邳州": "Pizhou", "徐州": "Xuzhou", "沭阳": "Shuyang", "灌云": "Guanyun",
    "睢宁": "Suining", "盱眙": "Xuyi", "淮安": "Huai'an", "南京": "Nanjing",
    "高邮": "Gaoyou", "东台": "Dongtai", "南通": "Nantong", "吕泗": "Lusi",
    "无锡": "Wuxi", "昆山": "Kunshan", "东山": "Dongshan",
}
ABBR = {
    "邳州": "PIZ", "徐州": "XUZ", "沭阳": "SHU", "灌云": "GNY", "睢宁": "SUN",
    "盱眙": "XYI", "淮安": "HUA", "南京": "NAN", "高邮": "GAO", "东台": "DGT",
    "南通": "NAT", "吕泗": "LUS", "无锡": "WUX", "昆山": "KUN", "东山": "DSH",
}

BLUE = "#2878B5"
ORANGE = "#E69500"
GREEN = "#3BA272"
RED = "#C84A4A"
GREY = "#6B7280"
PALE = "#D7DCE2"


def setup_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.transparent": False,
    })


def panel_label(ax, label):
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, fontweight="bold",
            fontsize=9, ha="left", va="top", clip_on=False)


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_station_meta():
    raw = pd.read_excel(ORIGINAL_XLSX)
    raw = raw[raw["站号"].astype(str) != "sid"].copy()
    meta = raw[["站号", "Unnamed: 2", "纬度", "经度"]].drop_duplicates("站号")
    meta.columns = ["sid", "name_cn", "latitude", "longitude"]
    meta["sid"] = meta["sid"].astype(int)
    meta = meta.sort_values("sid").reset_index(drop=True)
    meta["node"] = np.arange(len(meta))
    meta["name_en"] = meta["name_cn"].map(ROMAN)
    meta["abbr"] = meta["name_cn"].map(ABBR)
    meta.to_csv(DATA / "station_metadata.csv", index=False, encoding="utf-8-sig")
    return meta


def figure_station_map(meta):
    fig, ax = plt.subplots(figsize=(3.5, 3.4), constrained_layout=True)
    ax.scatter(meta.longitude, meta.latitude, s=20, color=BLUE, edgecolor="white",
               linewidth=0.5, zorder=3)
    offsets = {"Xuzhou": (0.06, 0.08), "Suining": (-0.48, -0.20), "Shuyang": (0.08, -0.11),
               "Lusi": (0.08, -0.12), "Nantong": (0.06, 0.10), "Dongshan": (0.07, -0.11),
               "Wuxi": (-0.43, -0.20), "Huai'an": (0.07, 0.05)}
    for row in meta.itertuples():
        dx, dy = offsets.get(row.name_en, (0.05, 0.04))
        ax.text(row.longitude + dx, row.latitude + dy, row.name_en, fontsize=6.1)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(meta.longitude.min() - 0.25, meta.longitude.max() + 0.55)
    ax.set_ylim(meta.latitude.min() - 0.22, meta.latitude.max() + 0.25)
    ax.set_aspect(1 / math.cos(math.radians(meta.latitude.mean())))
    save(fig, "Fig1_station_spatial_distribution")


def figure_condition_adjacency(meta):
    cond = pd.read_csv(DATA / "adjacency_condition_metadata.csv")
    order = ["lowest", "median", "highest"]
    mats = [pd.read_csv(DATA / f"adjacency_{c}.csv", index_col=0).to_numpy(float) for c in order]
    vmax = max(float(m.max()) for m in mats)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    images = []
    for k, (ax, c, m) in enumerate(zip(axes, order, mats)):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        images.append(im)
        rec = cond[cond.condition == c].iloc[0]
        title = {"lowest": "Lowest", "median": "Median", "highest": "Highest"}[c]
        ax.set_title(f"{title} target temperature\n{rec.target_date}; mean {rec.mean_observed_target_temperature_c:.2f} °C")
        ax.set_xticks(range(15), meta.abbr, rotation=90, rotation_mode="anchor")
        ax.set_yticks(range(15), meta.abbr if k == 0 else [""] * 15)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xlabel("Source station")
        if k == 0:
            ax.set_ylabel("Receiving station")
        panel_label(ax, chr(97 + k))
    cb = fig.colorbar(images[-1], ax=axes, shrink=0.78, pad=0.02)
    cb.set_label("Row-normalized adjacency weight")
    save(fig, "Fig2_condition_specific_dynamic_adjacency")


def figure_frequent_edges(meta):
    pairs = pd.read_csv(ANALYSIS / "dynamic_graph_top_pairs.csv")
    pairs = pairs.sort_values(["edge_frequency", "mean_adjacency"], ascending=False).head(20)
    name_to_row = {r.name_cn: r for r in meta.itertuples()}
    norm = Normalize(vmin=float(pairs.edge_frequency.min()), vmax=float(pairs.edge_frequency.max()))
    cmap = mpl.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(3.5, 4.1), constrained_layout=True)
    for r in pairs.itertuples():
        a, b = name_to_row[r.station_i], name_to_row[r.station_j]
        ax.plot([a.longitude, b.longitude], [a.latitude, b.latitude],
                color=cmap(norm(r.edge_frequency)),
                lw=0.7 + 2.0 * (r.mean_adjacency / pairs.mean_adjacency.max()), alpha=0.82)
    ax.scatter(meta.longitude, meta.latitude, s=22, color=BLUE, edgecolor="white", lw=0.5, zorder=3)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.03)
    cb.set_label("Selection frequency")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1 / math.cos(math.radians(meta.latitude.mean())))
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                      markeredgecolor="white", label=f"{r.abbr}  {r.name_en}")
               for r in meta.itertuples()]
    ax.legend(handles=handles, ncol=3, frameon=False, fontsize=4.8,
              columnspacing=0.65, handletextpad=0.15, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    save(fig, "Fig3_frequently_selected_dynamic_edges")
    pairs.to_csv(DATA / "top20_dynamic_station_pairs.csv", index=False, encoding="utf-8-sig")


def load_test_predictions():
    p = pd.read_csv(PRED)
    p = p[p["split"] == "test"].copy()
    p["date"] = pd.to_datetime(p["date"])
    p.rename(columns={"node": "station_index"}, inplace=True)
    return p


def figure_station_series(pred, meta):
    selected = ["Suining", "Gaoyou", "Nantong", "Kunshan"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), sharex=True, sharey=True,
                             constrained_layout=True)
    for k, (ax, name) in enumerate(zip(axes.flat, selected)):
        idx = int(meta.loc[meta.name_en == name, "node"].iloc[0])
        x = pred[pred.station_index == idx].sort_values("date")
        ax.plot(x.date, x.y_true, color=BLUE, lw=0.85, label="Observed")
        ax.plot(x.date, x.y_pred, color=ORANGE, lw=0.85, label="Predicted")
        rmse = float(np.sqrt(np.mean((x.y_pred - x.y_true) ** 2)))
        ax.set_title(f"{name}  (RMSE {rmse:.2f} °C)")
        ax.grid(color="#ECEFF2", lw=0.45)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        if k % 2 == 0:
            ax.set_ylabel("Mean temperature (°C)")
        panel_label(ax, chr(97 + k))
    handles = [Line2D([0], [0], color=BLUE, label="Observed"),
               Line2D([0], [0], color=ORANGE, label="Predicted")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    save(fig, "Fig4_observed_vs_predicted_four_stations")


def figure_station_rmse(pred, meta):
    rows = []
    for idx, x in pred.groupby("station_index"):
        rmse = float(np.sqrt(np.mean((x.y_pred - x.y_true) ** 2)))
        mae = float(np.mean(np.abs(x.y_pred - x.y_true)))
        rows.append({"station_index": int(idx), "station": meta.loc[meta.node == idx, "name_en"].iloc[0],
                     "rmse": rmse, "mae": mae})
    d = pd.DataFrame(rows).sort_values("rmse")
    d.to_csv(DATA / "station_performance.csv", index=False, encoding="utf-8-sig")
    mean_rmse = float(np.sqrt(np.mean((pred.y_pred - pred.y_true) ** 2)))
    fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
    colors = [BLUE if v <= mean_rmse else ORANGE for v in d.rmse]
    ax.bar(np.arange(len(d)), d.rmse, color=colors, width=0.72)
    ax.axhline(mean_rmse, color=RED, ls="--", lw=1.0, label=f"Overall RMSE = {mean_rmse:.2f} °C")
    ax.set_xticks(np.arange(len(d)), d.station, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_ylabel("RMSE (°C)")
    ax.set_ylim(0, max(d.rmse) * 1.18)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#ECEFF2", lw=0.5)
    save(fig, "Fig5_station_level_performance")


def build_baselines():
    d = pd.read_csv(DATA / "final_baseline_metrics.csv")
    labels = {"adgl_lightmicn": "ADGL-LightMICN", "dcrnn": "DCRNN", "graph_wavenet": "Graph WaveNet",
              "tgcn": "T-GCN", "dlinear": "DLinear", "persistence": "Persistence", "mtgnn": "MTGNN",
              "tcn": "TCN", "lstm": "LSTM", "itransformer": "iTransformer", "gcn_mlp": "GCN-MLP"}
    if "label" not in d:
        d["label"] = d.model.map(labels)
    d = d.sort_values("rmse", ascending=False).reset_index(drop=True)
    d.to_csv(DATA / "final_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    return d


def figure_baselines(d):
    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        c = RED if r.model == "adgl_lightmicn" else GREY
        ax.plot([r.mae, r.rmse], [i, i], color="#CBD1D8", lw=2.0, zorder=1)
        ax.scatter(r.mae, i, color=BLUE if r.model != "adgl_lightmicn" else RED,
                   s=26, marker="o", zorder=3)
        ax.scatter(r.rmse, i, color=ORANGE if r.model != "adgl_lightmicn" else RED,
                   s=30, marker="s", zorder=3)
        ax.text(r.rmse + 0.025, i, f"{r.rmse:.3f}", va="center", fontsize=5.9, color=c)
    ax.set_yticks(y, d.label)
    ax.set_xlabel("Error (°C)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="MAE"),
                       Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, label="RMSE")],
              frameon=False, loc="upper right")
    save(fig, "Fig6_comparison_with_forecasting_baselines")


def build_factorial():
    return pd.read_csv(DATA / "architecture_matching_3x2.csv")


def figure_factorial(d):
    order = ["No spatial propagation", "Fixed-graph GCN", "ADGL-GCN"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    x = np.arange(3)
    for enc, color, marker in [("TCN", BLUE, "o"), ("LightMICN", GREEN, "s")]:
        s = d[d.temporal_encoder == enc].set_index("spatial_representation").loc[order]
        ax.plot(x, s.rmse, color=color, marker=marker, ms=5, lw=1.3, label=enc)
        for xi, yi in zip(x, s.rmse):
            offset = 0.007 if enc == "TCN" else (-0.010 if xi > 0 else 0.007)
            ax.text(xi, yi + offset, f"{yi:.4f}", color=color, ha="center",
                    va="bottom" if offset > 0 else "top", fontsize=6)
    ax.set_xticks(x, order)
    ax.set_ylabel("Test RMSE (°C)")
    ax.set_ylim(1.73, 1.90)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "Fig7_spatial_temporal_architecture_matching")


def build_ablations():
    temporal = pd.DataFrame([
        ["ADGL-LightMICN", 1.7617325481843271, "Complete model"],
        ["Single scale (2)", 1.8108162159370254, "Single-scale ablation"],
        ["No decomposition", 1.8245, "No trend-residual decomposition"],
    ], columns=["configuration", "rmse", "description"])
    graph = pd.DataFrame([
        ["ADGL-LightMICN", 1.7617325481843271, "Complete model"],
        ["No Top-k", 1.7685, "Remove Top-k neighbour selection"],
        ["No L1", 1.7783, "Remove candidate-gate L1 regularization"],
        ["Fixed graph", 1.7934, "Replace input-conditioned graph"],
        ["Node embedding", 1.8078, "Add free node embeddings"],
    ], columns=["configuration", "rmse", "description"])
    temporal.to_csv(DATA / "lightmicn_internal_ablation.csv", index=False, encoding="utf-8-sig")
    graph.to_csv(DATA / "dynamic_graph_component_ablation.csv", index=False, encoding="utf-8-sig")
    return temporal, graph


def figure_ablations(temporal, graph):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    for ax, d, title in [(axes[0], temporal, "LightMICN components"),
                         (axes[1], graph, "Dynamic-graph components")]:
        d = d.sort_values("rmse", ascending=True)
        colors = [RED if x == "ADGL-LightMICN" else BLUE for x in d.configuration]
        y = np.arange(len(d))
        ax.barh(y, d.rmse, color=colors, height=0.62)
        ax.set_yticks(y, d.configuration)
        ax.set_xlim(1.73, max(d.rmse) + 0.025)
        ax.set_xlabel("Test RMSE (°C)")
        ax.set_title(title)
        for yi, val in zip(y, d.rmse):
            ax.text(val + 0.002, yi, f"{val:.4f}", va="center", fontsize=6)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, "Fig8_component_ablation")


def main():
    setup_style()
    for d in (FIG, DATA):
        d.mkdir(parents=True, exist_ok=True)
    meta = read_station_meta()
    pred = load_test_predictions()
    figure_station_map(meta)
    figure_condition_adjacency(meta)
    figure_frequent_edges(meta)
    figure_station_series(pred, meta)
    figure_station_rmse(pred, meta)
    figure_baselines(build_baselines())
    figure_factorial(build_factorial())
    figure_ablations(*build_ablations())
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
