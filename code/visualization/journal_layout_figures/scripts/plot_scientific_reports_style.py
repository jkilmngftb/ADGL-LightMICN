"""Scientific Reports-style figures for the ADGL-LightMICN manuscript.

The visual grammar is reimplemented with Matplotlib from published Scientific
Reports examples. All numerical values come from frozen project outputs copied
into ``source_data``; no values are simulated.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize

CODE_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = CODE_ROOT / "visualization" / "main_figures"
OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "source_data"
FIG = OUT / "figures"

PRED = (CODE_ROOT / "results" / "error_dynamic_graph" / "runs" /
        "w12_k2-4_lr0p0010_layers1" /
        "predictions_full_both.csv")
PERSISTENCE_PRED = (CODE_ROOT / "results" / "comparison" /
                    "persistence" / "test_predictions.csv")

BLUE = "#2F6FA3"
LIGHT_BLUE = "#76A5C8"
ORANGE = "#D55E00"
GOLD = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
BLACK = "#1A1A1A"


def style() -> None:
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
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def panel(ax, label: str) -> None:
    ax.text(-0.15, 1.07, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", clip_on=False)


def save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_source_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for p in (SRC_ROOT / "source_data").glob("*.csv"):
        shutil.copy2(p, DATA / p.name)
    for p in (SRC_ROOT / "source_data").glob("*.json"):
        shutil.copy2(p, DATA / p.name)
    analysis = CODE_ROOT / "results" / "error_dynamic_graph" / "analysis"
    for name in ["dynamic_graph_mean_adjacency.csv", "dynamic_graph_edge_frequency.csv"]:
        shutil.copy2(analysis / name, DATA / name)


def station_map(meta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 4.0), constrained_layout=True)
    ax.scatter(meta.longitude, meta.latitude, s=32, color=ORANGE,
               edgecolor="white", linewidth=0.65, zorder=3)
    offsets = {
        "Xuzhou": (0.06, 0.08), "Suining": (-0.50, -0.18),
        "Shuyang": (0.06, -0.12), "Lusi": (0.07, -0.13),
        "Nantong": (0.07, 0.09), "Dongshan": (0.07, -0.12),
        "Wuxi": (-0.40, -0.18), "Huai'an": (0.07, 0.05),
        "Gaoyou": (0.07, -0.13), "Kunshan": (0.07, -0.02),
    }
    for r in meta.itertuples():
        dx, dy = offsets.get(r.name_en, (0.06, 0.04))
        ax.text(r.longitude + dx, r.latitude + dy, f"{r.abbr}  {r.name_en}",
                fontsize=6.4, va="center")

    # Geographic orientation without inventing a provincial boundary.
    lon0 = meta.longitude.min() - 0.12
    lat0 = meta.latitude.min() + 0.10
    km = 100
    dlon = km / (111.32 * math.cos(math.radians(meta.latitude.mean())))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=BLACK, lw=1.5)
    ax.plot([lon0, lon0], [lat0 - 0.035, lat0 + 0.035], color=BLACK, lw=0.8)
    ax.plot([lon0 + dlon, lon0 + dlon], [lat0 - 0.035, lat0 + 0.035], color=BLACK, lw=0.8)
    ax.text(lon0 + dlon / 2, lat0 + 0.08, "100 km", ha="center", fontsize=6.3)
    ax.annotate("N", xy=(0.93, 0.91), xytext=(0.93, 0.78),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=BLACK))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(meta.longitude.min() - 0.30, meta.longitude.max() + 0.72)
    ax.set_ylim(meta.latitude.min() - 0.23, meta.latitude.max() + 0.26)
    ax.set_aspect(1 / math.cos(math.radians(meta.latitude.mean())))
    save(fig, "Fig1_station_network")


def dynamic_graph(meta: pd.DataFrame) -> None:
    cond = pd.read_csv(DATA / "adjacency_condition_metadata.csv")
    order = ["lowest", "median", "highest"]
    mats = [pd.read_csv(DATA / f"adjacency_{c}.csv", index_col=0).to_numpy(float)
            for c in order]
    # Use one scale across all three conditions so color intensity is directly
    # comparable. These are learned, Top-k-filtered adjacency weights, not
    # temperature correlations.
    vmax = max(float(m.max()) for m in mats)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.55), constrained_layout=True)
    titles = {"lowest": "Lowest-temperature case", "median": "Median-temperature case",
              "highest": "Highest-temperature case"}
    im = None
    for k, (ax, c, mat) in enumerate(zip(axes[:3], order, mats)):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        rec = cond.loc[cond.condition == c].iloc[0]
        ax.set_title(f"({chr(97 + k)}) {titles[c]}\n{rec.target_date}; mean {rec.mean_observed_target_temperature_c:.1f} °C")
        # Full station names let readers avoid decoding station abbreviations.
        ax.set_xticks(range(len(meta)), meta.name_en, rotation=90,
                      rotation_mode="anchor", ha="right")
        ax.set_yticks(range(len(meta)), meta.name_en)
        ax.tick_params(axis="x", labelsize=5.2, length=0, pad=2)
        ax.tick_params(axis="y", labelsize=5.2, length=0, pad=2)
        ax.set_xlabel("Source station")
        ax.set_ylabel("Receiving station")
        ax.set_xlim(-0.5, len(meta) - 0.5)
        ax.set_ylim(len(meta) - 0.5, -0.5)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.025)
    cbar.set_label("Row-normalized adjacency weight")
    fig.text(
        0.5, -0.015,
        "Top-k filtered (k = 4), row-normalized dynamic adjacency; "
        "rows: receiving stations, columns: source stations (Aᵢⱼ: j → i).",
        ha="center", va="top", fontsize=6.2)
    save(fig, "Fig2_input_conditioned_graphs")


def dynamic_edge_map(meta: pd.DataFrame) -> None:
    """Map high-frequency directed edges from the saved ADGL graph analysis."""
    freq_mat = pd.read_csv(DATA / "dynamic_graph_edge_frequency.csv", index_col=0).to_numpy(float)
    mean_mat = pd.read_csv(DATA / "dynamic_graph_mean_adjacency.csv", index_col=0).to_numpy(float)
    names = meta.name_cn.tolist()
    rows = []
    # A_ij is the weight from source j to receiving node i.
    for receiver in range(len(names)):
        for source in range(len(names)):
            if receiver == source:
                continue
            rows.append({"receiver_idx": receiver, "source_idx": source,
                         "edge_frequency": float(freq_mat[receiver, source]),
                         "mean_adjacency": float(mean_mat[receiver, source])})
    pairs = pd.DataFrame(rows).sort_values(
        ["edge_frequency", "mean_adjacency"], ascending=False).head(20).reset_index(drop=True)
    pairs.insert(0, "rank", np.arange(1, len(pairs) + 1))
    pairs["source_station"] = pairs.source_idx.map(lambda i: names[int(i)])
    pairs["receiver_station"] = pairs.receiver_idx.map(lambda i: names[int(i)])
    pairs.to_csv(DATA / "top20_dynamic_directed_edges.csv", index=False, encoding="utf-8-sig")
    station_rows = {int(r.node): r for r in meta.itertuples()}
    freq = pairs.edge_frequency.to_numpy(float)
    norm = Normalize(vmin=float(freq.min()), vmax=float(freq.max()))
    cmap = mpl.colormaps["Blues"]

    # Node size summarizes frequency-weighted incidence in the displayed edge set.
    incidence = {name: 0.0 for name in names}
    for r in pairs.itertuples():
        incidence[r.source_station] += float(r.edge_frequency)
        incidence[r.receiver_station] += float(r.edge_frequency)
    vals = np.array(list(incidence.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    sizes = {name: 150 + 90 * (value - lo) / max(hi - lo, 1e-12)
             for name, value in incidence.items()}

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=False)
    ax = fig.add_axes([0.10, 0.30, 0.74, 0.62])
    cax = fig.add_axes([0.88, 0.37, 0.025, 0.46])
    ax.set_title("Top 20 directed links across 662 test windows", loc="center")
    # A_ij means source j sends information to receiving station i.
    for r in pairs.itertuples():
        receiver = station_rows[int(r.receiver_idx)]
        source = station_rows[int(r.source_idx)]
        strength = norm(float(r.edge_frequency))
        # Reciprocal pairs bend in opposite directions so both directions remain visible.
        reciprocal = bool(((pairs.source_idx == r.receiver_idx) &
                           (pairs.receiver_idx == r.source_idx)).any())
        curve = (0.17 if int(r.source_idx) < int(r.receiver_idx) else -0.17)
        arrow = FancyArrowPatch(
            (source.longitude, source.latitude),
            (receiver.longitude, receiver.latitude),
            arrowstyle="-|>", mutation_scale=7.0,
            connectionstyle=f"arc3,rad={curve}",
            linewidth=0.75 + 1.35 * strength,
            color=cmap(0.42 + 0.52 * strength), alpha=0.82,
            shrinkA=5, shrinkB=7, zorder=1)
        ax.add_patch(arrow)

    for r in meta.itertuples():
        ax.scatter(r.longitude, r.latitude, s=sizes[r.name_cn], color=ORANGE,
                   edgecolor="white", linewidth=0.7, zorder=3)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Edge selection frequency")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(meta.longitude.min() - 0.45, meta.longitude.max() + 0.48)
    ax.set_ylim(meta.latitude.min() - 0.35, meta.latitude.max() + 0.35)
    ax.set_aspect(1 / math.cos(math.radians(meta.latitude.mean())))
    ax.grid(color="#EAEAEA", linewidth=0.45, zorder=0)
    fig.text(0.10, 0.20,
             "Node positions follow the station coordinates in Fig. 1; node size denotes frequency-weighted incidence.",
             ha="left", va="center", fontsize=6.2, color=BLACK)
    save(fig, "Fig3_dynamic_high_frequency_spatial_connections")


def load_predictions() -> pd.DataFrame:
    p = pd.read_csv(PRED)
    p = p[p["split"] == "test"].copy()
    p["date"] = pd.to_datetime(p["date"])
    return p.rename(columns={"node": "station_index"})


def prediction_series(pred: pd.DataFrame, meta: pd.DataFrame) -> None:
    selected = ["Suining", "Gaoyou", "Nantong", "Kunshan"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for k, (ax, name) in enumerate(zip(axes.flat, selected)):
        idx = int(meta.loc[meta.name_en == name, "node"].iloc[0])
        d = pred[pred.station_index == idx].sort_values("date")
        rmse = np.sqrt(np.mean((d.y_pred - d.y_true) ** 2))
        ax.plot(d.date, d.y_true, color=BLACK, lw=0.9, label="Observed")
        ax.plot(d.date, d.y_pred, color=ORANGE, lw=0.85, alpha=0.90, label="Predicted")
        ax.set_title(name, loc="center")
        ax.text(0.98, 0.96, f"RMSE = {rmse:.2f} °C", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.1)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        if k % 2 == 0:
            ax.set_ylabel("Daily mean temperature (°C)")
        panel(ax, chr(97 + k))
    fig.legend(handles=[Line2D([0], [0], color=BLACK, label="Observed"),
                        Line2D([0], [0], color=ORANGE, label="ADGL-LightMICN")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.025))
    save(fig, "Fig4_observed_and_predicted_temperature")


def benchmark() -> None:
    d = pd.read_csv(DATA / "final_baseline_metrics.csv").sort_values("rmse", ascending=False)
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.3), sharey=True,
                             gridspec_kw={"wspace": 0.08}, constrained_layout=True)
    colors = [ORANGE if m == "adgl_lightmicn" else (GREY if m == "persistence" else BLUE)
              for m in d.model]
    for k, (ax, metric, title, fmt) in enumerate(zip(
            axes, ["mae", "rmse", "r2"],
            ["MAE", "RMSE", "R²"],
            ["{:.3f}", "{:.3f}", "{:.4f}"])):
        vals = d[metric].to_numpy()
        if metric != "r2":
            ax.barh(y, vals, height=0.62, color=colors)
            ax.set_xlim(0, max(vals) * 1.18)
            for yi, val in zip(y, vals):
                ax.text(val + max(vals) * 0.018, yi, fmt.format(val), va="center", fontsize=5.7)
            ax.set_xlabel(f"{title} (°C)")
        else:
            xmin = min(vals) - 0.003
            ax.hlines(y, xmin, vals, color=LIGHT_GREY, lw=1.2)
            ax.scatter(vals, y, c=colors, s=22, zorder=3)
            ax.set_xlim(xmin, max(vals) + 0.004)
            ax.set_xlabel(title)
        ax.set_title(title, loc="center")
        ax.text(0.0, 1.06, chr(97 + k), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5, fontweight="bold",
                clip_on=False)
    axes[0].set_yticks(y, d.label)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE,
                      markeredgecolor="none", label="ADGL-LightMICN"),
               Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE,
                      markeredgecolor="none", label="Forecasting baselines"),
               Line2D([0], [0], marker="s", color="none", markerfacecolor=GREY,
                      markeredgecolor="none", label="Persistence")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    save(fig, "Fig5_benchmark_performance")


def architecture_and_ablation() -> None:
    match = pd.read_csv(DATA / "architecture_matching_3x2.csv")
    temporal = pd.read_csv(DATA / "lightmicn_internal_ablation.csv")
    graph = pd.read_csv(DATA / "dynamic_graph_component_ablation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.45), constrained_layout=True)

    ax = axes[0]
    order = ["No spatial propagation", "Fixed-graph GCN", "ADGL-GCN"]
    short = ["No spatial", "Fixed GCN", "ADGL-GCN"]
    x = np.arange(3)
    for enc, color, marker in [("TCN", BLUE, "o"), ("LightMICN", ORANGE, "s")]:
        s = match[match.temporal_encoder == enc].set_index("spatial_representation").loc[order]
        ax.plot(x, s.rmse, color=color, marker=marker, ms=4.6, lw=1.2, label=enc)
    ax.set_xticks(x, short, rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Test RMSE (°C)")
    ax.set_ylim(1.73, 1.90)
    ax.set_title("Architecture matching", loc="left")
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "a")

    def delta_panel(ax, d, title, labels):
        full = float(d.loc[d.configuration == "ADGL-LightMICN", "rmse"].iloc[0])
        dd = d.copy()
        dd["delta_pct"] = (dd.rmse / full - 1) * 100
        dd["label"] = dd.configuration.map(labels)
        dd = dd.sort_values("delta_pct", ascending=True)
        yy = np.arange(len(dd))
        cc = [ORANGE if c == "ADGL-LightMICN" else BLUE for c in dd.configuration]
        ax.barh(yy, dd.delta_pct, color=cc, height=0.58)
        ax.set_yticks(yy, dd.label)
        ax.set_xlabel("RMSE increase vs. full model (%)")
        ax.set_title(title, loc="left")
        ax.axvline(0, color=BLACK, lw=0.7)

    delta_panel(axes[1], temporal, "LightMICN ablation", {
        "ADGL-LightMICN": "Full model", "Single scale (2)": "Single scale",
        "No decomposition": "No decomposition"})
    panel(axes[1], "b")
    delta_panel(axes[2], graph, "Dynamic-graph ablation", {
        "ADGL-LightMICN": "Full model", "No Top-k": "No Top-k",
        "No L1": "No L1", "Fixed graph": "Fixed graph",
        "Node embedding": "Node embedding"})
    panel(axes[2], "c")
    # These panels intentionally use different coordinate systems and label widths.
    save(fig, "Fig6_architecture_matching_and_ablation")


def supplementary_station_error() -> None:
    d = pd.read_csv(DATA / "station_performance.csv").sort_values("rmse", ascending=True)
    y = np.arange(len(d))
    overall = 1.7616789545944471
    colors = [BLUE if v <= overall else GOLD for v in d.rmse]
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.barh(y, d.rmse, color=colors, height=0.62)
    ax.axvline(overall, color=ORANGE, ls="--", lw=1.0,
               label=f"Overall RMSE = {overall:.2f} °C")
    ax.set_yticks(y, d.station)
    ax.set_xlabel("Station-level RMSE (°C)")
    ax.set_xlim(0, max(d.rmse) * 1.17)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "FigS1_station_level_rmse")


def station_comparative_diagnostics(pred: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Compare the full model and persistence at every station."""
    persistence = pd.read_csv(PERSISTENCE_PRED)

    def metrics(d: pd.DataFrame):
        err = d.y_pred.to_numpy() - d.y_true.to_numpy()
        truth = d.y_true.to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = float(np.sum((truth - truth.mean()) ** 2))
        r2 = float(1 - np.sum(err ** 2) / denom)
        return mae, rmse, r2

    rows = []
    for node, d_model in pred.groupby("station_index"):
        rec = meta.loc[meta.node == node].iloc[0]
        d_pers = persistence[persistence.station_id == int(rec.sid)]
        mae_m, rmse_m, r2_m = metrics(d_model)
        mae_p, rmse_p, r2_p = metrics(d_pers)
        rows.append({
            "station": rec.name_en,
            "station_id": int(rec.sid),
            "mae_model": mae_m, "rmse_model": rmse_m, "r2_model": r2_m,
            "mae_persistence": mae_p, "rmse_persistence": rmse_p,
            "r2_persistence": r2_p,
            "rmse_reduction_pct": 100 * (rmse_p - rmse_m) / rmse_p,
        })
    d = pd.DataFrame(rows).sort_values("rmse_model", ascending=True).reset_index(drop=True)
    d.to_csv(DATA / "station_model_vs_persistence.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), sharey=True,
                             constrained_layout=True)
    y = np.arange(len(d))
    h = 0.34
    panels = [
        (axes[0, 0], "mae_model", "mae_persistence", "Station-level MAE", "MAE (°C)"),
        (axes[0, 1], "rmse_model", "rmse_persistence", "Station-level RMSE", "RMSE (°C)"),
    ]
    for ax, model_col, pers_col, title, xlabel in panels:
        ax.barh(y - h / 2, d[model_col], height=h, color=ORANGE,
                label="ADGL-LightMICN")
        ax.barh(y + h / 2, d[pers_col], height=h, color=LIGHT_BLUE,
                label="Persistence")
        ax.set_title(title, loc="center")
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.8, len(d) - 0.2)

    ax = axes[1, 0]
    for yi, a, b in zip(y, d.r2_model, d.r2_persistence):
        ax.plot([a, b], [yi, yi], color=LIGHT_GREY, lw=1.2, zorder=1)
    ax.scatter(d.r2_model, y, s=20, color=ORANGE, label="ADGL-LightMICN", zorder=3)
    ax.scatter(d.r2_persistence, y, s=20, color=LIGHT_BLUE,
               label="Persistence", zorder=3)
    ax.set_title("Station-level R²", loc="center")
    ax.set_xlabel("R²")
    ax.set_xlim(min(d.r2_persistence.min(), d.r2_model.min()) - 0.003,
                max(d.r2_persistence.max(), d.r2_model.max()) + 0.003)

    ax = axes[1, 1]
    ax.barh(y, d.rmse_reduction_pct, height=0.60, color=BLUE)
    overall_gain = 100 * (2.014772483157517 - 1.7616789545944471) / 2.014772483157517
    ax.axvline(overall_gain, color=ORANGE, ls="--", lw=1.0)
    ax.set_title("RMSE reduction over persistence", loc="center")
    ax.set_xlabel("RMSE reduction (%)")
    ax.set_xlim(0, max(d.rmse_reduction_pct) * 1.12)
    ax.text(0.98, 0.02, f"Overall: {overall_gain:.2f}%", transform=ax.transAxes,
            ha="right", va="bottom", color=ORANGE, fontsize=6.2)

    axes[0, 0].invert_yaxis()
    for k, ax in enumerate(axes.flat):
        ax.text(0.0, 1.04, chr(97 + k), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5,
                fontweight="bold", clip_on=False)
    axes[0, 0].set_yticks(y, d.station)
    axes[1, 0].set_yticks(y, d.station)
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE,
                      markeredgecolor="none", label="ADGL-LightMICN"),
               Line2D([0], [0], marker="s", color="none", markerfacecolor=LIGHT_BLUE,
                      markeredgecolor="none", label="Persistence")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    save(fig, "Fig6_station_level_comparative_diagnostics")


def write_notes() -> None:
    text = """# Scientific Reports 图形正式版\n\n

## 主图结构\n\n
1. `Fig1_station_network`：站点空间分布。图中加入指北针和按经纬度换算的 100 km 比例尺。\n
2. `Fig2_input_conditioned_graphs`：三类真实温度条件下的输入条件邻接矩阵。\n
3. `Fig3_dynamic_high_frequency_spatial_connections`：按真实经纬度绘制的高频动态图连接；箭头方向遵循 A_ij 的来源站点 j 至接收站点 i，线宽和蓝色深浅表示边选择频率，节点大小表示展示边集合中的频率加权关联度。\n
4. `Fig4_observed_and_predicted_temperature`：四个地理分区代表站点在完整测试时段的观测与预测轨迹。\n
5. `Fig5_benchmark_performance`：用 Scientific Reports 常见的三面板横向比较形式分别展示 MAE、RMSE 和 R²，完整模型统一用橙色突出。\n
6. `Fig6_architecture_matching_and_ablation`：把 3×2 架构匹配、LightMICN 内部消融和动态图组件消融合并为同一证据图。\n
7. `FigS1_station_level_rmse`：站点级 RMSE，建议作为补充图。\n\n
## 视觉依据\n\n
- Scientific Reports 投稿图形要求：https://www.nature.com/srep/author-instructions/submission-guidelines\n
- 预测轨迹的黑/橙折线与紧凑多面板形式参考同刊气象预测论文：https://www.nature.com/articles/s41598-026-55817-x\n
- MAE、RMSE、R²分面比较的组织方式参考同刊误差比较图：https://www.nature.com/articles/s41598-025-99109-2\n
- 环境数据空间—时间证据组合参考：https://www.nature.com/articles/s41598-020-79148-7\n
- 站点地图的制图要素参考：https://www.nature.com/articles/s41598-025-07805-w\n\n

## 数据边界\n\n
所有数值来自冻结预测文件、已核实指标表或从冻结权重提取的动态图矩阵。未生成虚构数据。动态条件图对应 2026-01-21、2025-05-09 和 2025-07-06。\n"""
    (OUT / "README_图形设计与数据边界.md").write_text(text, encoding="utf-8")


def main() -> None:
    style()
    copy_source_data()
    meta = pd.read_csv(DATA / "station_metadata.csv")
    station_map(meta)
    dynamic_graph(meta)
    dynamic_edge_map(meta)
    prediction_series(load_predictions(), meta)
    benchmark()
    architecture_and_ablation()
    station_comparative_diagnostics(load_predictions(), meta)
    supplementary_station_error()
    write_notes()
    print(OUT)


if __name__ == "__main__":
    main()
