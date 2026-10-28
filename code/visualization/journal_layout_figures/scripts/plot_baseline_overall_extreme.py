"""Baseline performance on all and extreme-temperature test observations.

All metrics are recomputed from frozen per-observation predictions. Extreme
temperature observations use the training-only station-by-month statistics of
the original weighting protocol: low z <= -1.5 and high z >= 1.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
DATA_OUT = OUT / "source_data"

RAW_XLSX = CODE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
EXPECTED_DATA_SHA256 = "bf6b4eeb3278ba995da7c1929a9653d143999924edcdd67eff2492f19055c6d7"

BASELINES = CODE_ROOT / "results" / "comparison"
MODEL_FILES = {
    "ADGL-LightMICN": BASELINES / "adgl_lightmicn/test_predictions.csv",
    "DCRNN": BASELINES / "dcrnn/test_predictions.csv",
    "Graph WaveNet": BASELINES / "graph_wavenet/test_predictions.csv",
    "T-GCN": BASELINES / "tgcn/test_predictions.csv",
    "DLinear": BASELINES / "dlinear/test_predictions.csv",
    "Persistence": BASELINES / "persistence/test_predictions.csv",
    "MTGNN": BASELINES / "mtgnn/test_predictions.csv",
    "TCN": BASELINES / "tcn/test_predictions.csv",
    "LSTM": BASELINES / "lstm/test_predictions.csv",
    "iTransformer": BASELINES / "itransformer/test_predictions.csv",
    "GCN-MLP": BASELINES / "gcn_mlp/test_predictions.csv",
}

BLUE = "#4E97B8"
LIGHT_BLUE = "#91BED2"
ORANGE = "#D8783D"
LIGHT_ORANGE = "#E7AA7F"
GRID = "#E6E6E6"
BLACK = "#1A1A1A"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def load_temperature_reference() -> tuple[pd.DataFrame, dict]:
    if sha256(RAW_XLSX) != EXPECTED_DATA_SHA256:
        raise RuntimeError("Raw workbook hash does not match the frozen experiment configuration")
    raw = pd.read_excel(RAW_XLSX, header=0)
    raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.rename(columns={"站号": "station_id", "年": "year", "月": "month", "日": "day"})
    for col in ["station_id", "year", "month", "day", "平均气温"]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=raw.day))
    pivot = raw.pivot(index="date", columns="station_id", values="平均气温").sort_index()
    station_ids = list(pivot.columns)
    n_days = len(pivot)
    train_end = int(n_days * 0.70)
    train = pivot.iloc[:train_end]
    stats = (train.stack().rename("temperature").reset_index()
             .assign(month=lambda x: x.date.dt.month)
             .groupby(["station_id", "month"], as_index=False)
             .temperature.agg(monthly_mean="mean", monthly_sd=lambda x: x.std(ddof=0)))
    if (stats.monthly_sd <= 1e-6).any():
        raise RuntimeError("A station-month training standard deviation is effectively zero")
    meta = {
        "data_sha256": EXPECTED_DATA_SHA256,
        "n_days": n_days,
        "training_days": train_end,
        "training_date_start": str(pivot.index[0].date()),
        "training_date_end": str(pivot.index[train_end - 1].date()),
        "station_count": len(station_ids),
        "extreme_definition": "signed station-month z-score using training-only mean and population SD",
        "low_rule": "z <= -1.5",
        "high_rule": "z >= 1.5",
    }
    return stats, meta


def standardize_prediction_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "station_id", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Missing columns in {path}: {sorted(required - set(df.columns))}")
    df = df[list(required)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["station_id"] = pd.to_numeric(df["station_id"], errors="raise").astype(int)
    df = df.sort_values(["date", "station_id"]).reset_index(drop=True)
    if df.duplicated(["date", "station_id"]).any():
        raise RuntimeError(f"Duplicate date-station rows in {path}")
    return df


def compute_metrics() -> tuple[pd.DataFrame, dict]:
    stats, meta = load_temperature_reference()
    frames = {name: standardize_prediction_frame(path) for name, path in MODEL_FILES.items()}
    reference = frames["ADGL-LightMICN"][["date", "station_id", "y_true"]].copy()
    reference["month"] = reference.date.dt.month
    reference = reference.merge(stats, on=["station_id", "month"], how="left", validate="many_to_one")
    if reference[["monthly_mean", "monthly_sd"]].isna().any().any():
        raise RuntimeError("Missing training station-month statistics for test observations")
    reference["z"] = (reference.y_true - reference.monthly_mean) / reference.monthly_sd
    reference["subset"] = np.select(
        [reference.z <= -1.5, reference.z >= 1.5], ["low", "high"], default="normal")

    key_hashes = {}
    true_max_abs_diff = {}
    rows = []
    key_ref = reference[["date", "station_id"]]
    y_ref = reference.y_true.to_numpy(float)
    low_mask = reference.subset.eq("low").to_numpy()
    high_mask = reference.subset.eq("high").to_numpy()
    for model, df in frames.items():
        if not df[["date", "station_id"]].equals(key_ref):
            raise RuntimeError(f"Test key order/content differs for {model}")
        true_diff = float(np.max(np.abs(df.y_true.to_numpy(float) - y_ref)))
        if true_diff > 1e-5:
            raise RuntimeError(f"y_true differs for {model}: max abs diff={true_diff}")
        true_max_abs_diff[model] = true_diff
        key_hashes[model] = hashlib.sha256(
            pd.util.hash_pandas_object(df[["date", "station_id"]], index=False).values.tobytes()
        ).hexdigest()
        err = df.y_pred.to_numpy(float) - y_ref
        def summarize(mask: np.ndarray, subset: str) -> None:
            e = err[mask]
            rows.append({
                "model": model,
                "subset": subset,
                "n_station_date_observations": int(mask.sum()),
                "n_unique_dates": int(reference.loc[mask, "date"].nunique()),
                "mae_c": float(np.mean(np.abs(e))),
                "rmse_c": float(np.sqrt(np.mean(e ** 2))),
            })
        summarize(np.ones(len(err), dtype=bool), "overall")
        summarize(low_mask, "low")
        summarize(high_mask, "high")

    meta.update({
        "test_date_start": str(reference.date.min().date()),
        "test_date_end": str(reference.date.max().date()),
        "test_station_date_observations": int(len(reference)),
        "test_unique_dates": int(reference.date.nunique()),
        "low_station_date_observations": int(low_mask.sum()),
        "low_unique_dates": int(reference.loc[low_mask, "date"].nunique()),
        "high_station_date_observations": int(high_mask.sum()),
        "high_unique_dates": int(reference.loc[high_mask, "date"].nunique()),
        "prediction_key_sha256": key_hashes["ADGL-LightMICN"],
        "all_prediction_key_hashes_equal": len(set(key_hashes.values())) == 1,
        "max_y_true_difference_by_model": true_max_abs_diff,
        "uncertainty": "not shown; one frozen seed per model",
    })
    if not meta["all_prediction_key_hashes_equal"]:
        raise RuntimeError("Prediction key hashes differ across models")
    return pd.DataFrame(rows), meta


def plot(metrics: pd.DataFrame, meta: dict) -> None:
    # Preserve one order across all panels: lowest overall RMSE first.
    order = (metrics.loc[metrics.subset.eq("overall")]
             .sort_values("rmse_c").model.tolist())
    y = np.arange(len(order))
    proposed = np.array([m == "ADGL-LightMICN" for m in order])
    base_colors = np.where(proposed, ORANGE, BLUE)
    low_colors = np.where(proposed, ORANGE, BLUE)
    high_colors = np.where(proposed, LIGHT_ORANGE, LIGHT_BLUE)

    wide = metrics.pivot(index="model", columns="subset", values=["mae_c", "rmse_c"])
    overall_mae = wide.loc[order, ("mae_c", "overall")].to_numpy()
    low_mae = wide.loc[order, ("mae_c", "low")].to_numpy()
    high_mae = wide.loc[order, ("mae_c", "high")].to_numpy()
    overall_rmse = wide.loc[order, ("rmse_c", "overall")].to_numpy()
    low_rmse = wide.loc[order, ("rmse_c", "low")].to_numpy()
    high_rmse = wide.loc[order, ("rmse_c", "high")].to_numpy()

    mae_max = max(overall_mae.max(), low_mae.max(), high_mae.max()) * 1.08
    rmse_max = max(overall_rmse.max(), low_rmse.max(), high_rmse.max()) * 1.08

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 7.0), constrained_layout=True)
    axa, axb, axc, axd = axes.ravel()

    def common(ax, label, title, xlabel, xmax):
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=7, fontweight="bold")
        ax.xaxis.grid(True, color=GRID, lw=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.text(-0.18, 1.045, label, transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    axa.barh(y, overall_mae, height=0.66, color=base_colors, edgecolor="none")
    common(axa, "a", "Overall test set", "MAE (°C)", mae_max)

    h = 0.32
    axb.barh(y - h/2, low_mae, height=h, color=low_colors, edgecolor="none")
    axb.barh(y + h/2, high_mae, height=h, color=high_colors,
             edgecolor="none")
    common(axb, "b", "Extreme-temperature subset", "MAE (°C)", mae_max)

    axc.barh(y, overall_rmse, height=0.66, color=base_colors, edgecolor="none")
    common(axc, "c", "Overall test set", "RMSE (°C)", rmse_max)

    axd.barh(y - h/2, low_rmse, height=h, color=low_colors, edgecolor="none")
    axd.barh(y + h/2, high_rmse, height=h, color=high_colors,
             edgecolor="none")
    common(axd, "d", "Extreme-temperature subset", "RMSE (°C)", rmse_max)

    handles = [
        Patch(facecolor=BLUE, edgecolor="none", label="Baselines"),
        Patch(facecolor=ORANGE, edgecolor="none", label="ADGL-LightMICN"),
        Patch(facecolor="#777777", edgecolor="none",
              label=f"Low temperature (dark; n={meta['low_station_date_observations']:,})"),
        Patch(facecolor="#BBBBBB", edgecolor="none",
              label=f"High temperature (light; n={meta['high_station_date_observations']:,})"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4,
               handlelength=1.5, columnspacing=1.5, borderaxespad=0.2)

    stem = "Fig7_baseline_overall_and_extreme_temperature"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_2x3(metrics: pd.DataFrame, meta: dict) -> None:
    """Alternative six-panel layout: overall, high and low in separate columns."""
    order = (metrics.loc[metrics.subset.eq("overall")]
             .sort_values("rmse_c").model.tolist())
    y = np.arange(len(order))
    proposed = np.array([m == "ADGL-LightMICN" for m in order])
    colors = np.where(proposed, ORANGE, BLUE)
    wide = metrics.pivot(index="model", columns="subset", values=["mae_c", "rmse_c"])

    values = {
        ("mae_c", "overall"): wide.loc[order, ("mae_c", "overall")].to_numpy(),
        ("mae_c", "high"): wide.loc[order, ("mae_c", "high")].to_numpy(),
        ("mae_c", "low"): wide.loc[order, ("mae_c", "low")].to_numpy(),
        ("rmse_c", "overall"): wide.loc[order, ("rmse_c", "overall")].to_numpy(),
        ("rmse_c", "high"): wide.loc[order, ("rmse_c", "high")].to_numpy(),
        ("rmse_c", "low"): wide.loc[order, ("rmse_c", "low")].to_numpy(),
    }
    mae_max = max(values[("mae_c", s)].max() for s in ("overall", "high", "low")) * 1.08
    rmse_max = max(values[("rmse_c", s)].max() for s in ("overall", "high", "low")) * 1.08

    fig, axes = plt.subplots(2, 3, figsize=(7.35, 6.75), sharey=True,
                             constrained_layout=True)
    columns = [
        ("overall", "Overall test set", meta["test_station_date_observations"]),
        ("high", "High-temperature samples", meta["high_station_date_observations"]),
        ("low", "Low-temperature samples", meta["low_station_date_observations"]),
    ]
    letters = iter("abcdef")
    for row, (metric, xlabel, xmax) in enumerate([
        ("mae_c", "MAE (°C)", mae_max),
        ("rmse_c", "RMSE (°C)", rmse_max),
    ]):
        for col, (subset, title, n_obs) in enumerate(columns):
            ax = axes[row, col]
            ax.barh(y, values[(metric, subset)], height=0.66,
                    color=colors, edgecolor="none")
            ax.set_yticks(y, order)
            ax.set_xlim(0, xmax)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{title}\n(n={n_obs:,})", pad=6, fontweight="bold")
            ax.xaxis.grid(True, color=GRID, lw=0.55)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0, pad=4,
                           labelleft=(col == 0))
            ax.text(-0.20, 1.045, next(letters), transform=ax.transAxes,
                    fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    # All six panels share the same y-axis; invert it exactly once so the best
    # overall-RMSE model is placed at the top in every panel.
    axes[0, 0].invert_yaxis()

    handles = [
        Patch(facecolor=BLUE, edgecolor="none", label="Baselines"),
        Patch(facecolor=ORANGE, edgecolor="none", label="ADGL-LightMICN"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               handlelength=1.5, columnspacing=1.6, borderaxespad=0.2)

    stem = "Fig7_alt_baseline_overall_high_low_2x3"
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alt-only", action="store_true",
                        help="Generate only the 2x3 alternative layout")
    args = parser.parse_args()
    set_style()
    metrics, meta = compute_metrics()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_OUT / "baseline_overall_extreme_metrics.csv", index=False,
                   encoding="utf-8-sig")
    (DATA_OUT / "baseline_overall_extreme_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.alt_only:
        plot(metrics, meta)
    plot_2x3(metrics, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
