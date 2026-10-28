"""Dataset split overview for the ADGL-LightMICN manuscript.

Open-source structural references:
- Matplotlib boxplot example and percentile-whisker API (Matplotlib license):
  https://github.com/matplotlib/matplotlib/blob/main/galleries/examples/statistics/boxplot_demo.py
- skforecast time-series interval shading example (BSD-3-Clause):
  https://github.com/skforecast/skforecast/blob/main/docs/user_guides/weighted-time-series-forecasting.ipynb

Only the plot structures are adapted. All values come from the frozen project
workbook and the chronological 70%/10%/20% split used by the experiments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[3]
RAW_XLSX = CODE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
EXPECTED_SHA256 = "bf6b4eeb3278ba995da7c1929a9653d143999924edcdd67eff2492f19055c6d7"

OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
DATA_OUT = OUT / "source_data"

BLUE = "#2F6FA3"
TRAIN = "#DCEBF3"
VALIDATION = "#F5E7C7"
TEST = "#DCEBDD"
ORANGE = "#D8783D"
BLACK = "#1A1A1A"
GRID = "#E6E6E6"


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    actual_hash = sha256(RAW_XLSX)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError("Raw workbook hash does not match the frozen experiment configuration")

    raw = pd.read_excel(RAW_XLSX, header=0)
    raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.rename(columns={"站号": "station_id", "年": "year",
                              "月": "month", "日": "day"})
    required = ["station_id", "year", "month", "day", "平均气温"]
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    raw["station_id"] = raw.station_id.astype(int)
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=raw.day))
    if raw.duplicated(["date", "station_id"]).any():
        raise RuntimeError("Duplicate station-date observations detected")

    pivot = raw.pivot(index="date", columns="station_id", values="平均气温").sort_index()
    if pivot.isna().any().any():
        raise RuntimeError("Missing station-date temperatures detected")
    if not np.all(np.diff(pivot.index.values).astype("timedelta64[D]") == np.timedelta64(1, "D")):
        raise RuntimeError("The daily timeline is not continuous")

    n_days = len(pivot)
    train_end = int(n_days * 0.70)
    val_end = int(n_days * 0.80)
    boundaries = {
        "Train": (0, train_end, TRAIN),
        "Validation": (train_end, val_end, VALIDATION),
        "Test": (val_end, n_days, TEST),
    }

    long = pivot.stack().rename("temperature_c").reset_index()
    long.columns = ["date", "station_id", "temperature_c"]
    split_by_date = pd.Series(index=pivot.index, dtype="object")
    for split, (start, end, _) in boundaries.items():
        split_by_date.iloc[start:end] = split
    long["split"] = long.date.map(split_by_date)
    daily = pivot.mean(axis=1).rename("mean_temperature_c").reset_index()
    daily["split"] = daily.date.map(split_by_date)

    summaries = []
    for split, (start, end, _) in boundaries.items():
        values = long.loc[long.split.eq(split), "temperature_c"].to_numpy(float)
        summaries.append({
            "split": split,
            "start_date": str(pivot.index[start].date()),
            "end_date": str(pivot.index[end - 1].date()),
            "n_days": int(end - start),
            "n_station_date_observations": int(values.size),
            "mean_c": float(np.mean(values)),
            "median_c": float(np.median(values)),
            "q1_c": float(np.quantile(values, 0.25)),
            "q3_c": float(np.quantile(values, 0.75)),
            "p05_c": float(np.quantile(values, 0.05)),
            "p95_c": float(np.quantile(values, 0.95)),
            "minimum_c": float(np.min(values)),
            "maximum_c": float(np.max(values)),
        })
    summary = pd.DataFrame(summaries)
    metadata = {
        "data_sha256": actual_hash,
        "station_count": int(pivot.shape[1]),
        "date_start": str(pivot.index.min().date()),
        "date_end": str(pivot.index.max().date()),
        "n_days": int(n_days),
        "split_rule": "chronological 70% train, 10% validation, 20% test",
        "panel_a_unit": "station-date daily mean temperature observations",
        "panel_b_unit": "daily mean across 15 stations",
        "box_whiskers": "5th and 95th percentiles",
        "unused_period": "none; all raw dates belong to one chronological split",
    }
    return long, daily, summary, metadata


def draw(long: pd.DataFrame, daily: pd.DataFrame,
         summary: pd.DataFrame, metadata: dict) -> None:
    order = ["Train", "Validation", "Test"]
    colors = [TRAIN, VALIDATION, TEST]
    groups = [long.loc[long.split.eq(s), "temperature_c"].to_numpy(float) for s in order]

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.35),
                             gridspec_kw={"width_ratios": [0.92, 1.55]},
                             constrained_layout=True)
    axa, axb = axes

    bp = axa.boxplot(
        groups, tick_labels=order, patch_artist=True, showfliers=False,
        whis=(5, 95), widths=0.52,
        medianprops={"color": ORANGE, "linewidth": 1.25},
        boxprops={"color": BLACK, "linewidth": 0.7},
        whiskerprops={"color": BLACK, "linewidth": 0.7},
        capprops={"color": BLACK, "linewidth": 0.7})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axa.set_title("Distribution by chronological split", fontweight="bold")
    axa.set_ylabel("Station-level daily mean temperature (°C)")
    axa.yaxis.grid(True, color=GRID, lw=0.55)
    axa.set_axisbelow(True)
    axa.text(-0.16, 1.04, "a", transform=axa.transAxes, fontsize=9.5,
             fontweight="bold", ha="left", va="bottom")

    dates = pd.to_datetime(daily.date)
    axb.plot(dates, daily.mean_temperature_c, color=BLUE, lw=0.75, zorder=2)
    for split, color in zip(order, colors):
        rec = summary.loc[summary.split.eq(split)].iloc[0]
        start = pd.Timestamp(rec.start_date)
        end = pd.Timestamp(rec.end_date) + pd.Timedelta(days=1)
        axb.axvspan(start, end, color=color, alpha=0.62, lw=0, zorder=0)
    axb.set_title("Daily mean temperature across 15 stations", fontweight="bold")
    axb.set_ylabel("Mean temperature (°C)")
    axb.set_xlabel("Date")
    axb.xaxis.set_major_locator(mdates.YearLocator(2))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axb.margins(x=0)
    axb.yaxis.grid(True, color=GRID, lw=0.55)
    axb.set_axisbelow(True)
    axb.text(-0.12, 1.04, "b", transform=axb.transAxes, fontsize=9.5,
             fontweight="bold", ha="left", va="bottom")
    handles = [Patch(facecolor=c, edgecolor="none", label=s)
               for s, c in zip(order, colors)]
    axb.legend(handles=handles, loc="upper right", ncol=3,
               handlelength=1.25, columnspacing=0.9)

    stem = "Fig_dataset_chronological_split_overview"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    long, daily, summary, metadata = load_data()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DATA_OUT / "dataset_daily_15station_mean_by_split.csv",
                 index=False, encoding="utf-8-sig")
    long.to_csv(DATA_OUT / "dataset_station_date_temperature_by_split.csv",
                index=False, encoding="utf-8-sig")
    summary.to_csv(DATA_OUT / "dataset_split_temperature_summary.csv",
                   index=False, encoding="utf-8-sig")
    (DATA_OUT / "dataset_split_overview_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    draw(long, daily, summary, metadata)
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
