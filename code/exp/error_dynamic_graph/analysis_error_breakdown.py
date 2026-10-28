"""Error breakdown for the frozen MICN: per-station, per-season, per-month, per-temperature.

Uses ONLY the already-saved predictions (no retraining, no GPU).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
PRED_CSV = PACKAGE_ROOT / "results" / "error_dynamic_graph" / "runs" / "w12_k2-4_lr0p0010_layers1" / "predictions_full_both.csv"
PACKAGE_ROOT = ROOT.parents[1]
XLSX = PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
OUT = PACKAGE_ROOT / "results" / "error_dynamic_graph" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEASON_OF = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


def station_names():
    frame = pd.read_excel(XLSX, usecols=[0, 2])
    frame.columns = ["sid", "name"]
    frame = frame[frame["sid"].astype(str) != "sid"]
    pairs = dict(zip(frame["sid"].astype(str), frame["name"].astype(str)))
    ids = sorted(pairs, key=lambda s: int(s) if str(s).isdigit() else 0)
    return ids, [pairs[i] for i in ids]


def metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse


def main():
    ids, names = station_names()
    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(SEASON_OF)
    print("rows:", len(df), "stations:", len(names), "dates(val+test):", df["date"].nunique())

    summary = {"config_id": "w12_k2-4_lr0p0010_layers1", "split_info": {}}

    for split in ["test", "val"]:
        sub = df[df["split"] == split].copy()
        summary["split_info"][split] = {
            "rows": int(len(sub)),
            "dates": int(sub["date"].nunique()),
            "date_min": str(sub["date"].min().date()),
            "date_max": str(sub["date"].max().date()),
            "overall_mae": float(np.mean(np.abs(sub["residual"]))),
            "overall_rmse": float(np.sqrt(np.mean(sub["residual"] ** 2))),
        }
    summary["date_ranges"] = {"test": summary["split_info"]["test"]}

    test = df[df["split"] == "test"].copy()
    test["station"] = test["node"].map(lambda i: names[i] if 0 <= i < len(names) else str(i))

    # ---- per station ----
    rows = []
    for node, group in test.groupby("node"):
        mae, rmse = metrics(group["residual"])
        ss_res = float(np.sum(group["residual"] ** 2))
        ss_tot = float(np.sum((group["y_true"] - group["y_true"].mean()) ** 2))
        rows.append({"node": int(node), "station": names[int(node)] if int(node) < len(names) else str(node),
                     "n": int(len(group)), "mae": mae, "rmse": rmse,
                     "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")})
    per_station = pd.DataFrame(rows).sort_values("rmse")
    per_station.to_csv(OUT / "error_by_station.csv", index=False, encoding="utf-8-sig")

    # ---- per season ----
    rows = []
    for season in SEASON_ORDER:
        group = test[test["season"] == season]
        if not len(group):
            continue
        mae, rmse = metrics(group["residual"])
        rows.append({"season": season, "n": int(len(group)), "mae": mae, "rmse": rmse})
    per_season = pd.DataFrame(rows)
    per_season.to_csv(OUT / "error_by_season.csv", index=False, encoding="utf-8-sig")

    # ---- per month ----
    rows = []
    for month, group in test.groupby("month"):
        mae, rmse = metrics(group["residual"])
        rows.append({"month": int(month), "n": int(len(group)), "mae": mae, "rmse": rmse})
    per_month = pd.DataFrame(rows).sort_values("month")
    per_month.to_csv(OUT / "error_by_month.csv", index=False, encoding="utf-8-sig")

    # ---- per station x season (RMSE) ----
    pivot = (test.groupby(["station", "season"])["residual"]
             .apply(lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2))))
             .unstack("season"))
    pivot = pivot.reindex(columns=[s for s in SEASON_ORDER if s in pivot.columns])
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot.to_csv(OUT / "error_by_station_season_rmse.csv", encoding="utf-8-sig")

    # ---- error vs true temperature level ----
    bins = [-100, 0, 5, 10, 15, 20, 25, 30, 100]
    labels = ["<0", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"]
    test["temp_bin"] = pd.cut(test["y_true"], bins=bins, labels=labels)
    rows = []
    for label, group in test.groupby("temp_bin", observed=True):
        mae, rmse = metrics(group["residual"])
        rows.append({"temp_bin": str(label), "n": int(len(group)), "mae": mae, "rmse": rmse,
                     "mean_true": float(group["y_true"].mean())})
    per_temp = pd.DataFrame(rows)
    per_temp.to_csv(OUT / "error_by_temperature.csv", index=False, encoding="utf-8-sig")

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(per_station["station"], per_station["rmse"], color="#3b6ea5")
    ax.axhline(summary["split_info"]["test"]["overall_rmse"], color="crimson", ls="--",
               label="overall RMSE = %.3f" % summary["split_info"]["test"]["overall_rmse"])
    ax.set_ylabel("Test RMSE (degC)")
    ax.set_title("Per-station test RMSE (frozen MICN)")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "error_by_station.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(per_season["season"], per_season["rmse"], color="#4c9f70")
    for i, row in per_season.reset_index().iterrows():
        ax.text(i, row["rmse"], "%.3f" % row["rmse"], ha="center", va="bottom")
    ax.set_ylabel("Test RMSE (degC)")
    ax.set_title("Per-season test RMSE (frozen MICN)")
    fig.tight_layout()
    fig.savefig(OUT / "error_by_season.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_xlabel("Season")
    ax.set_ylabel("Station")
    ax.set_title("Test RMSE by station x season")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, "%.2f" % value, ha="center", va="center",
                        color="white" if value < np.nanmean(pivot.values) else "black", fontsize=7)
    fig.colorbar(image, ax=ax, label="RMSE")
    fig.tight_layout()
    fig.savefig(OUT / "error_heatmap_station_season.png", dpi=200)
    plt.close(fig)

    # ---- textual takeaways ----
    best = per_station.iloc[0]
    worst = per_station.iloc[-1]
    best_season = per_season.sort_values("rmse").iloc[0]
    worst_season = per_season.sort_values("rmse").iloc[-1]
    summary["findings"] = {
        "best_station": f"{best['station']} (RMSE {best['rmse']:.3f})",
        "worst_station": f"{worst['station']} (RMSE {worst['rmse']:.3f})",
        "best_season": f"{best_season['season']} (RMSE {best_season['rmse']:.3f})",
        "worst_season": f"{worst_season['season']} (RMSE {worst_season['rmse']:.3f})",
        "station_rmse_spread": float(per_station["rmse"].max() - per_station["rmse"].min()),
        "season_rmse_spread": float(per_season["rmse"].max() - per_season["rmse"].min()),
    }
    (OUT / "error_breakdown_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["findings"], ensure_ascii=False, indent=2))
    print("[written] to", OUT)
    print(per_station.to_string(index=False))
    print(per_season.to_string(index=False))


if __name__ == "__main__":
    main()
