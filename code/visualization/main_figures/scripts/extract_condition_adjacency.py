"""Extract three real test-window adjacency matrices from the frozen ADGL-LightMICN checkpoint.

No training or parameter update is performed. Windows are selected by the mean
observed target temperature across the 15 stations: minimum, nearest median,
and maximum.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / "source_data"
RUN_DIR = Path(__file__).resolve().parents[3] / "checkpoints" / "error_dynamic_graph" / "runs" / "w12_k2-4_lr0p0010_layers1"
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
XLSX = PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
ADJ_CSV = PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv"

sys.path.insert(0, str(PACKAGE_ROOT))

from exp.error_dynamic_graph.workflow_utils import align_to_reference_dates, COMMON_WINDOW, configure_device  # noqa: E402


def station_names():
    frame = pd.read_excel(XLSX, usecols=[0, 2])
    frame.columns = ["sid", "name"]
    frame = frame[frame["sid"].astype(str) != "sid"]
    pairs = dict(zip(frame["sid"].astype(str), frame["name"].astype(str)))
    ids = sorted(pairs, key=lambda s: int(s))
    return ids, [pairs[i] for i in ids]


def main():
    configure_device("cpu")
    from models.adgl_lightmicn import ModelConfig
    from utils.training import set_seed, _inputs, _predict
    from data_provider.data_loader import load_common_xlsx
    from exp.error_dynamic_graph.variants import build_model

    OUT.mkdir(parents=True, exist_ok=True)
    from utils.protocol import MODEL_CONFIG
    cfg = MODEL_CONFIG
    set_seed(int(cfg["seed"]))

    bundle = load_common_xlsx(XLSX, ADJ_CSV, seq_len=int(cfg["seq_len"]), allow_positional_adj=True)
    reference = load_common_xlsx(XLSX, ADJ_CSV, seq_len=COMMON_WINDOW, allow_positional_adj=True)
    align_to_reference_dates(bundle, reference, splits=("val", "test"))

    values = vars(SimpleNamespace(**cfg))
    model_cfg = ModelConfig(
        **{f.name: values[f.name] for f in fields(ModelConfig) if f.name in values},
        num_nodes=bundle.num_nodes,
    )
    model = build_model(model_cfg, bundle.adj, "full")
    model([x[:1] for x in _inputs(bundle, "test", target_calendar=True)], training=False)
    model.load_weights(str(RUN_DIR / "best.weights.h5"))
    _, details = _predict(model, bundle, "test", return_adgl=True)
    adjacency = details["adjacency"].astype(np.float64)

    _, names = station_names()
    labels = names[: adjacency.shape[1]]
    mean_target = bundle.y_raw_test.astype(np.float64).mean(axis=1)
    median_value = float(np.median(mean_target))
    indices = {
        "lowest": int(np.argmin(mean_target)),
        "median": int(np.argmin(np.abs(mean_target - median_value))),
        "highest": int(np.argmax(mean_target)),
    }

    meta = []
    for condition, idx in indices.items():
        pd.DataFrame(adjacency[idx], index=labels, columns=labels).to_csv(
            OUT / f"adjacency_{condition}.csv", encoding="utf-8-sig"
        )
        meta.append({
            "condition": condition,
            "test_window_index": idx,
            "target_date": str(pd.Timestamp(bundle.date_test[idx]).date()),
            "mean_observed_target_temperature_c": float(mean_target[idx]),
            "minimum_station_temperature_c": float(bundle.y_raw_test[idx].min()),
            "maximum_station_temperature_c": float(bundle.y_raw_test[idx].max()),
        })

    pd.DataFrame(meta).to_csv(OUT / "adjacency_condition_metadata.csv", index=False, encoding="utf-8-sig")
    (OUT / "adjacency_condition_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
