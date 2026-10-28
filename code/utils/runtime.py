"""Data alignment, device, reproducibility and output helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.data_loader import load_common_xlsx
from utils.protocol import ADJ_CSV, COMMON_WINDOW, DATA_XLSX, SEED, SEQ_LEN


def configure_tensorflow(device: str = "auto"):
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if device == "gpu" and not gpus:
        raise RuntimeError("--device gpu was requested but TensorFlow sees no GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    return tf, [d.name for d in gpus]


def _indices_for_dates(source_dates, reference_dates):
    source = np.asarray(source_dates).astype("datetime64[D]")
    reference = np.asarray(reference_dates).astype("datetime64[D]")
    mapping = {value: i for i, value in enumerate(source)}
    missing = [str(value) for value in reference if value not in mapping]
    if missing:
        raise ValueError(f"target-date alignment failed; missing examples: {missing[:3]}")
    return np.asarray([mapping[value] for value in reference], dtype=np.int64)


def load_aligned_bundle():
    bundle = load_common_xlsx(DATA_XLSX, ADJ_CSV, seq_len=SEQ_LEN,
                              train_rate=0.8, val_ratio=0.10,
                              allow_positional_adj=True)
    reference = load_common_xlsx(DATA_XLSX, ADJ_CSV, seq_len=COMMON_WINDOW,
                                 train_rate=0.8, val_ratio=0.10,
                                 allow_positional_adj=True)
    for split in ("train", "val", "test"):
        idx = _indices_for_dates(getattr(bundle, f"date_{split}"),
                                 getattr(reference, f"date_{split}"))
        for prefix in ("x", "y", "y_raw", "calendar", "date", "sample_weight"):
            name = f"{prefix}_{split}"
            value = getattr(bundle, name)
            if value is not None:
                setattr(bundle, name, value[idx])
    return bundle, reference


def inverse_temperature(values, bundle):
    values = np.asarray(values, np.float32)
    return values * (bundle.scaler_max - bundle.scaler_min) + bundle.scaler_min


def date_hash(dates) -> str:
    payload = np.asarray(dates).astype("datetime64[D]").astype(str)
    return hashlib.sha256("\n".join(payload.tolist()).encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(path, dates, station_ids, y_true, y_pred):
    rows = []
    for day, date in enumerate(np.asarray(dates).astype("datetime64[D]").astype(str)):
        for node, station in enumerate(station_ids):
            rows.append({"date": date, "station_id": station,
                         "y_true": float(y_true[day, node]),
                         "y_pred": float(y_pred[day, node]),
                         "error": float(y_pred[day, node] - y_true[day, node])})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
