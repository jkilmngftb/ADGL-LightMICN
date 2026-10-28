"""Evaluate the exact frozen proposed-model source/checkpoint shipped with this package."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from utils.metrics import metrics
from utils.protocol import DATA_XLSX, ADJ_CSV, RESULTS, COMMON_WINDOW, MODEL_CONFIG
from utils.runtime import configure_tensorflow, date_hash, write_json, write_predictions


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "checkpoints" / "adgl_lightmicn"


def _align(bundle, reference, split):
    dates = getattr(bundle, "date_" + split)
    ref = getattr(reference, "date_" + split)
    index = {value: i for i, value in enumerate(dates)}
    idx = np.asarray([index[value] for value in ref], np.int64)
    for prefix in ("x_", "y_", "y_raw_", "calendar_", "date_", "sample_weight_"):
        name = prefix + split
        value = getattr(bundle, name)
        if value is not None:
            setattr(bundle, name, value[idx])


def run(device="auto"):
    tf, _ = configure_tensorflow(device)
    from models.frozen_adgl_lightmicn.models import ModelConfig
    from models.frozen_adgl_lightmicn.common_train import _inputs, set_seed
    from models.frozen_adgl_lightmicn.data import inverse_scale, load_common_xlsx
    from models.frozen_adgl_lightmicn.variants import build_model

    values = MODEL_CONFIG
    set_seed(int(values["seed"]))
    bundle = load_common_xlsx(DATA_XLSX, ADJ_CSV, seq_len=int(values["seq_len"]),
                              allow_positional_adj=True)
    reference = load_common_xlsx(DATA_XLSX, ADJ_CSV, seq_len=COMMON_WINDOW,
                                 allow_positional_adj=True)
    for split in ("val", "test"):
        _align(bundle, reference, split)
    cfg = ModelConfig(**{field.name: values[field.name]
                         for field in fields(ModelConfig) if field.name in values},
                      num_nodes=bundle.num_nodes)
    model = build_model(cfg, bundle.adj, "full")
    model([value[:1] for value in _inputs(bundle, "train", target_calendar=True)], training=False)
    model.load_weights(str(RUN / "best.weights.h5"))
    out = RESULTS / "adgl_lightmicn"
    out.mkdir(parents=True, exist_ok=True)
    results = {"model": "adgl_lightmicn", "source": "exact_frozen_source_and_checkpoint",
               "parameter_count": int(model.count_params())}
    for split in ("val", "test"):
        predicted = model(_inputs(bundle, split, target_calendar=True), training=False).numpy()
        raw = inverse_scale(predicted, bundle.scaler_min, bundle.scaler_max)
        truth = getattr(bundle, "y_raw_" + split)
        dates = getattr(bundle, "date_" + split)
        results["validation" if split == "val" else "test"] = metrics(truth, raw)
        results[split + "_date_sha256"] = date_hash(dates)
        write_predictions(out / f"{split}_predictions.csv", dates, bundle.station_ids, truth, raw)
    write_json(out / "test_metrics.json", results)
    write_json(out / "run_status.json", {"status": "complete", "model": "adgl_lightmicn"})
    return results
