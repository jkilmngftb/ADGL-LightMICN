"""Evaluate the fixed/dynamic graph x TCN/LightMICN four-cell table."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def _result_dir(variant, legacy=None):
    """Prefer a freshly trained run; fall back to the packaged paper checkpoint."""
    fresh = PACKAGE_ROOT / "results" / "architecture_matching" / "runs" / variant
    if (fresh / "best.weights.h5").is_file():
        return fresh
    if legacy is not None:
        packaged = PACKAGE_ROOT / "checkpoints" / "architecture_matching" / "reference" / legacy
        if (packaged / "best.weights.h5").is_file():
            return packaged
    return fresh


GROUPS = {
    "fixed_gcn_tcn": (
        "fixed_tcn",
        _result_dir("fixed_tcn", "trained_results/fixed_tcn"),
    ),
    "adgl_tcn": ("adgl_tcn", _result_dir("adgl_tcn", "reused_results/adgl_tcn")),
    "fixed_gcn_lightmicn": (
        "fixed_lightmicn", _result_dir("fixed_lightmicn", "reused_results/fixed_lightmicn"),
    ),
    "adgl_lightmicn": (
        "adgl_lightmicn", _result_dir("adgl_lightmicn", "reused_results/adgl_lightmicn"),
    ),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def align(bundle, reference):
    for split in ("val", "test"):
        dates = getattr(bundle, "date_" + split)
        target = getattr(reference, "date_" + split)
        indices = np.flatnonzero(np.isin(dates, target))
        if not np.array_equal(dates[indices], target):
            raise ValueError(f"Cannot align {split} target dates")
        for prefix in ("x_", "y_", "y_raw_", "calendar_", "date_", "sample_weight_"):
            name = prefix + split
            value = getattr(bundle, name)
            if value is not None:
                setattr(bundle, name, value[indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--timing-repeats", type=int, default=20)
    options = parser.parse_args()
    if options.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    import tensorflow as tf
    from utils.training import _inputs, set_seed
    from data_provider.data_loader import load_common_xlsx, inverse_scale
    from utils.metrics import metrics
    from models.architecture_matched import ModelConfig
    from exp.architecture_matching.variants import build_model

    gpus = tf.config.list_physical_devices("GPU")
    if options.device == "gpu":
        if not gpus:
            raise RuntimeError("GPU requested but TensorFlow cannot detect a GPU")
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.experimental.enable_op_determinism()
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    rows = {}
    hashes = {"val": None, "test": None}
    for label, (variant, result_dir) in GROUPS.items():
        recorded = read_json(result_dir / "metrics.json")["best_checkpoint_validation"]
        from utils.protocol import MODEL_CONFIG
        values = MODEL_CONFIG
        bundle = load_common_xlsx(
            PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
            PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
            seq_len=int(values["seq_len"]),
            allow_positional_adj=True,
        )
        reference = load_common_xlsx(
            PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
            PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
            seq_len=48,
            allow_positional_adj=True,
        )
        align(bundle, reference)
        for split in ("val", "test"):
            current = hashlib.sha256(
                np.asarray(getattr(bundle, "date_" + split)).astype(str).tobytes()
            ).hexdigest()
            if hashes[split] is None:
                hashes[split] = current
            elif hashes[split] != current:
                raise ValueError(f"Different {split} target dates")

        config = ModelConfig(**{
            field.name: values[field.name]
            for field in fields(ModelConfig) if field.name in values
        }, num_nodes=bundle.num_nodes)
        tf.keras.backend.clear_session()
        set_seed(int(values["seed"]))
        model = build_model(config, bundle.adj, variant)
        model([value[:1] for value in _inputs(bundle, "train", target_calendar=True)],
              training=False)
        model.load_weights(str(result_dir / "best.weights.h5"))

        evaluated = {}
        for split in ("val", "test"):
            prediction = model(_inputs(bundle, split, target_calendar=True),
                               training=False).numpy()
            raw = inverse_scale(prediction, bundle.scaler_min, bundle.scaler_max)
            evaluated[split] = metrics(getattr(bundle, "y_raw_" + split), raw)
            if split == "test":
                with (ROOT / f"predictions_graph_encoder_{label}_test.csv").open(
                    "w", encoding="utf-8-sig", newline=""
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["date", "station_index", "y_true", "y_pred", "error"])
                    for sample, date in enumerate(bundle.date_test):
                        for station in range(raw.shape[1]):
                            truth = float(bundle.y_raw_test[sample, station])
                            pred = float(raw[sample, station])
                            writer.writerow([str(date), station, truth, pred, pred - truth])
        difference = abs(evaluated["val"]["rmse"] - float(recorded["rmse"]))
        if difference > 1e-4:
            raise ValueError(f"Validation reproduction failed for {label}: {difference}")

        inputs = _inputs(bundle, "test", target_calendar=True)
        for _ in range(3):
            model(inputs, training=False).numpy()
        start = time.perf_counter()
        for _ in range(options.timing_repeats):
            model(inputs, training=False).numpy()
        seconds = (time.perf_counter() - start) / options.timing_repeats
        rows[label] = {
            "variant": variant,
            "graph_mode": model.graph_mode,
            "temporal_mode": model.temporal_mode,
            "validation": evaluated["val"],
            "test": evaluated["test"],
            "validation_reproduction_abs_rmse_diff": difference,
            "parameters": int(model.count_params()),
            "sparse_lambda": float(values["sparse_lambda"]),
            "inference_seconds_full_test": seconds,
            "inference_ms_per_sample": 1000.0 * seconds / len(bundle.y_raw_test),
            "result_dir": str(result_dir.relative_to(PACKAGE_ROOT)),
        }

    report = {
        "comparison": "graph_type_x_temporal_encoder",
        "strict_single_variable_dynamicity_comparison": False,
        "reason_not_strict": (
            "The fixed graph removes the dynamic graph generator and its gate-specific L1 term; "
            "the GCN propagation, graph projection, calendar fusion, station-conditioned head, "
            "residual prediction, data split, and training protocol are retained."
        ),
        "validation_dynamic_minus_fixed_rmse": {
            "tcn": rows["adgl_tcn"]["validation"]["rmse"]
                   - rows["fixed_gcn_tcn"]["validation"]["rmse"],
            "lightmicn": rows["adgl_lightmicn"]["validation"]["rmse"]
                         - rows["fixed_gcn_lightmicn"]["validation"]["rmse"],
        },
        "test_dynamic_minus_fixed_rmse": {
            "tcn": rows["adgl_tcn"]["test"]["rmse"]
                   - rows["fixed_gcn_tcn"]["test"]["rmse"],
            "lightmicn": rows["adgl_lightmicn"]["test"]["rmse"]
                         - rows["fixed_gcn_lightmicn"]["test"]["rmse"],
        },
        "groups": rows,
        "target_dates_sha256": hashes,
        "fixed_adjacency_sha256": hashlib.sha256(
            (PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv").read_bytes()
        ).hexdigest(),
        "fixed_adjacency_provenance": (
            "project-provided prespecified static adjacency; its construction provenance "
            "is not documented in the available project artifacts and must be confirmed "
            "before claiming that it is geographic or training-only"
        ),
    }
    (ROOT / "GRAPH_ENCODER_FACTORIAL_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
