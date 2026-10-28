"""Reproduce validation metrics, then evaluate the frozen macro 2x2 on Test."""
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
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
CODE = ROOT
sys.path.insert(0, str(CODE))


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
    "A_tcn_base": ("tcn_base", _result_dir("tcn_base", "macro_results/tcn_base")),
    "B_adgl_tcn": ("adgl_tcn", _result_dir("adgl_tcn", "reused_results/adgl_tcn")),
    "C_lightmicn_base": ("lightmicn_base", _result_dir("lightmicn_base", "macro_results/lightmicn_base")),
    "D_adgl_lightmicn": ("adgl_lightmicn", _result_dir("adgl_lightmicn", "reused_results/adgl_lightmicn")),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def align(bundle, reference):
    for split in ("val", "test"):
        dates = getattr(bundle, "date_" + split)
        reference_dates = getattr(reference, "date_" + split)
        indices = np.flatnonzero(np.isin(dates, reference_dates))
        if not np.array_equal(dates[indices], reference_dates):
            raise ValueError(f"Target-date alignment failed for {split}")
        for prefix in ("x_", "y_", "y_raw_", "calendar_", "date_", "sample_weight_"):
            name = prefix + split
            value = getattr(bundle, name)
            if value is not None:
                setattr(bundle, name, value[indices])


def recorded_val(metrics):
    if "best_checkpoint_validation" in metrics:
        return metrics["best_checkpoint_validation"]
    if "validation" in metrics:
        return metrics["validation"]
    raise KeyError("No recorded validation metrics")


def save_predictions(path, bundle, truth, prediction):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "station_index", "y_true", "y_pred", "error"])
        for sample, date in enumerate(bundle.date_test):
            for node in range(prediction.shape[1]):
                error = float(prediction[sample, node] - truth[sample, node])
                writer.writerow([str(date), node, float(truth[sample, node]),
                                 float(prediction[sample, node]), error])


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

    results = {}
    test_hash = None
    for group, (variant, result_dir) in GROUPS.items():
        metric_record = read_json(result_dir / "metrics.json")
        from utils.protocol import MODEL_CONFIG
        values = MODEL_CONFIG
        seed = int(values["seed"])
        set_seed(seed)
        bundle = load_common_xlsx(PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
                                  PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
                                  seq_len=int(values["seq_len"]),
                                  allow_positional_adj=True)
        reference = load_common_xlsx(PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
                                     PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
                                     seq_len=48,
                                     allow_positional_adj=True)
        align(bundle, reference)
        date_hash = hashlib.sha256(np.asarray(bundle.date_test).astype(str).tobytes()).hexdigest()
        if test_hash is None:
            test_hash = date_hash
        elif test_hash != date_hash:
            raise ValueError("The four groups do not use identical Test target dates")

        config = ModelConfig(**{
            field.name: values[field.name]
            for field in fields(ModelConfig) if field.name in values
        }, num_nodes=bundle.num_nodes)
        tf.keras.backend.clear_session()
        set_seed(seed)
        model = build_model(config, bundle.adj, variant)
        train_inputs = _inputs(bundle, "train", target_calendar=True)
        model([value[:1] for value in train_inputs], training=False)
        model.load_weights(str(result_dir / "best.weights.h5"))

        split_metrics = {}
        raw_predictions = {}
        for split in ("val", "test"):
            inputs = _inputs(bundle, split, target_calendar=True)
            prediction = model(inputs, training=False).numpy()
            raw = inverse_scale(prediction, bundle.scaler_min, bundle.scaler_max)
            raw_predictions[split] = raw
            split_metrics[split] = metrics(getattr(bundle, "y_raw_" + split), raw)
        expected = recorded_val(metric_record)
        if abs(split_metrics["val"]["rmse"] - float(expected["rmse"])) > 1e-4:
            raise ValueError(f"Validation checkpoint reproduction failed for {group}")

        test_inputs = _inputs(bundle, "test", target_calendar=True)
        for _ in range(3):
            model(test_inputs, training=False).numpy()
        start = time.perf_counter()
        for _ in range(options.timing_repeats):
            model(test_inputs, training=False).numpy()
        elapsed = time.perf_counter() - start
        inference_seconds = elapsed / options.timing_repeats
        save_predictions(ROOT / f"predictions_{group}_test.csv", bundle,
                         bundle.y_raw_test, raw_predictions["test"])
        results[group] = {
            "variant": variant,
            "result_dir": str(result_dir.relative_to(PACKAGE_ROOT)),
            "validation": split_metrics["val"],
            "test": split_metrics["test"],
            "parameters": int(model.count_params()),
            "inference_seconds_full_test": inference_seconds,
            "inference_ms_per_sample": 1000.0 * inference_seconds / len(bundle.y_raw_test),
            "test_target_dates_sha256": date_hash,
            "checkpoint_sha256": hashlib.sha256((result_dir / "best.weights.h5").read_bytes()).hexdigest(),
        }

    def rmse(group, split):
        return float(results[group][split]["rmse"])

    comparisons = {}
    for split in ("validation", "test"):
        a, b = rmse("A_tcn_base", split), rmse("B_adgl_tcn", split)
        c, d = rmse("C_lightmicn_base", split), rmse("D_adgl_lightmicn", split)
        comparisons[split] = {
            "B_minus_A_rmse": b - a,
            "C_minus_A_rmse": c - a,
            "D_minus_B_rmse": d - b,
            "D_minus_C_rmse": d - c,
            "adgl_improvement_with_tcn_A_minus_B": a - b,
            "lightmicn_improvement_without_graph_A_minus_C": a - c,
            "lightmicn_improvement_with_adgl_B_minus_D": b - d,
            "adgl_improvement_with_lightmicn_C_minus_D": c - d,
            "factorial_interaction_contrast_D_minus_B_minus_C_plus_A": d - b - c + a,
        }

    report = {
        "protocol": read_json(ROOT / "configs" / "final_selected_config.json")["config"],
        "groups": results,
        "comparisons": comparisons,
        "interpretation_guard": (
            "The interaction contrast is descriptive. Do not claim complementarity or causality "
            "unless the observed four-cell pattern supports it."
        ),
        "test_target_dates_sha256": test_hash,
        "runtime": {"tensorflow": tf.__version__, "device": options.device,
                    "gpus": [gpu.name for gpu in gpus]},
    }
    write_json(ROOT / "MACRO_2X2_RESULTS.json", report)
    with (ROOT / "MACRO_2X2_RESULTS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "variant", "val_mae", "val_rmse", "val_r2",
                         "test_mae", "test_rmse", "test_r2", "parameters",
                         "inference_ms_per_sample"])
        for group, row in results.items():
            writer.writerow([group, row["variant"], row["validation"]["mae"],
                             row["validation"]["rmse"], row["validation"]["r2"],
                             row["test"]["mae"], row["test"]["rmse"], row["test"]["r2"],
                             row["parameters"], row["inference_ms_per_sample"]])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
