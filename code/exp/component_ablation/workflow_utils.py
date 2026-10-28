"""Shared helpers for candidate evaluation, ablations, and multi-seed runs."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
CODE = ROOT
TRAIN = CODE / "train_variant.py"
sys.path.insert(0, str(CODE))

COMMON_WINDOW = 48
MAX_EPOCHS = 500
PATIENCE = 50
MIN_DELTA = 0.001
_TF_RUNTIME = None
RESULTS_ROOT = PACKAGE_ROOT / "results" / "component_ablation"
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoints" / "component_ablation"


def model_config_values(variant="full"):
    from utils.protocol import MODEL_CONFIG
    values = dict(MODEL_CONFIG)
    if variant == "single_scale":
        values["light_micn_conv_kernels"] = (2,)
    return values


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_to_reference_dates(bundle, reference, splits=("train", "val", "test")):
    for split in splits:
        dates = getattr(bundle, "date_" + split)
        reference_dates = getattr(reference, "date_" + split)
        indices = np.flatnonzero(np.isin(dates, reference_dates))
        if not np.array_equal(dates[indices], reference_dates):
            raise ValueError(f"Cannot align {split} dates to common-window protocol")
        for prefix in ("x_", "y_", "y_raw_", "calendar_", "date_", "sample_weight_"):
            name = prefix + split
            value = getattr(bundle, name)
            if value is not None:
                setattr(bundle, name, value[indices])


def configure_device(device):
    global _TF_RUNTIME
    if _TF_RUNTIME is not None:
        return _TF_RUNTIME
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu":
        if not gpus:
            raise RuntimeError("GPU requested but TensorFlow cannot detect a GPU")
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.experimental.enable_op_determinism()
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    _TF_RUNTIME = tf
    return tf


def checkpoint_path(result_dir):
    """Resolve a fresh-run checkpoint or its packaged mirror."""
    result_dir = Path(result_dir)
    checkpoint = result_dir / "best.weights.h5"
    if checkpoint.is_file():
        return checkpoint
    relative = result_dir.resolve().relative_to(RESULTS_ROOT.resolve())
    checkpoint = CHECKPOINT_ROOT / relative / "best.weights.h5"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def evaluate_checkpoint(result_dir, variant, device, align_test=True):
    """Evaluate one saved checkpoint on validation and the existing test split."""
    result_dir = Path(result_dir)
    config_values = model_config_values(variant)
    checkpoint = checkpoint_path(result_dir)

    tf = configure_device(device)
    tf.keras.backend.clear_session()
    from models.adgl_lightmicn import ModelConfig
    from utils.training import set_seed, _inputs
    from data_provider.data_loader import load_common_xlsx, inverse_scale
    from utils.metrics import metrics
    from exp.component_ablation.variants import build_model

    set_seed(int(config_values["seed"]))
    bundle = load_common_xlsx(
        PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
        PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
        seq_len=int(config_values["seq_len"]),
        allow_positional_adj=True,
    )
    if align_test:
        reference = load_common_xlsx(
            PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
            PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
            seq_len=COMMON_WINDOW,
            allow_positional_adj=True,
        )
        align_to_reference_dates(bundle, reference, splits=("val", "test"))
    values = vars(SimpleNamespace(**config_values))
    model_config = ModelConfig(
        **{field.name: values[field.name] for field in fields(ModelConfig) if field.name in values},
        num_nodes=bundle.num_nodes,
    )
    model = build_model(model_config, bundle.adj, variant)
    train_inputs = _inputs(bundle, "train", target_calendar=True)
    model([value[:1] for value in train_inputs], training=False)
    model.load_weights(str(checkpoint))
    output = {}
    predictions = {}
    for split in ("val", "test"):
        inputs = _inputs(bundle, split, target_calendar=True)
        predicted = model(inputs, training=False).numpy()
        raw = inverse_scale(predicted, bundle.scaler_min, bundle.scaler_max)
        output[split] = metrics(getattr(bundle, "y_raw_" + split), raw)
        predictions[split] = raw
    return {
        "validation": output["val"],
        "test": output["test"],
        "test_dates": [str(value) for value in bundle.date_test],
        "test_dates_sha256": hashlib.sha256(np.asarray(bundle.date_test).astype(str).tobytes()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": int(model.count_params()),
        "predictions": predictions,
        "bundle": bundle,
    }


def run_variant(config, variant, result_dir, device):
    """Run one predeclared variant, refusing to overwrite partial output."""
    result_dir = Path(result_dir)
    metrics_path = result_dir / "metrics.json"
    if metrics_path.is_file():
        return summarize_training_result(result_dir, config, variant)
    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"Interrupted output retained; refusing overwrite: {result_dir}")
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(TRAIN),
        "--variant", variant,
        "--seed", str(int(config["seed"])),
        "--device", device,
        "--seq-len", str(int(config["window"])),
        "--common-window", str(int(config.get("common_window", COMMON_WINDOW))),
        "--learning-rate", str(float(config["learning_rate"])),
        "--kernels", *(str(int(value)) for value in config["kernels"]),
        "--layers", str(int(config["layers"])),
        "--epochs", str(int(config.get("epochs", MAX_EPOCHS))),
        "--output", str(result_dir),
    ]
    log_path = result_dir.parent / f"{variant}.log"
    with log_path.open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return summarize_training_result(result_dir, config, variant)


def summarize_training_result(result_dir, config, variant):
    result_dir = Path(result_dir).resolve()
    report = read_json(result_dir / "metrics.json")
    state = read_json(result_dir / "run_status.json")
    if state.get("state") != "complete" or report.get("test_evaluated"):
        raise ValueError(f"Incomplete result or unexpected test evaluation: {result_dir}")
    candidate_kernels = [int(value) for value in config["kernels"]]
    effective_kernels = ([min(candidate_kernels)] if variant == "single_scale"
                         else candidate_kernels)
    if variant != "single_scale" and effective_kernels != candidate_kernels:
        raise ValueError(f"Kernel drift in {result_dir}")
    # `no_decomposition` removes the trend branch entirely, so it has no trend
    # initialization to compare. All other paper component variants retain it.
    from utils.protocol import PARAMETER_COUNTS
    return {
        "variant": variant,
        "validation_mae": float(report["best_checkpoint_validation"]["mae"]),
        "validation_rmse": float(report["best_checkpoint_validation"]["rmse"]),
        "validation_r2": float(report["best_checkpoint_validation"]["r2"]),
        "best_epoch": int(report["best_epoch"]),
        "epochs_trained": int(report["epochs_trained"]),
        "parameters": PARAMETER_COUNTS[variant],
        "candidate_kernels": candidate_kernels,
        "effective_kernels": effective_kernels,
        "sparse_lambda": 0.0 if variant in ("fixed_graph", "no_l1") else 0.5,
        "result_dir": str(result_dir.relative_to(PACKAGE_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path(result_dir)),
    }


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
