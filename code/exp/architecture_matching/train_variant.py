"""Compiled MICN ablations: fixed validation-selected configuration."""
from pathlib import Path
from datetime import datetime
import json
import os
import sys
import time
from dataclasses import fields
import argparse
import numpy as np
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
VARIANT = "adgl_micn_station_head_no_embedding_residual_calendar"


def make_args():
    """Self-contained project-local configuration for the copied experiment."""
    return SimpleNamespace(
        data_xlsx=str(PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"),
        adj=str(PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv"), output=str(HERE / "formal_tests"),
        variants=VARIANT, seq_len=12, epochs=500, batch_size=256,
        hidden_dim=64, heads=4, dropout=0.1, learning_rate=1e-3,
        node_embedding_dim=16, top_k=4, sparse_lambda=0.5,
        light_micn_conv_kernels=(2, 4), light_micn_layers=1,
        temporal_token_kernel_size=3,
        patience=50, min_delta=0.001, evaluate_test=False, seed=2026,
        allow_positional_adj=True,
    )


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))

def captured_rms(captured, key):
    return rms(captured[key].numpy()) if key in captured else None


def main():
    parser = argparse.ArgumentParser()
    from exp.architecture_matching.variants import SPECS, effective_spec, build_model
    parser.add_argument("--variant", choices=tuple(SPECS), default="full")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--kernels", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--common-window", type=int, default=48)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    options = parser.parse_args()
    if options.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if os.name == "nt" and library_bin.is_dir():
        os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            cuda_dll_handle = os.add_dll_directory(str(library_bin))
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if options.device == "gpu":
        if not gpus:
            raise RuntimeError("GPU requested but TensorFlow cannot detect a GPU")
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.experimental.enable_op_determinism()
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    import pandas as pd
    from models.architecture_matched import ModelConfig
    from utils.training import set_seed, _inputs, _loss_terms
    from data_provider.data_loader import load_common_xlsx, inverse_scale
    from utils.metrics import metrics
    args = make_args()
    spec = effective_spec(options.variant, options.kernels)
    args.variants = options.variant
    args.seed = options.seed
    args.epochs = options.epochs
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.learning_rate = options.learning_rate
    args.light_micn_conv_kernels = tuple(spec["kernels"])
    args.sparse_lambda = spec["sparse_lambda"]
    args.light_micn_layers = options.layers
    args.seq_len = options.seq_len
    args.output = str((options.output or PACKAGE_ROOT / "results" / "architecture_matching" / "runs" / options.variant).resolve())
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to repeat/overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "pid": os.getpid(), "epoch": 0}
    write(output / "run_status.json", status)
    try:
        set_seed(args.seed)
        bundle = load_common_xlsx(args.data_xlsx, args.adj, seq_len=args.seq_len, allow_positional_adj=True)
        if options.common_window:
            reference = load_common_xlsx(args.data_xlsx, args.adj, seq_len=options.common_window, allow_positional_adj=True)
            assert np.array_equal(bundle.scaler_min, reference.scaler_min)
            assert np.array_equal(bundle.scaler_max, reference.scaler_max)
            for split in ("train", "val"):
                dates = getattr(bundle, "date_" + split)
                reference_dates = getattr(reference, "date_" + split)
                indices = np.flatnonzero(np.isin(dates, reference_dates))
                if not np.array_equal(dates[indices], reference_dates):
                    raise ValueError("Window comparison target dates cannot be aligned")
                for name in ("x_", "y_", "y_raw_", "calendar_", "date_", "sample_weight_"):
                    attr = name + split
                    value = getattr(bundle, attr)
                    if value is not None:
                        setattr(bundle, attr, value[indices])
        values = vars(args)
        config = ModelConfig(**{f.name: values[f.name] for f in fields(ModelConfig) if f.name in values}, num_nodes=bundle.num_nodes)
        model = build_model(config, bundle.adj, options.variant)
        train = _inputs(bundle, "train", target_calendar=True)
        val = _inputs(bundle, "val", target_calendar=True)
        model([x[:1] for x in train], training=False)
        encoder = model.temporal_encoder
        trend_layer = getattr(encoder, "trend_regression", None)
        if trend_layer is not None:
            kernel = trend_layer.kernel
            assert np.all(kernel.numpy() == 1.0)
            kernel.assign(tf.ones_like(kernel) / args.seq_len)
        if options.check_only:
            assert model.temporal_mode == spec["temporal_mode"]
            assert model.graph_mode == spec["graph_mode"]
            assert model.station_conditioned_head == spec["station_conditioned_head"]
            assert model.use_spatial_module == spec["use_spatial_module"]
            assert model.use_dynamic_graph == (
                spec["dynamic"] if spec["use_spatial_module"] else False
            )
            assert model.use_graph_propagation == spec["use_graph_propagation"]
            if model.dynamic_adjacency is not None:
                assert model.dynamic_adjacency.use_embedding == spec["use_embedding"]
                assert model.dynamic_adjacency.top_k == spec["top_k"]
            assert model.residual_output == spec["residual_output"]
            assert model.use_target_calendar == spec["use_target_calendar"]
            if model.temporal_mode == "micn":
                assert encoder.conv_kernels == args.light_micn_conv_kernels
                assert encoder.use_decomposition == spec["micn_use_decomposition"]
            status["state"] = "check_passed"
            write(output / "run_status.json", status)
            print("CONFIGURATION CHECK PASS", flush=True)
            return
        captured = {}
        seasonal_layer = getattr(encoder, "seasonal_projection", None)
        seasonal_call = seasonal_layer.call if seasonal_layer is not None else None
        trend_call = trend_layer.call if trend_layer is not None else None
        def seasonal(*a, **kw):
            result = seasonal_call(*a, **kw)
            captured["seasonal"] = result
            return result
        def trend(*a, **kw):
            result = trend_call(*a, **kw)
            captured["trend"] = result
            return result
        if seasonal_layer is not None:
            seasonal_layer.call = seasonal
        if trend_layer is not None:
            trend_layer.call = trend
        initial = model(val, training=False).numpy()
        initial_metrics = metrics(bundle.y_raw_val, inverse_scale(initial, bundle.scaler_min, bundle.scaler_max))
        print(json.dumps({"initial_validation": initial_metrics, "trend_rms": captured_rms(captured,"trend"),
            "seasonal_rms": captured_rms(captured,"seasonal")}), flush=True)
        optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        from speed_runtime import make_train_step
        train_step = make_train_step(tf, model, optimizer, _loss_terms, args.sparse_lambda)
        rng = np.random.default_rng(args.seed)
        history = []
        # Checkpoint selection and early stopping intentionally use different
        # thresholds.  Every strict validation-RMSE minimum is saved, while
        # min_delta only decides whether patience is reset.
        checkpoint_best = float("inf")
        checkpoint_best_epoch = 0
        stop_best = float("inf")
        stop_best_epoch = 0
        wait = 0
        stopped_early = False
        start_time = time.perf_counter()
        for epoch in range(args.epochs):
            optimizer.learning_rate.assign(max(1e-5, args.learning_rate * 0.5**(epoch // 50)))
            order = rng.permutation(len(bundle.y_train))
            losses = {k: [] for k in ("task", "l2", "sparse", "weighted_sparse", "total")}
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start + args.batch_size]
                terms = train_step([tf.convert_to_tensor(x[idx]) for x in train],
                                   tf.convert_to_tensor(bundle.y_train[idx]),
                                   tf.convert_to_tensor(bundle.sample_weight_train[idx]))
                for k in losses:
                    losses[k].append(float(terms[k].numpy()))
            prediction = model(val, training=False).numpy()
            row = {"epoch": epoch + 1, **metrics(bundle.y_raw_val, inverse_scale(prediction, bundle.scaler_min, bundle.scaler_max)),
                **{k + "_loss": float(np.mean(v)) for k, v in losses.items()},
                "trend_rms": captured_rms(captured,"trend"), "seasonal_rms": captured_rms(captured,"seasonal"),
                "elapsed_seconds": time.perf_counter() - start_time}
            history.append(row)
            if not all(row[k] is None or np.isfinite(row[k]) for k in ("mae", "rmse", "r2", "total_loss", "trend_rms", "seasonal_rms")):
                raise FloatingPointError("Nonfinite validation metric or branch statistic")
            if row["rmse"] < checkpoint_best:
                checkpoint_best = row["rmse"]
                checkpoint_best_epoch = epoch + 1
                model.save_weights(str(output / "best.weights.h5"))
            if row["rmse"] < stop_best - args.min_delta:
                stop_best = row["rmse"]
                stop_best_epoch = epoch + 1
                wait = 0
            else:
                wait += 1
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            status.update(epoch=epoch + 1,
                          best_epoch=checkpoint_best_epoch,
                          best_val_rmse=checkpoint_best,
                          early_stopping_reference_epoch=stop_best_epoch,
                          early_stopping_reference_rmse=stop_best,
                          early_stopping_wait=wait,
                          updated_at=datetime.now().isoformat())
            write(output / "run_status.json", status)
            print(json.dumps(row), flush=True)
            if wait >= args.patience:
                stopped_early = True
                break
        model.load_weights(str(output / "best.weights.h5"))
        best_prediction = model(val, training=False).numpy()
        reproduced = metrics(bundle.y_raw_val, inverse_scale(best_prediction, bundle.scaler_min, bundle.scaler_max))
        if abs(reproduced["rmse"] - checkpoint_best) > 1e-4:
            raise ValueError("Best checkpoint validation reproduction failed")
        write(output / "metrics.json", {"initial_validation": initial_metrics, "last_epoch": history[-1],
            "best_validation": history[checkpoint_best_epoch - 1],
            "best_epoch": checkpoint_best_epoch,
            "early_stopping_reference_rmse": stop_best,
            "early_stopping_reference_epoch": stop_best_epoch,
            "epochs_trained": len(history),
            "test_evaluated": False, "max_epochs": args.epochs, "patience": args.patience,
            "min_delta": args.min_delta, "stopped_early": stopped_early,
            "stop_reason": "patience_exhausted" if stopped_early else "max_epochs_reached",
            "best_checkpoint_validation": reproduced,
            "best_checkpoint_trend_rms": captured_rms(captured,"trend"),
            "best_checkpoint_seasonal_rms": captured_rms(captured,"seasonal")})
        status["state"] = "complete"
        write(output / "run_status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        write(output / "run_status.json", status)
        raise


if __name__ == "__main__":
    main()
