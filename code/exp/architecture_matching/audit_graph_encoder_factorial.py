"""Audit the fixed/dynamic graph x TCN/LightMICN comparison."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    import tensorflow as tf
    from utils.training import _inputs, _loss_terms, set_seed
    from data_provider.data_loader import load_common_xlsx, inverse_scale
    from utils.metrics import metrics
    from models.architecture_matched import ModelConfig
    from exp.architecture_matching.variants import build_model, effective_spec

    selected = read_json(ROOT / "configs" / "final_selected_config.json")["config"]
    bundle = load_common_xlsx(
        PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
        PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
        seq_len=int(selected["window"]),
        allow_positional_adj=True,
    )
    reference = load_common_xlsx(
        PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
        PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv",
        seq_len=int(selected["common_window"]),
        allow_positional_adj=True,
    )
    for split in ("train", "val"):
        dates = getattr(bundle, "date_" + split)
        target = getattr(reference, "date_" + split)
        indices = np.flatnonzero(np.isin(dates, target))
        assert np.array_equal(dates[indices], target)
        for prefix in ("x_", "y_", "y_raw_", "calendar_", "date_", "sample_weight_"):
            name = prefix + split
            value = getattr(bundle, name)
            if value is not None:
                setattr(bundle, name, value[indices])

    values = {
        "seq_len": selected["window"], "hidden_dim": 64, "heads": 4,
        "dropout": 0.1, "node_embedding_dim": 16, "top_k": 4,
        "light_micn_conv_kernels": tuple(selected["kernels"]),
        "light_micn_layers": selected["layers"],
        "temporal_token_kernel_size": 3,
    }
    config = ModelConfig(**{
        field.name: values[field.name]
        for field in fields(ModelConfig) if field.name in values
    }, num_nodes=bundle.num_nodes)
    inputs = [tf.convert_to_tensor(value[:2]) for value in _inputs(
        bundle, "train", target_calendar=True
    )]
    result_dirs = {
        "adgl_tcn": PACKAGE_ROOT / "checkpoints" / "architecture_matching" / "reference" / "reused_results" / "adgl_tcn",
        "fixed_lightmicn": PACKAGE_ROOT / "checkpoints" / "architecture_matching" / "reference" / "reused_results" / "fixed_lightmicn",
        "adgl_lightmicn": PACKAGE_ROOT / "checkpoints" / "architecture_matching" / "reference" / "reused_results" / "adgl_lightmicn",
    }
    rows = []
    for variant in ("fixed_tcn", "adgl_tcn", "fixed_lightmicn", "adgl_lightmicn"):
        tf.keras.backend.clear_session()
        set_seed(int(selected["seed"]))
        spec = effective_spec(variant, selected["kernels"])
        model = build_model(config, bundle.adj, variant)
        prediction = model(inputs, training=False)
        assert tuple(prediction.shape) == (2, bundle.num_nodes)
        assert np.isfinite(prediction.numpy()).all()
        assert model.graph_mode == spec["graph_mode"]
        assert model.graph_projection is not None
        assert model.residual_output and model.use_target_calendar
        assert model.station_conditioned_head and model.use_graph_propagation
        if spec["graph_mode"] == "fixed":
            assert model.dynamic_adjacency is None
            assert model.fixed_adjacency is not None
            assert float(model.sparsity_loss().numpy()) == 0.0
        else:
            assert model.dynamic_adjacency is not None
            assert model.fixed_adjacency is None
            assert model.dynamic_adjacency.top_k == 4
        with tf.GradientTape() as tape:
            pred = model(inputs, training=True)
            terms = _loss_terms(
                model,
                tf.convert_to_tensor(bundle.y_train[:2]),
                pred,
                tf.convert_to_tensor(bundle.sample_weight_train[:2]),
                spec["sparse_lambda"],
            )
        gradients = tape.gradient(terms["total"], model.trainable_variables)
        active = [gradient for gradient in gradients if gradient is not None]
        assert active
        assert all(np.isfinite(tf.convert_to_tensor(value).numpy()).all()
                   for value in active)
        row = {
            "variant": variant,
            "graph_mode": model.graph_mode,
            "temporal_mode": model.temporal_mode,
            "parameters": int(model.count_params()),
            "sparse_lambda": spec["sparse_lambda"],
            "graph_projection_retained": True,
            "finite_gradients": True,
        }
        if variant in result_dirs:
            result_dir = result_dirs[variant]
            model.load_weights(str(result_dir / "best.weights.h5"))
            raw_prediction = inverse_scale(
                model(_inputs(bundle, "val", target_calendar=True), training=False).numpy(),
                bundle.scaler_min,
                bundle.scaler_max,
            )
            current = metrics(bundle.y_raw_val, raw_prediction)
            recorded = read_json(result_dir / "metrics.json")["best_checkpoint_validation"]
            difference = abs(current["rmse"] - float(recorded["rmse"]))
            assert difference < 1e-4, (variant, difference)
            row["reused_checkpoint_val_rmse"] = current["rmse"]
            row["reproduction_abs_diff"] = difference
        rows.append(row)

    raw_adj = np.loadtxt(PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv", delimiter=",")
    assert raw_adj.shape == (bundle.num_nodes, bundle.num_nodes)
    assert np.allclose(raw_adj, raw_adj.T)
    assert np.allclose(np.diag(raw_adj), 0.0)
    assert set(np.unique(raw_adj)).issubset({0.0, 1.0})
    report = {
        "passed": True,
        "formal_training_performed": False,
        "test_evaluated": False,
        "rows": rows,
        "fixed_adjacency": {
            "shape": list(raw_adj.shape),
            "binary": True,
            "symmetric": True,
            "zero_diagonal": True,
            "sha256": hashlib.sha256((PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv").read_bytes()).hexdigest(),
            "provenance_status": "construction origin not documented in available artifacts",
        },
        "target_date_sha256": {
            split: hashlib.sha256(
                np.asarray(getattr(bundle, "date_" + split)).astype(str).tobytes()
            ).hexdigest()
            for split in ("train", "val")
        },
        "tensorflow": tf.__version__,
    }
    (ROOT / "GRAPH_ENCODER_FACTORIAL_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
