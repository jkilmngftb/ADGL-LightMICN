from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import metrics
from models.baselines import ModelConfig, NEURAL_MODELS, build_model
from utils.protocol import (BATCH_SIZE, DROPOUT, HEADS, HIDDEN_DIM, L2_LAMBDA,
                       LEARNING_RATE, LR_FLOOR, LR_HALVE_EVERY, MAX_EPOCHS,
                       MIN_DELTA, PATIENCE, RESULTS, SEED, SEQ_LEN, FROZEN_PROTOCOL)
from utils.runtime import (configure_tensorflow, date_hash, inverse_temperature,
                      load_aligned_bundle, write_json, write_predictions)


def _predict(model, x, calendar, batch_size=BATCH_SIZE):
    parts = []
    for start in range(0, len(x), batch_size):
        parts.append(model([x[start:start + batch_size], calendar[start:start + batch_size]],
                           training=False).numpy())
    return np.concatenate(parts, axis=0)


def train(name: str, output: Path, device="auto", max_epochs=MAX_EPOCHS):
    if name not in NEURAL_MODELS:
        raise ValueError(name)
    tf, gpus = configure_tensorflow(device)
    bundle, reference = load_aligned_bundle()
    target_feature = bundle.feature_names.index("平均气温")
    cfg = ModelConfig(num_nodes=bundle.num_nodes, seq_len=SEQ_LEN,
                      input_dim=bundle.x_train.shape[-1], hidden_dim=HIDDEN_DIM,
                      heads=HEADS, dropout=DROPOUT, target_feature=target_feature)
    model = build_model(name, cfg, bundle.adj)
    model([tf.convert_to_tensor(bundle.x_train[:2]),
           tf.convert_to_tensor(bundle.calendar_train[:2])], training=False)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    @tf.function(reduce_retracing=True)
    def train_step(xb, cb, yb, wb):
        with tf.GradientTape() as tape:
            pred = model([xb, cb], training=True)
            weights = wb / (tf.reduce_mean(wb) + 1e-8)
            task = 0.5 * tf.reduce_sum(weights * tf.square(pred - yb))
            l2 = L2_LAMBDA * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
            loss = task + l2
        grads = tape.gradient(loss, model.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(pairs)
        return task, l2, loss

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_status.json", {"status": "running", "model": name})
    rng = np.random.default_rng(SEED)
    checkpoint_best = float("inf")
    stop_best = float("inf")
    best_epoch = stop_best_epoch = 0
    wait = 0
    history = []
    started = time.perf_counter()
    stopped_early = False
    for epoch in range(max_epochs):
        lr = max(LR_FLOOR, LEARNING_RATE * 0.5 ** (epoch // LR_HALVE_EVERY))
        optimizer.learning_rate.assign(lr)
        order = rng.permutation(len(bundle.y_train))
        totals = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            _, _, total = train_step(
                tf.convert_to_tensor(bundle.x_train[idx]),
                tf.convert_to_tensor(bundle.calendar_train[idx]),
                tf.convert_to_tensor(bundle.y_train[idx]),
                tf.convert_to_tensor(bundle.sample_weight_train[idx]))
            totals.append(float(total.numpy()))
        val_scaled = _predict(model, bundle.x_val, bundle.calendar_val)
        val_pred = inverse_temperature(val_scaled, bundle)
        score = metrics(bundle.y_raw_val, val_pred)
        row = {"epoch": epoch + 1, "learning_rate": lr,
               "total_loss": float(np.mean(totals)),
               "val_mae": score["mae"], "val_rmse": score["rmse"], "val_r2": score["r2"]}
        history.append(row)
        print(json.dumps({"model": name, **row}), flush=True)
        if score["rmse"] < checkpoint_best:
            checkpoint_best = score["rmse"]
            best_epoch = epoch + 1
            model.save_weights(str(output / "best.weights.h5"))
        if score["rmse"] < stop_best - MIN_DELTA:
            stop_best = score["rmse"]
            stop_best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            stopped_early = True
            break

    model.load_weights(str(output / "best.weights.h5"))
    val_scaled = _predict(model, bundle.x_val, bundle.calendar_val)
    val_pred = inverse_temperature(val_scaled, bundle)
    best_metrics = metrics(bundle.y_raw_val, val_pred)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False, encoding="utf-8-sig")
    write_predictions(output / "validation_predictions.csv", bundle.date_val,
                      bundle.station_ids, bundle.y_raw_val, val_pred)
    config = {**FROZEN_PROTOCOL, "model": name, "model_config": asdict(cfg)}
    result = {
        "model": name, "validation": best_metrics, "best_epoch": best_epoch,
        "epochs_trained": len(history), "max_epochs": max_epochs,
        "stopped_early": stopped_early,
        "stop_reason": "patience_exhausted" if stopped_early else "max_epochs_reached",
        "early_stopping_reference_rmse": stop_best,
        "early_stopping_reference_epoch": stop_best_epoch,
        "test_evaluated": False,
        "parameter_count": int(model.count_params()),
        "training_seconds": time.perf_counter() - started,
        "validation_date_sha256": date_hash(bundle.date_val),
        "test_date_sha256": date_hash(bundle.date_test),
        "aligned_counts": {"train": len(bundle.y_train), "val": len(bundle.y_val),
                           "test": len(bundle.y_test)},
        "tensorflow_gpus": gpus,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "run_status.json", {"status": "complete", "model": name})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=NEURAL_MODELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help="Smoke-only override; formal runs must use 500")
    args = parser.parse_args()
    output = args.output or RESULTS / args.model
    train(args.model, output, args.device, args.max_epochs)


if __name__ == "__main__":
    main()
