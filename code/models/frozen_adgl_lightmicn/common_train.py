"""Shared training, loss, input, and prediction utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf


LAMBDA_L2 = 0.0015


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _l2(model):
    variables = model.trainable_variables
    return tf.add_n([tf.nn.l2_loss(v) for v in variables]) if variables else tf.constant(0.0)


def _loss_terms(model, y_true, y_pred, weights, sparse_lambda=0.0):
    sw = tf.convert_to_tensor(weights, dtype=y_pred.dtype)
    sw = sw / (tf.reduce_mean(sw) + 1e-8)
    sq = tf.square(y_pred - y_true)
    task = 0.5 * tf.reduce_sum(sw * sq)
    l2 = LAMBDA_L2 * _l2(model)
    sparse = model.sparsity_loss() if hasattr(model, "sparsity_loss") else tf.constant(0.0, dtype=y_pred.dtype)
    weighted_sparse = sparse_lambda * sparse
    return {"task": task, "l2": l2, "sparse": sparse,
            "weighted_sparse": weighted_sparse,
            "total": task + l2 + weighted_sparse}


def _forward(model, inputs, training=False):
    if isinstance(model, tf.keras.Model):
        return model(inputs, training=training)
    return model(inputs, training=training)


def _target_calendar(dates):
    dates = pd.to_datetime(np.asarray(dates))
    angle = 2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.2425
    return np.stack(
        [np.sin(angle), np.cos(angle), np.sin(2.0 * angle), np.cos(2.0 * angle)],
        axis=-1,
    ).astype(np.float32)


def _inputs(bundle, split, target_calendar=False):
    x = getattr(bundle, f"x_{split}")
    c = getattr(bundle, f"calendar_{split}")
    values = [x, c]
    if target_calendar:
        values.append(_target_calendar(getattr(bundle, f"date_{split}")))
    return values


def _predict(model, bundle, split, attention=False, return_adgl=False):
    inputs = _inputs(
        bundle,
        split,
        target_calendar=bool(getattr(model, "use_target_calendar", False)),
    )
    if return_adgl:
        pred, adjacency, selected_gate, raw_gate = model(
            inputs, training=False, return_adjacency=True
        )
        return pred.numpy(), {
            "adjacency": adjacency.numpy(),
            "selected_gate": selected_gate.numpy(),
            "raw_gate": raw_gate.numpy() if raw_gate is not None else None,
        }
    if attention:
        pred, weights = model(inputs, training=False, return_attention=True)
        return pred.numpy(), weights.numpy()
    return _forward(model, inputs, training=False).numpy(), None


if __name__ == "__main__":
    main()
