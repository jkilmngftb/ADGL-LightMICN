from __future__ import annotations

import argparse
import json
import numpy as np

from models.baselines import ModelConfig, NEURAL_MODELS, build_model
from utils.runtime import configure_tensorflow, load_aligned_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    args = parser.parse_args()
    tf, gpus = configure_tensorflow(args.device)
    bundle, _ = load_aligned_bundle()
    cfg = ModelConfig(num_nodes=bundle.num_nodes, seq_len=12,
                      input_dim=bundle.x_train.shape[-1], hidden_dim=64,
                      heads=4, dropout=0.1,
                      target_feature=bundle.feature_names.index("平均气温"))
    report = {"device": args.device, "visible_gpus": gpus, "models": {}}
    for name in NEURAL_MODELS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(2026)
        model = build_model(name, cfg, bundle.adj)
        x = tf.convert_to_tensor(bundle.x_train[:2])
        c = tf.convert_to_tensor(bundle.calendar_train[:2])
        y = tf.convert_to_tensor(bundle.y_train[:2])
        with tf.GradientTape() as tape:
            pred = model([x, c], training=True)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, model.trainable_variables)
        ok = tuple(pred.shape) == (2, bundle.num_nodes) and np.isfinite(pred.numpy()).all()
        ok = ok and all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
        report["models"][name] = {"pass": bool(ok), "shape": list(pred.shape),
                                  "parameters": int(model.count_params())}
        if not ok:
            raise RuntimeError(f"smoke failed: {name}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
