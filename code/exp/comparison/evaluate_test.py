from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from utils.metrics import metrics
from models.baselines import ModelConfig, build_model
from utils.protocol import (BATCH_SIZE, CHECKPOINTS, DROPOUT, HEADS, HIDDEN_DIM,
                       MODEL_ORDER, RESULTS, SEED, SEQ_LEN)
from utils.runtime import (configure_tensorflow, date_hash, inverse_temperature,
                      load_aligned_bundle, write_json, write_predictions)
from exp.comparison.evaluate_main import run as evaluate_frozen_main


def _complete(model):
    path = RESULTS / model / "run_status.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    args = parser.parse_args()
    missing = [name for name in MODEL_ORDER if not _complete(name)]
    if missing:
        raise RuntimeError(f"Refusing test evaluation before all fixed baselines complete: {missing}")
    tf, _ = configure_tensorflow(args.device)
    bundle, _ = load_aligned_bundle()
    target_idx = bundle.feature_names.index("平均气温")
    rows = []
    for name in MODEL_ORDER:
        out = RESULTS / name
        if name == "persistence":
            scaled = bundle.x_test[:, -1, :, target_idx]
        else:
            cfg = ModelConfig(num_nodes=bundle.num_nodes, seq_len=SEQ_LEN,
                              input_dim=bundle.x_test.shape[-1], hidden_dim=HIDDEN_DIM,
                              heads=HEADS, dropout=DROPOUT, target_feature=target_idx)
            model = build_model(name, cfg, bundle.adj)
            model([tf.convert_to_tensor(bundle.x_test[:2]),
                   tf.convert_to_tensor(bundle.calendar_test[:2])], training=False)
            checkpoint = out / "best.weights.h5"
            if not checkpoint.is_file():
                checkpoint = CHECKPOINTS / name / "best.weights.h5"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model.load_weights(str(checkpoint))
            parts = []
            for start in range(0, len(bundle.x_test), BATCH_SIZE):
                parts.append(model([bundle.x_test[start:start+BATCH_SIZE],
                                    bundle.calendar_test[start:start+BATCH_SIZE]],
                                   training=False).numpy())
            scaled = np.concatenate(parts, axis=0)
        pred = inverse_temperature(scaled, bundle)
        score = metrics(bundle.y_raw_test, pred)
        payload = {"model": name, "test": score, "n_dates": len(bundle.date_test),
                   "date_sha256": date_hash(bundle.date_test)}
        write_json(out / "test_metrics.json", payload)
        write_predictions(out / "test_predictions.csv", bundle.date_test,
                          bundle.station_ids, bundle.y_raw_test, pred)
        rows.append({"model": name, **score})
    main_result = evaluate_frozen_main(args.device)
    rows.append({"model": "adgl_lightmicn", **main_result["test"]})
    frame = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame.to_csv(RESULTS / "baseline_test_summary.csv", index=False, encoding="utf-8-sig")
    write_json(RESULTS / "baseline_test_summary.json", frame.to_dict("records"))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
