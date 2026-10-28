from __future__ import annotations

from pathlib import Path
import numpy as np

from utils.metrics import metrics
from utils.protocol import FROZEN_PROTOCOL, RESULTS
from utils.runtime import inverse_temperature, load_aligned_bundle, write_json, write_predictions


def run(output: Path | None = None):
    bundle, _ = load_aligned_bundle()
    output = output or RESULTS / "persistence"
    output.mkdir(parents=True, exist_ok=True)
    target_idx = bundle.feature_names.index("平均气温")
    val_scaled = bundle.x_val[:, -1, :, target_idx]
    val_pred = inverse_temperature(val_scaled, bundle)
    result = {"model": "persistence", "validation": metrics(bundle.y_raw_val, val_pred),
              "trained": False, "test_evaluated": False, "protocol": FROZEN_PROTOCOL}
    write_predictions(output / "validation_predictions.csv", bundle.date_val,
                      bundle.station_ids, bundle.y_raw_val, val_pred)
    write_json(output / "metrics.json", result)
    write_json(output / "run_status.json", {"status": "complete", "model": "persistence"})
    return result


if __name__ == "__main__":
    run()
