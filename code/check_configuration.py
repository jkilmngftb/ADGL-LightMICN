from __future__ import annotations

import ast
import json
from pathlib import Path

from utils.protocol import FROZEN_PROTOCOL, MODEL_CONFIG, MODEL_ORDER, ROOT


def main():
    errors = []
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception as exc:
            errors.append(f"syntax: {path}: {exc}")
    required = [ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx",
                ROOT / "dataset" / "adjacency_matrix.csv",
                ROOT / "checkpoints" / "adgl_lightmicn" / "best.weights.h5"]
    for path in required:
        if not path.is_file():
            errors.append(f"missing: {path}")
    expected = {"seq_len": 12, "hidden_dim": 64, "dropout": 0.1,
                "learning_rate": 0.001, "top_k": 4,
                "sparse_lambda": 0.5, "seed": 2026}
    for key, value in expected.items():
        if MODEL_CONFIG.get(key) != value:
            errors.append(f"frozen main drift: {key}={MODEL_CONFIG.get(key)!r}, expected {value!r}")
    report = {"status": "PASS" if not errors else "FAIL",
              "baseline_models": list(MODEL_ORDER),
              "comparison_rows": len(MODEL_ORDER) + 1,
              "shared_protocol": FROZEN_PROTOCOL,
              "architecture_boundary": (
                  "graph propagation, calendar, residual, station-conditioned head, Top-k/L1 "
                  "and MICN decomposition belong only to the proposed model; baselines retain "
                  "their defining architectures"),
              "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
