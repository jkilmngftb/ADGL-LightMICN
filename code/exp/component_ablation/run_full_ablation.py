"""Run the complete ablation suite from the immutable selected configuration."""
from datetime import datetime
import argparse
import json

from exp.component_ablation.workflow_utils import (
    ROOT, PACKAGE_ROOT, evaluate_checkpoint, read_json, run_variant, write_csv, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    frozen_path = ROOT / "selection" / "final_selected_config.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Freeze final_selected_config.json before full ablation")
    frozen = read_json(frozen_path)
    if not frozen.get("selection_frozen"):
        raise ValueError("Final configuration is not frozen")
    config = frozen["config"]
    from exp.component_ablation.variants import SPECS
    precheck_rows = {row["variant"]: row for row in frozen["precheck_rows"]}
    output_root = PACKAGE_ROOT / "results" / "component_ablation" / "full_ablation" / config["config_id"]
    trained_rows = []
    for variant in SPECS:
        if variant == "full":
            result_dir = PACKAGE_ROOT / precheck_rows["full"]["result_dir"]
            trained = {
                "variant": "full",
                "validation_mae": precheck_rows["full"]["validation_mae"],
                "validation_rmse": precheck_rows["full"]["validation_rmse"],
                "validation_r2": precheck_rows["full"]["validation_r2"],
                "effective_kernels": config["kernels"],
                "result_dir": precheck_rows["full"]["result_dir"],
            }
        else:
            result_dir = PACKAGE_ROOT / "results" / "component_ablation" / "candidate_precheck" / config["config_id"] / variant
            trained = run_variant(config, variant, result_dir, args.device)
        trained_rows.append((variant, result_dir, trained))

    # Start evaluation only after every missing training subprocess has exited,
    # so the parent process does not retain GPU memory while a child trains.
    rows = []
    for variant, result_dir, trained in trained_rows:
        evaluated = evaluate_checkpoint(result_dir, variant, args.device, align_test=True)
        rows.append({
            **trained,
            "delta_validation_rmse": float(trained["validation_rmse"] - frozen["full_validation_rmse"]),
            "aligned_test_mae": float(evaluated["test"]["mae"]),
            "aligned_test_rmse": float(evaluated["test"]["rmse"]),
            "aligned_test_r2": float(evaluated["test"]["r2"]),
        })
    full_test = next(row["aligned_test_rmse"] for row in rows if row["variant"] == "full")
    for row in rows:
        row["delta_test_rmse"] = float(row["aligned_test_rmse"] - full_test)
    payload = {
        "created_at": datetime.now().isoformat(),
        "config": config,
        "read_from": "selection/final_selected_config.json",
        "fixed_graph_interpretation": "固定图替代动态图，并移除仅作用于动态图 gate 的 L1 项",
        "rows": rows,
    }
    write_json(output_root / "full_ablation_summary.json", payload)
    write_csv(
        output_root / "full_ablation_summary.csv", rows,
        ["variant", "validation_mae", "validation_rmse", "validation_r2", "delta_validation_rmse",
         "aligned_test_mae", "aligned_test_rmse", "aligned_test_r2", "delta_test_rmse",
         "effective_kernels", "parameters", "result_dir"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
