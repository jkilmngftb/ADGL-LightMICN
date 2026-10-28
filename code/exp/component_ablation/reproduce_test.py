"""Reproduce validation/test metrics and predictions from a saved checkpoint.

The training configuration is defined in utils/protocol.py, so you only need to
point this at a run directory containing best.weights.h5. Uses the same evaluation path as the pipeline
(workflow_utils.evaluate_checkpoint), so the numbers match the recorded results.

Examples
--------
# Frozen final MICN, aligned Test metrics + predictions CSV
python reproduce_test.py --result-dir results/component_ablation/runs/w12_k2-4_lr0p0010_layers1 --variant full --device gpu

# A single-scale ablation, both val and test
python reproduce_test.py --result-dir results/component_ablation/candidate_precheck/w12_k2-4_lr0p0010_layers1/single_scale --variant single_scale --device gpu --split both
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

from exp.component_ablation.workflow_utils import PACKAGE_ROOT, RESULTS_ROOT, CHECKPOINT_ROOT, evaluate_checkpoint, model_config_values


DEFAULT_RESULT_DIR = "results/component_ablation/runs/w12_k2-4_lr0p0010_layers1"


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate a saved MICN checkpoint.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR,
                        help="directory containing best.weights.h5 (relative to the project root)")
    parser.add_argument("--variant", default="full",
                        choices=("full", "fixed_graph", "no_decomposition", "single_scale",
                                 "no_top_k", "no_l1", "graph_node_embedding"),
                        help="paper-reported component configuration")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--split", choices=("test", "val", "both"), default="test")
    parser.add_argument("--predictions-csv", default=None,
                        help="output CSV path (default: <result-dir>/predictions_<variant>_<split>.csv)")
    parser.add_argument("--metrics-json", default=None,
                        help="output metrics JSON path (default: <result-dir>/reproduced_metrics_<variant>.json)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = PACKAGE_ROOT / result_dir

    evaluated = evaluate_checkpoint(result_dir, args.variant, args.device, align_test=True)
    summary = {
        "checkpoint_dir": str(result_dir),
        "variant": args.variant,
        "config": model_config_values(args.variant),
        "parameters": evaluated["parameters"],
        "checkpoint_sha256": evaluated["checkpoint_sha256"],
        "test_dates_sha256": evaluated["test_dates_sha256"],
        "validation": evaluated["validation"],
        "test": evaluated["test"],
    }

    if args.split == "test":
        splits = ["test"]
    elif args.split == "val":
        splits = ["val"]
    else:
        splits = ["val", "test"]

    bundle = evaluated["bundle"]
    predictions = evaluated["predictions"]
    csv_path = Path(args.predictions_csv) if args.predictions_csv else (
        result_dir / f"predictions_{args.variant}_{args.split}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "date", "node", "y_true", "y_pred", "residual"])
        for split in splits:
            dates = getattr(bundle, "date_" + split)
            truth = getattr(bundle, "y_raw_" + split)
            pred = predictions[split]
            for i, date in enumerate(dates):
                for node in range(bundle.num_nodes):
                    true_value = float(truth[i, node])
                    pred_value = float(pred[i, node])
                    writer.writerow([split, str(date), node, true_value, pred_value, true_value - pred_value])
                    n_rows += 1

    json_path = Path(args.metrics_json) if args.metrics_json else (
        result_dir / f"reproduced_metrics_{args.variant}.json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[written] predictions CSV: {csv_path}  ({n_rows} rows)", flush=True)
    print(f"[written] metrics JSON:    {json_path}", flush=True)


if __name__ == "__main__":
    main()
