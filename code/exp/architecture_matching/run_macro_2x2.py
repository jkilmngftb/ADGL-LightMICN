"""Run only the two missing cells of the frozen ADGL x backbone 2x2 design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
TRAIN = ROOT / "train_variant.py"
CONFIG = ROOT / "configs" / "final_selected_config.json"
MISSING = ("tcn_base", "lightmicn_base")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def command_for(variant, output, device, check_only=False):
    config = read_json(CONFIG)["config"]
    command = [
        sys.executable, "-u", str(TRAIN),
        "--variant", variant,
        "--seed", str(config["seed"]),
        "--device", device,
        "--seq-len", str(config["window"]),
        "--common-window", str(config["common_window"]),
        "--learning-rate", str(config["learning_rate"]),
        "--kernels", *(str(value) for value in config["kernels"]),
        "--layers", str(config["layers"]),
        "--epochs", str(config["epochs"]),
        "--output", str(output),
    ]
    if check_only:
        command.append("--check-only")
    return command


def run_one(variant, output, device, check_only=False):
    complete_marker = output / ("run_status.json" if check_only else "metrics.json")
    if complete_marker.is_file():
        state = read_json(output / "run_status.json")
        expected = "check_passed" if check_only else "complete"
        if state.get("state") == expected:
            print(f"SKIP completed {variant}: {output}", flush=True)
            return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial output retained; refusing silent retry: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / f"{variant}.log"
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(
            command_for(variant, output, device, check_only),
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"COMPLETE {variant}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    options = parser.parse_args()

    # Preflight is isolated from formal result directories.
    for variant in MISSING:
        run_one(variant, PACKAGE_ROOT / "test_results" / "architecture_matching" / "preflight" / variant, options.device, check_only=True)
    if options.preflight_only:
        return

    for variant in MISSING:
        run_one(variant, PACKAGE_ROOT / "results" / "architecture_matching" / "runs" / variant, options.device)

    if options.evaluate_test:
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "evaluate_macro_2x2.py"),
             "--device", options.device],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
