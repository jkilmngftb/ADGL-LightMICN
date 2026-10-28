"""Run the missing fixed-GCN-TCN cell of the graph-type x encoder table."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from exp.architecture_matching.run_macro_2x2 import run_one


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
PREFLIGHT = ("fixed_tcn", "fixed_lightmicn")
MISSING = ("fixed_tcn",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    options = parser.parse_args()

    for variant in PREFLIGHT:
        run_one(
            variant,
            PACKAGE_ROOT / "test_results" / "architecture_matching" / f"preflight_graph_encoder_{options.device}" / variant,
            options.device,
            check_only=True,
        )
    if options.preflight_only:
        return

    for variant in MISSING:
        run_one(
            variant,
            PACKAGE_ROOT / "results" / "architecture_matching" / "runs" / "graph_encoder_factorial" / variant,
            options.device,
        )

    if options.evaluate_test:
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "evaluate_graph_encoder_factorial.py"),
             "--device", options.device],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
