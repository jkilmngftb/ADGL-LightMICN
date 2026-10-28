"""Dispatch the paper experiment groups from the package root."""
from __future__ import annotations

import argparse
import subprocess
import sys


COMMANDS = {
    "architecture": "exp.architecture_matching.run_all_architecture_matching",
    "component": "exp.component_ablation.run_full_ablation",
    "error": "exp.error_dynamic_graph.analysis_error_breakdown",
    "dynamic-graph": "exp.error_dynamic_graph.analysis_dynamic_graph",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=COMMANDS, nargs="?")
    parser.add_argument("--list", action="store_true", help="list available experiment groups")
    options, extra = parser.parse_known_args()
    if options.list:
        print("\n".join(COMMANDS))
        return
    if options.experiment is None:
        parser.error("provide an experiment name or use --list")
    subprocess.run([sys.executable, "-m", COMMANDS[options.experiment], *extra], check=True)


if __name__ == "__main__":
    main()
