from __future__ import annotations
import argparse
from pathlib import Path
from exp.architecture_matching.run_macro_2x2 import run_one
ROOT=Path(__file__).resolve().parent
PACKAGE_ROOT=ROOT.parents[1]
VARIANTS=("tcn_base","lightmicn_base","fixed_tcn","fixed_lightmicn","adgl_tcn","adgl_lightmicn")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",choices=("cpu","gpu"),default="gpu"); ap.add_argument("--preflight-only",action="store_true"); a=ap.parse_args()
    for name in VARIANTS: run_one(name,PACKAGE_ROOT/"test_results"/"architecture_matching"/"preflight"/name,a.device,check_only=True)
    if a.preflight_only:return
    for name in VARIANTS: run_one(name,PACKAGE_ROOT/"results"/"architecture_matching"/"runs"/name,a.device)
if __name__=="__main__":main()
