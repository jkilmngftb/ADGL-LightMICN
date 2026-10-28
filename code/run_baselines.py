from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from utils.protocol import MODEL_ORDER, RESULTS, SEED
ROOT=Path(__file__).resolve().parent
def complete(name):
    p=RESULTS/name/"run_status.json"
    return p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("status")=="complete"
def call(args):
    print("[run]", " ".join(map(str,args)), flush=True)
    subprocess.run(args,cwd=ROOT,check=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",choices=("gpu","cpu","auto"),default="gpu"); ap.add_argument("--skip-test",action="store_true"); a=ap.parse_args()
    call([sys.executable,"check_configuration.py"])
    if not complete("persistence"): call([sys.executable,"-m","exp.comparison.compute_persistence"])
    for model in MODEL_ORDER:
        if model == "persistence":
            continue
        if not complete(model): call([sys.executable,"-m","exp.comparison.train_neural","--model",model,"--device",a.device])
    if not a.skip_test: call([sys.executable,"-m","exp.comparison.evaluate_test","--device",a.device])
if __name__=="__main__": main()
