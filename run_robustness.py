"""Study B: train/calibrate at NOMINAL, test frozen under shifted regimes.
Usage: python run_robustness.py --quick | (full) | --no-torch | --oracle (also trains a specialist per cell; slow)"""
import argparse, json, os
from dh.robustness import run_robustness
ap = argparse.ArgumentParser()
for f in ("--quick", "--no-torch", "--oracle"): ap.add_argument(f, action="store_true")
a = ap.parse_args()
steps, ntest = (500, 20_000) if a.quick else (3000, 60_000)
res = run_robustness(train_steps=steps, n_test=ntest, use_torch=not a.no_torch, oracle=a.oracle)
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/robustness.json", "w"), indent=1, default=float)
