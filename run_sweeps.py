"""One-factor-at-a-time sweeps around a base regime -> results/*.json and a summary table.
Usage: python run_sweeps.py --quick   (small, minutes)     python run_sweeps.py   (full)"""
import argparse, json, os
from dh.experiment import Regime, run_regime

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--no-torch", action="store_true")
a = ap.parse_args()
steps, ntest = (400, 30_000) if a.quick else (2000, 100_000)

SWEEPS = {
    "cost":   [Regime(cost=c) for c in (0.0, 0.0005, 0.001, 0.003, 0.01)],
    "volvol": [Regime(xi=x) for x in (0.1, 0.5, 1.0, 1.5)],
    "rebal":  [Regime(n_steps=n) for n in (5, 15, 30, 60)],
    "maturity": [Regime(T_days=d, n_steps=max(d, 5)) for d in (10, 30, 90)],
}
os.makedirs("results", exist_ok=True)
for name, regs in SWEEPS.items():
    rows = []
    for r in regs:
        out = run_regime(r, train_steps=steps, n_test=ntest, use_torch=not a.no_torch)
        rows.append(out)
        print(name, {k: v for k, v in out["regime"].items() if k in ("cost", "xi", "n_steps", "T_days")},
              "best baseline:", out["best_baseline"], {k: v["verdict"] for k, v in out["vs_best"].items()})
    json.dump(rows, open(f"results/{name}.json", "w"), indent=1, default=float)
