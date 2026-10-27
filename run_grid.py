"""2D regime grids -> results/grid_<name>.json.  Usage: python run_grid.py [--quick] [--no-torch]
Grids: cost x n_steps (european/heston), xi x payoff, lam x payoff (merton)."""
import argparse, json, os, itertools
from dh.experiment import Regime, run_regime

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--no-torch", action="store_true")
a = ap.parse_args()
steps, ntest = (400, 30_000) if a.quick else (2000, 100_000)
GRIDS = {
    "cost_x_rebal": ("cost", (0.0, 0.0005, 0.002, 0.01), "n_steps", (10, 30, 60),
                     lambda c, n: Regime(model="heston", cost=c, n_steps=n)),
    "volvol_x_payoff": ("xi", (0.2, 0.8, 1.5), "payoff", ("european", "asian", "barrier", "lookback"),
                        lambda x, p: Regime(model="heston", xi=x, payoff=p)),
    "jumps_x_payoff": ("lam", (0.0001, 5.0, 15.0), "payoff", ("european", "asian", "barrier", "lookback"),
                       lambda l, p: Regime(model="merton", lam=l, payoff=p)),
}
os.makedirs("results", exist_ok=True)
for name, (ka, va, kb, vb, mk) in GRIDS.items():
    cells = []
    for x, y in itertools.product(va, vb):
        out = run_regime(mk(x, y), train_steps=steps, n_test=ntest, use_torch=not a.no_torch)
        out["axes"] = {ka: x, kb: y}; cells.append(out)
        print(name, ka, x, kb, y, "best:", out["best_baseline"], {k: v["verdict"] for k, v in out["vs_best"].items()})
    json.dump(dict(axes=[ka, kb], values=[list(va), list(vb)], cells=cells), open(f"results/grid_{name}.json", "w"), indent=1, default=float)
