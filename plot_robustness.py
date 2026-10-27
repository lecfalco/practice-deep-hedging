"""results/robustness.json -> results/robustness.png. y = CVaR95 / CVaR95 of the best frozen baseline in that cell
(1.0 = parity, below 1 = better than the best frozen baseline). Open markers = outside the randomised training support."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/robustness.json")); nominal = next(c for c in R if c["axis"] == "nominal")
axes = [("vol", "true volatility (baselines assume 0.20)"), ("volvol", "vol-of-vol xi (nominal 0.3)"),
        ("jumps", "jump intensity per year (0 = no jumps)"), ("cost", "true one-way cost (baselines assume 0.001)")]
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
for a, (name, label) in zip(ax.ravel(), axes):
    cells = [c for c in R if c["axis"] == name]
    nominal_x = {"vol": 0.2, "volvol": 0.3, "jumps": 0.0, "cost": 0.001}[name]
    cells = sorted(cells + [dict(nominal, value=nominal_x)], key=lambda c: c["value"])
    keys = [k for k in cells[0]["metrics"]]
    for k in keys:
        y = [c["metrics"][k]["cvar95"] / c["metrics"][c["best_frozen_baseline"]]["cvar95"] for c in cells]
        x = [c["value"] for c in cells]
        ls = "--" if k.startswith("ref") else "-"
        a.plot(x, y, ls, label=k, lw=1.5 if not k.startswith("net") else 2.2, alpha=0.9)
        for xi, yi, c in zip(x, y, cells):
            a.plot([xi], [yi], "o", ms=5, mfc="none" if not c["in_support"] else None, color=a.lines[-1].get_color())
    a.axhline(1.0, color="k", lw=0.6); a.set_xlabel(label); a.set_ylabel("CVaR95 / best frozen baseline")
    if name in ("cost", "vol"): a.set_xscale("log")
a = ax[0, 0]; a.legend(fontsize=7, ncol=2)
fig.suptitle("Robustness: strategies calibrated at nominal, tested frozen under shifted markets", fontsize=10)
fig.tight_layout(); fig.savefig("results/robustness.png", dpi=130); print("saved results/robustness.png")
