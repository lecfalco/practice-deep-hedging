"""Heatmaps from results/grid_*.json. If net results exist: CVaR gap (net - best baseline), annotated with verdict.
Otherwise: how much the best baseline improves on plain delta (sanity view; no net yet)."""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

for f in sorted(glob.glob("results/grid_*.json")):
    g = json.load(open(f)); ka, kb = g["axes"]; va, vb = g["values"]
    Z = np.full((len(vb), len(va)), np.nan); T = [[""] * len(va) for _ in vb]; has_net = False
    for c in g["cells"]:
        i, j = vb.index(c["axes"][kb]), va.index(c["axes"][ka]); m = c["metrics"]
        nets = {k: v for k, v in c["vs_best"].items() if "var" not in k} or c["vs_best"]
        if nets:
            has_net = True; k0 = min(nets, key=lambda k: nets[k]["cvar_diff"]); Z[i, j] = nets[k0]["cvar_diff"]
            T[i][j] = {"NET WINS": "W", "NET LOSES": "L"}.get(nets[k0]["verdict"], "=") + f"\n{Z[i, j]:+.2f}"
        else:
            Z[i, j] = 100 * (m[c["best_baseline"]]["cvar95"] / m["delta"]["cvar95"] - 1)
            T[i][j] = f"{c['best_baseline'][:5]}\n{Z[i, j]:+.0f}%"
    fig, ax = plt.subplots(figsize=(1.8 * len(va) + 2, 1.1 * len(vb) + 1.5))
    lim = np.nanmax(np.abs(Z)) or 1
    im = ax.imshow(Z, cmap="RdYlGn_r" if has_net else "Blues_r", vmin=-lim if has_net else None, vmax=lim if has_net else None, aspect="auto")
    ax.set_xticks(range(len(va)), [str(v) for v in va]); ax.set_yticks(range(len(vb)), [str(v) for v in vb])
    ax.set_xlabel(ka); ax.set_ylabel(kb)
    for i in range(len(vb)):
        for j in range(len(va)): ax.text(j, i, T[i][j], ha="center", va="center", fontsize=8)
    ax.set_title(("Net CVaR95 minus best baseline (neg = net better)" if has_net else "Best baseline vs plain delta, CVaR95 change") + "\n" + f.split("grid_")[1][:-5], fontsize=9)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f.replace(".json", ".png"), dpi=130); print("saved", f.replace(".json", ".png"))
