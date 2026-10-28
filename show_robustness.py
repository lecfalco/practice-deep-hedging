"""Print results/robustness.json as a compact table.  Usage (project folder):  python show_robustness.py
Each network column shows: change in CVaR95 vs the best FROZEN baseline in that cell, as a percentage
[95% bootstrap interval], and the verdict. Negative = network has LOWER tail risk (better)."""
import json

R = json.load(open("results/robustness.json"))
nets = sorted({k for c in R for k in c["vs_best"]})
print(f"{'axis':8s}{'value':>8s} {'in_supp':>8s} {'best frozen baseline':>22s} {'CVaR95':>7s} | " + " | ".join(nets))
for c in R:
    b = c["metrics"][c["best_frozen_baseline"]]["cvar95"]
    cols = []
    for k in nets:
        v = c["vs_best"].get(k)
        if v:
            p = lambda x: 100 * x / b
            verdict = v['verdict'][4:] if v['verdict'].startswith("NET ") else v['verdict']
            cols.append(f"{p(v['cvar_diff']):+6.1f}% [{p(v['ci_lo']):+6.1f},{p(v['ci_hi']):+6.1f}] {verdict or 'tie'}")
        else:
            cols.append("-")
    print(f"{c['axis']:8s}{str(c['value']):>8s} {str(c['in_support']):>8s} {c['best_frozen_baseline']:>22s} {b:7.3f} | " + " | ".join(cols))
print("\nin_supp = False: the shift lies outside the range the randomised-training networks saw.")