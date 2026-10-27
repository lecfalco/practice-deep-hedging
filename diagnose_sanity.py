"""Diagnose the sanity check (GBM, zero cost): is the net really matching Black-Scholes delta?
Usage (inside the project folder):  python diagnose_sanity.py [--steps 4000] [--seeds 0 1]
Prints, per training seed: position gap vs BS delta, CVaR95 of net vs delta with a bootstrap interval, and P&L std.
For the first seed it also shows WHERE (which time steps, which price levels) the net differs from delta.
NOTE: written without being able to run PyTorch; if it errors, paste me the error."""
import argparse
import numpy as np
from dh.experiment import Regime, make_sim, boot_diff, cvar95
from dh.baselines import bs_call_price, delta_hedge
from dh.payoffs import payoff
from dh.deep import train, net_positions
from dh.pnl import hedge_pnl
from dh.tune import tune_sigma

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
a = ap.parse_args()

r = Regime(model="gbm", cost=0.0, n_steps=30, T_days=30)
T = r.T_days / 252
sim = make_sim(r)
pf = lambda S: payoff(S, "european", r.K)
prem = float(bs_call_price(100.0, r.K, T, r.sigma))

S, V = sim(100_000, 999)                      # test paths never used in training
P = pf(S)
dd = delta_hedge(S, r.K, T, r.sigma)          # Black-Scholes delta positions
pnl_d = hedge_pnl(S, dd, P, 0.0, prem)[0]
Sv, _ = sim(100_000, 777)                     # validation paths: choose the volatility multiplier here, not on test paths
ms = tune_sigma(Sv, pf(Sv), r.K, T, r.sigma, 0.0, prem)
pnl_s = hedge_pnl(S, delta_hedge(S, r.K, T, r.sigma * ms), P, 0.0, prem)[0]
print(f"tuned-sigma delta (sigma x {ms:.1f}): CVaR95 {cvar95(pnl_s):.3f} | plain delta {cvar95(pnl_d):.3f}")

for seed in a.seeds:
    net = train(sim, pf, r.K, T, 0.0, prem, steps=a.steps, seed=seed, verbose=False)
    dn = net_positions(net, S, V, r.K, T)
    pnl_n = hedge_pnl(S, dn, P, 0.0, prem)[0]
    mean, lo, hi = boot_diff(pnl_n, pnl_d)    # CVaR95(net) - CVaR95(delta); interval containing 0 = statistical tie
    print(f"seed {seed}: mean|net-delta| {np.abs(dn - dd).mean():.4f} | "
          f"CVaR95 net {cvar95(pnl_n):.3f} delta {cvar95(pnl_d):.3f} | "
          f"diff {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] | std net {pnl_n.std():.3f} delta {pnl_d.std():.3f}")
    m2, lo2, hi2 = boot_diff(pnl_n, pnl_s)
    print(f"        vs tuned-sigma delta: diff {m2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]  (positive = network WORSE)")
    if seed == a.seeds[0]:
        print("  mean |net-delta| at time steps 0,3,6,...,27:",
              [round(float(np.abs(dn[:, t] - dd[:, t]).mean()), 3) for t in range(0, 30, 3)])
        m = S[:, :-1] / r.K
        for lo_, hi_ in [(0, .97), (.97, .99), (.99, 1.01), (1.01, 1.03), (1.03, 9)]:
            mask = (m >= lo_) & (m < hi_)
            print(f"  S/K in [{lo_}, {hi_}): n={mask.sum():7d}  mean delta {dd[mask].mean():.3f}  "
                  f"mean net {dn[mask].mean():.3f}  mean |diff| {np.abs(dn[mask] - dd[mask]).mean():.3f}")
