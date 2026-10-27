"""Step 1 sanity check: GBM, NO costs. The net should recover ~BS delta. If it doesn't, the pipeline is buggy."""
import numpy as np
from dh.experiment import Regime, make_sim
from dh.baselines import bs_call_price, delta_hedge
from dh.deep import train, net_positions
from dh.pnl import hedge_pnl, metrics

r = Regime(model="gbm", cost=0.0, n_steps=30, T_days=30); T = r.T_days / 252; sim = make_sim(r)
from dh.payoffs import payoff
pf = lambda S: payoff(S, 'european', r.K)
prem = float(bs_call_price(100.0, r.K, T, r.sigma))
net = train(sim, pf, r.K, T, 0.0, prem, steps=2000)
S, V = sim(50_000, 999)
dn, dd = net_positions(net, S, V, r.K, T), delta_hedge(S, r.K, T, r.sigma)
print("mean |net - BS delta|:", np.abs(dn - dd).mean().round(4), "(want < ~0.03)")
for n, d in [("net", dn), ("BS delta", dd)]:
    print(n, {k: round(float(v), 3) for k, v in metrics(*hedge_pnl(S, d, pf(S), 0.0, prem)).items()})
