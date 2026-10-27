"""One regime -> train net, tune baselines, evaluate on held-out paths, bootstrap the verdict."""
from dataclasses import dataclass, asdict
import numpy as np
from .sim import simulate_gbm, simulate_heston, simulate_merton
from .payoffs import payoff as payoff_fn
from .baselines import delta_hedge, leland_hedge, whalley_wilmott_hedge
from .pnl import hedge_pnl, metrics
from .tune import tune_ww, tune_leland, tune_sigma


@dataclass
class Regime:
    model: str = "heston"      # "gbm" | "heston" | "merton"
    payoff: str = "european"   # "european" | "asian" | "barrier" | "lookback"
    cost: float = 0.001        # one-way proportional cost (0.001 = 10 bps)
    impact: float = 0.0        # quadratic market-impact coefficient
    xi: float = 0.5            # vol-of-vol (heston only)
    lam: float = 5.0           # jump intensity per year (merton only)
    n_steps: int = 30          # rebalancing dates
    T_days: int = 30           # maturity in trading days (252/yr)
    sigma: float = 0.2         # base vol (total vol for merton, sqrt(v0)=sqrt(theta) for heston)
    K: float = 100.0
    barrier: float = 115.0


def make_sim(r: Regime):
    T = r.T_days / 252
    if r.model == "gbm":
        return lambda n, seed: simulate_gbm(n, r.n_steps, T, r.sigma, seed=seed)
    if r.model == "merton":
        return lambda n, seed: simulate_merton(n, r.n_steps, T, sigma_total=r.sigma, lam=r.lam, seed=seed)
    return lambda n, seed: simulate_heston(n, r.n_steps, T, v0=r.sigma**2, theta=r.sigma**2, xi=r.xi, seed=seed)


def cvar95(pnl):
    L = -pnl; return L[L >= np.quantile(L, 0.95)].mean()


def boot_diff(pnl_a, pnl_b, B=300, seed=0):
    """Paired bootstrap of CVaR95(a) - CVaR95(b). Negative => a has lower risk (better) than b."""
    rng = np.random.default_rng(seed); n = len(pnl_a); out = []
    for _ in range(B):
        i = rng.integers(0, n, n); out.append(cvar95(pnl_a[i]) - cvar95(pnl_b[i]))
    return np.mean(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


def run_regime(r: Regime, train_steps=1500, n_test=100_000, seeds=(0,), use_torch=True):
    T = r.T_days / 252; K = r.K; sim = make_sim(r)
    pf = lambda S: payoff_fn(S, r.payoff, K, r.barrier)
    Sv, _ = sim(40_000, 777); Pv = pf(Sv)
    prem = float(pf(sim(200_000, 555)[0]).mean())               # Monte-Carlo premium (mean-P&L bookkeeping only; CVaR ignores it)
    g = tune_ww(Sv, Pv, K, T, r.sigma, r.cost, prem, r.impact)
    m = tune_leland(Sv, Pv, K, T, r.sigma, r.cost, prem, r.impact)
    ms = tune_sigma(Sv, Pv, K, T, r.sigma, r.cost, prem, r.impact)
    St, Vt = sim(n_test, 999); Pt = pf(St)
    # NOTE: for non-European payoffs these baselines hedge with the VANILLA call delta (a proxy, weak by construction).
    strategies = {"delta": delta_hedge(St, K, T, r.sigma),
                  "leland": leland_hedge(St, K, T, r.sigma, r.cost) if r.cost > 0 else delta_hedge(St, K, T, r.sigma),
                  "infl_delta_tuned": leland_hedge(St, K, T, r.sigma, r.cost * m) if m > 0 else delta_hedge(St, K, T, r.sigma),
                  "delta_sigma_tuned": delta_hedge(St, K, T, r.sigma * ms),
                  "ww": whalley_wilmott_hedge(St, K, T, r.sigma, r.cost, g)}
    if use_torch:
        from .deep import train, net_positions
        for uv in ([False, True] if r.model == "heston" else [False]):
            for s in seeds:
                net = train(sim, pf, K, T, r.cost, prem, impact=r.impact, use_var=uv, path_dependent=r.payoff != 'european', steps=train_steps, seed=s, verbose=False)
                strategies[f"net{'_var' if uv else ''}_s{s}"] = net_positions(net, St, Vt, K, T)
    res = {k: hedge_pnl(St, d, Pt, r.cost, prem, r.impact) for k, d in strategies.items()}
    rows = {k: metrics(*v) for k, v in res.items()}
    best = min((k for k in rows if not k.startswith("net")), key=lambda k: rows[k]["cvar95"])
    edge = bool(ms >= 2.0 or m >= 8.0 or g >= 100.0 or (g <= 0.1 and r.cost > 0))
    out = dict(regime=asdict(r), tuned=dict(ww_gamma=float(g), leland_mult=float(m), sigma_mult=float(ms), grid_edge_warning=edge),
               proxy_baselines=r.payoff != "european", best_baseline=best, metrics=rows, vs_best={})
    for k in res:
        if k.startswith("net"):
            mean, lo, hi = boot_diff(res[k][0], res[best][0])
            out["vs_best"][k] = dict(cvar_diff=mean, ci_lo=lo, ci_hi=hi,
                                     verdict="NET WINS" if hi < 0 else "NET LOSES" if lo > 0 else "tie (CI includes 0)")
    return out
