"""Study B - robustness to misspecification. Everything is calibrated/trained at a NOMINAL regime, then evaluated,
FROZEN, in shifted ("stress") regimes. Baselines keep assuming nominal sigma and nominal cost (like a desk that has not
noticed the market changed). Reference-only: delta with the TRUE sigma (unattainable in practice)."""
from dataclasses import replace, asdict
import numpy as np
from .experiment import Regime, make_sim, boot_diff
from .payoffs import payoff
from .baselines import delta_hedge, leland_hedge, whalley_wilmott_hedge
from .pnl import hedge_pnl, metrics
from .tune import tune_ww, tune_leland, tune_sigma

NOMINAL = Regime(model="heston", payoff="european", cost=0.001, xi=0.3, sigma=0.2, n_steps=30, T_days=30)
# Support of the randomised TRAINING distribution (for the 'random' nets). Test cells outside it are flagged.
SUPPORT = dict(sigma=(0.12, 0.40), xi=(0.2, 1.8), cost=(0.0005, 0.01), lam=(0.0, 15.0))


def in_support(r):
    ok = SUPPORT["sigma"][0] <= r.sigma <= SUPPORT["sigma"][1] and SUPPORT["cost"][0] <= r.cost <= SUPPORT["cost"][1]
    if r.model == "heston": ok = ok and SUPPORT["xi"][0] <= r.xi <= SUPPORT["xi"][1]
    return bool(ok)


def sample_train_regimes(rng, nom=NOMINAL, m=4):
    regs = []
    for _ in range(m):
        sigma = float(rng.choice([0.12, 0.16, 0.2, 0.3, 0.4]))
        cost = float(np.exp(rng.uniform(np.log(SUPPORT["cost"][0]), np.log(SUPPORT["cost"][1]))))
        if rng.random() < 0.75:
            regs.append(replace(nom, model="heston", sigma=sigma, xi=float(rng.uniform(*SUPPORT["xi"])), cost=cost))
        else:
            lam = min(float(rng.uniform(1, SUPPORT["lam"][1])), 0.8 * sigma ** 2 / 0.002)   # jump variance must stay < total variance
            regs.append(replace(nom, model="merton", sigma=sigma, lam=lam, cost=cost))
    return regs


def stress_cells(nom=NOMINAL):
    cells = [("nominal", 0.0, nom)]
    cells += [("vol", s, replace(nom, sigma=s)) for s in (0.1, 0.15, 0.3, 0.4, 0.5)]
    cells += [("volvol", x, replace(nom, xi=x)) for x in (0.1, 0.8, 1.5, 2.2)]
    cells += [("jumps", float(l), replace(nom, model="merton", lam=float(l))) for l in (5, 10, 15)]
    cells += [("cost", c, replace(nom, cost=c)) for c in (0.0005, 0.003, 0.01, 0.02)]
    cells += [("crisis", 0.0, replace(nom, sigma=0.35, xi=1.5, cost=0.005))]
    return cells


def run_robustness(train_steps=3000, n_test=50_000, use_torch=True, oracle=False, seed=0, log=print):
    nom = NOMINAL; T = nom.T_days / 252; K = nom.K; sim_nom = make_sim(nom)
    pf = lambda S: payoff(S, "european", K)
    prem = float(pf(sim_nom(200_000, 555)[0]).mean())
    Sv, _ = sim_nom(40_000, 777); Pv = pf(Sv)
    g = tune_ww(Sv, Pv, K, T, nom.sigma, nom.cost, prem); m = tune_leland(Sv, Pv, K, T, nom.sigma, nom.cost, prem)
    ms = tune_sigma(Sv, Pv, K, T, nom.sigma, nom.cost, prem)
    nets = {}
    if use_torch:
        from .deep import train, train_mixture
        log("training net_nominal"); nets["net_nominal"] = train(sim_nom, pf, K, T, nom.cost, prem, steps=train_steps, seed=seed, verbose=False)
        smp = lambda rng: sample_train_regimes(rng, nom)
        log("training net_random"); nets["net_random"] = train_mixture(smp, make_sim, pf, K, T, prem, use_rv=False, steps=train_steps, seed=seed, verbose=False)
        log("training net_random_rv"); nets["net_random_rv"] = train_mixture(smp, make_sim, pf, K, T, prem, use_rv=True, steps=train_steps, seed=seed, verbose=False)
    out = []
    for axis, val, r in stress_cells():
        S, V = make_sim(r)(n_test, 999); P = pf(S)
        prem_c = float(P.mean())   # price each cell FAIRLY: CVaR then measures hedging error, not mispricing profit/loss
        d_nom = delta_hedge(S, K, T, nom.sigma)
        strat = {"delta": d_nom,
                 "leland": leland_hedge(S, K, T, nom.sigma, nom.cost),
                 "infl_delta_tuned": leland_hedge(S, K, T, nom.sigma, nom.cost * m) if m > 0 else d_nom,
                 "delta_sigma_tuned": delta_hedge(S, K, T, nom.sigma * ms),
                 "ww": whalley_wilmott_hedge(S, K, T, nom.sigma, nom.cost, g)}
        if use_torch:
            from .deep import net_positions, train
            for k, net in nets.items(): strat[k] = net_positions(net, S, V, K, T)
            strat["net_nominal_shrunk"] = 0.5 * strat["net_nominal"] + 0.5 * d_nom       # 50/50 blend with plain delta
            if oracle:
                log(f"  oracle net for {axis}={val}")
                strat["net_oracle"] = net_positions(train(make_sim(r), pf, K, T, r.cost, prem, steps=train_steps, seed=seed, verbose=False), S, V, K, T)
        ref = {"ref_delta_true_sigma": delta_hedge(S, K, T, r.sigma)}        # NOT a competitor: needs knowledge a desk lacks
        res = {k: hedge_pnl(S, d, P, r.cost, prem_c, 0.0) for k, d in {**strat, **ref}.items()}
        rows = {k: metrics(*v) for k, v in res.items()}
        best = min((k for k in strat if not k.startswith("net")), key=lambda k: rows[k]["cvar95"])
        vs = {}
        for k in strat:
            if k.startswith("net"):
                mean, lo, hi = boot_diff(res[k][0], res[best][0])
                vs[k] = dict(cvar_diff=mean, ci_lo=lo, ci_hi=hi, verdict="NET WINS" if hi < 0 else "NET LOSES" if lo > 0 else "tie")
        out.append(dict(axis=axis, value=val, in_support=in_support(r), regime=asdict(r),
                        tuned=dict(ww_gamma=float(g), leland_mult=float(m), sigma_mult=float(ms)), best_frozen_baseline=best, metrics=rows, vs_best=vs))
        log(f"{axis:7s} {val:<8} best frozen baseline: {best:17s} " + " ".join(f"{k}:{v['verdict']}" for k, v in vs.items()))
    return out
