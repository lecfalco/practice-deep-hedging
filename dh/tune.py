"""Give the analytic baselines a FAIR shot: tune their free parameter on validation paths."""
import numpy as np
from .baselines import leland_hedge, whalley_wilmott_hedge, delta_hedge
from .pnl import hedge_pnl, metrics


def _cvar(S, d, payoff, cost, prem, impact):
    return metrics(*hedge_pnl(S, d, payoff, cost, prem, impact))["cvar95"]


def tune_ww(S, payoff, K, T, sigma, cost, prem, impact=0.0, grid=np.logspace(-1, 2, 13)):
    return min(grid, key=lambda g: _cvar(S, whalley_wilmott_hedge(S, K, T, sigma, cost, g), payoff, cost, prem, impact))


def tune_leland(S, payoff, K, T, sigma, cost, prem, impact=0.0, scales=np.linspace(0, 8, 17)):
    """Vol-inflation multiplier on the Leland cost term (1.0 = textbook, 0 = plain delta)."""
    f = lambda m: leland_hedge(S, K, T, sigma, cost * m) if m > 0 else delta_hedge(S, K, T, sigma)
    return min(scales, key=lambda m: _cvar(S, f(m), payoff, cost, prem, impact))


def tune_sigma(S, payoff, K, T, sigma, cost, prem, impact=0.0, mults=np.linspace(0.8, 2.0, 13)):
    """Plain BS delta evaluated at sigma*mult (free volatility multiplier, NO link to cost). A higher sigma flattens the
    delta curve, which lowers tail risk even with zero fees (see README: found via the zero-cost sanity check)."""
    return min(mults, key=lambda m: _cvar(S, delta_hedge(S, K, T, sigma * m), payoff, cost, prem, impact))
