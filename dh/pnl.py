"""P&L, costs and risk metrics (numpy), used for ALL strategies incl. the net's output."""
import numpy as np


def hedge_pnl(S, delta, payoff, cost, premium, impact=0.0):
    """Seller of a claim: receives premium, pays payoff, holds delta shares each step.
    Cost per trade of size x at price S: cost*S*|x| (proportional, one-way) + impact*S*x^2 (quadratic market impact).
    Includes initial purchase and final unwind."""
    gains = (delta * (S[:, 1:] - S[:, :-1])).sum(1)
    z = np.zeros((S.shape[0], 1))
    trades = np.abs(np.concatenate([delta, z], 1) - np.concatenate([z, delta], 1))
    costs = (cost * S * trades + impact * S * trades**2).sum(1)
    return premium - payoff + gains - costs, costs, trades.sum(1)


def metrics(pnl, costs, turnover, alpha=0.95):
    loss = -pnl
    var = np.quantile(loss, alpha)
    return dict(mean_pnl=pnl.mean(), std=pnl.std(), cvar95=loss[loss >= var].mean(),
                mean_cost=costs.mean(), turnover=turnover.mean())
