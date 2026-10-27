"""Analytic hedging strategies. All return a position array delta[:, t] for t=0..n_steps-1
(shares held during step t -> t+1), for a SHORT call that we are hedging (so delta is in [0,1])."""
import numpy as np
from scipy.stats import norm


def bs_d1(S, K, tau, sigma):
    tau = np.maximum(tau, 1e-8)
    return (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))


def bs_delta(S, K, tau, sigma):
    return norm.cdf(bs_d1(S, K, tau, sigma))


def bs_gamma(S, K, tau, sigma):
    tau = np.maximum(tau, 1e-8)
    return norm.pdf(bs_d1(S, K, tau, sigma)) / (S * sigma * np.sqrt(tau))


def bs_call_price(S, K, tau, sigma):
    d1 = bs_d1(S, K, tau, sigma); d2 = d1 - sigma * np.sqrt(np.maximum(tau, 1e-8))
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def delta_hedge(S, K, T, sigma):
    n = S.shape[1] - 1
    tau = T - np.arange(n) * T / n
    return bs_delta(S[:, :n], K, tau[None, :], sigma)


def leland_hedge(S, K, T, sigma, cost):
    """Leland (1985): BS delta evaluated at an inflated vol. `cost` = one-way proportional cost,
    round-trip k = 2*cost. Le = sqrt(2/pi) * k / (sigma*sqrt(dt)); sigma_L^2 = sigma^2 (1 + Le)."""
    n = S.shape[1] - 1; dt = T / n
    Le = np.sqrt(2 / np.pi) * (2 * cost) / (sigma * np.sqrt(dt))
    sigma_l = sigma * np.sqrt(1 + Le)   # seller of the option hedges with an INFLATED vol (verify vs Leland 1985)
    tau = T - np.arange(n) * dt
    return bs_delta(S[:, :n], K, tau[None, :], sigma_l)


def whalley_wilmott_hedge(S, K, T, sigma, cost, gamma_ra=1.0, r=0.0):
    """Whalley-Wilmott (1997) no-trade band around BS delta, half-width
    H = (3/(2*gamma_ra) * exp(-r*tau) * cost * S * Gamma^2)^(1/3).
    gamma_ra = risk-aversion parameter (TUNE IT - see tune_ww)."""
    n = S.shape[1] - 1; dt = T / n
    tau = T - np.arange(n) * dt
    pos = np.zeros((S.shape[0], n))
    prev = bs_delta(S[:, 0], K, T, sigma)    # initial position = BS delta
    for t in range(n):
        d = bs_delta(S[:, t], K, tau[t], sigma)
        G = bs_gamma(S[:, t], K, tau[t], sigma)
        H = (1.5 / gamma_ra * np.exp(-r * tau[t]) * cost * S[:, t] * G**2) ** (1 / 3)
        prev = np.clip(prev, d - H, d + H)   # trade only to the nearest band edge
        pos[:, t] = prev
    return pos
