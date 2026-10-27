"""Payoffs on discretely monitored paths S[:, 0..n]. Computed in numpy (no gradient needed: payoff doesn't depend on the net)."""
import numpy as np

KINDS = ("european", "asian", "barrier", "lookback")


def payoff(S, kind="european", K=100.0, barrier=115.0):
    ST = S[:, -1]
    if kind == "european":   return np.maximum(ST - K, 0.0)
    if kind == "asian":      return np.maximum(S[:, 1:].mean(1) - K, 0.0)                     # arithmetic average call
    if kind == "barrier":    return np.where(S.max(1) < barrier, np.maximum(ST - K, 0.0), 0.0)  # up-and-out call
    if kind == "lookback":   return ST - S.min(1)                                               # floating-strike lookback call
    raise ValueError(kind)
