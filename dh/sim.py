"""Market simulators (numpy). Rates = 0, no dividends, to keep the maths clean."""
import numpy as np


def simulate_gbm(n_paths, n_steps, T, sigma, S0=100.0, seed=0):
    """Geometric Brownian motion. Returns S with shape (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    logret = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
    S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logret, 1)], 1))
    return S, np.full_like(S, sigma**2)


def simulate_heston(n_paths, n_steps, T, v0=0.04, kappa=2.0, theta=0.04, xi=0.5,
                    rho=-0.7, S0=100.0, seed=0, substeps=4):
    """Heston stochastic vol, full-truncation Euler. xi = vol-of-vol.
    Returns S, V with shape (n_paths, n_steps+1). Simulated on a finer grid, sampled at hedge dates."""
    rng = np.random.default_rng(seed)
    m = n_steps * substeps
    dt = T / m
    logS = np.zeros(n_paths)
    v = np.full(n_paths, v0)
    S_out = np.empty((n_paths, n_steps + 1)); V_out = np.empty_like(S_out)
    S_out[:, 0] = S0; V_out[:, 0] = v0
    for i in range(m):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        logS += -0.5 * vp * dt + np.sqrt(vp * dt) * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        if (i + 1) % substeps == 0:
            j = (i + 1) // substeps
            S_out[:, j] = S0 * np.exp(logS); V_out[:, j] = np.maximum(v, 0.0)
    return S_out, V_out


def simulate_merton(n_paths, n_steps, T, sigma_total=0.2, lam=5.0, mu_j=-0.02, sig_j=0.04, S0=100.0, seed=0):
    """Merton jump-diffusion, martingale-corrected. Diffusion vol is set so TOTAL variance rate = sigma_total^2.
    Returns S and a constant 'variance' array (interface compatibility with Heston)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sd2 = sigma_total**2 - lam * (mu_j**2 + sig_j**2)
    assert sd2 > 0, "jump variance exceeds total variance"
    k = np.exp(mu_j + 0.5 * sig_j**2) - 1
    drift = (-0.5 * sd2 - lam * k) * dt
    N = rng.poisson(lam * dt, (n_paths, n_steps))
    jumps = rng.standard_normal((n_paths, n_steps)) * np.sqrt(N) * sig_j + N * mu_j
    logret = drift + np.sqrt(sd2 * dt) * rng.standard_normal((n_paths, n_steps)) + jumps
    S = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logret, 1)], 1))
    return S, np.full_like(S, sigma_total**2)
