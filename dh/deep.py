"""Deep hedging network + CVaR training (PyTorch). NOT YET EXECUTED by the author - run run_sanity.py first."""
import numpy as np
import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """Shared MLP applied at every step. Inputs: log-moneyness, time-to-maturity, previous position,
    optional: sqrt(variance) [use_var; NOT observable in real markets], running price-history features [n_extra],
    causal realised-vol estimate [use_rv; observable]. Output: position in [0,1] (hedging a short call)."""
    def __init__(self, use_var=False, width=64, n_extra=0, use_rv=False, nominal_sigma=0.2, rv_decay=0.8):
        super().__init__()
        self.use_var, self.n_extra, self.use_rv = use_var, n_extra, use_rv
        self.nominal_sigma, self.rv_decay = nominal_sigma, rv_decay
        d_in = 3 + int(use_var) + n_extra + int(use_rv)
        self.f = nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, width),
                               nn.ReLU(), nn.Linear(width, 1))

    def forward(self, S, V, K, T, extra=None):
        n = S.shape[1] - 1; dt = T / n
        prev = torch.zeros(S.shape[0])
        rv = torch.full((S.shape[0],), self.nominal_sigma ** 2)     # EWMA of squared returns / dt, starts at nominal
        out = []
        for t in range(n):
            if self.use_rv and t > 0:
                r = torch.log(S[:, t] / S[:, t - 1])
                rv = self.rv_decay * rv + (1 - self.rv_decay) * (r ** 2 / dt)
            tau = torch.full_like(prev, T * (1 - t / n))
            x = [torch.log(S[:, t] / K), tau, prev]
            if self.use_var: x.append(torch.sqrt(V[:, t]))
            if self.n_extra: x.append(extra[:, t])
            if self.use_rv: x.append(torch.sqrt(rv))
            prev = torch.sigmoid(self.f(torch.stack(x, 1)).squeeze(1))
            out.append(prev)
        return torch.stack(out, 1)


def running_features(S, K):
    """Running max/min/mean of log(S/K) up to t (causal). Shape (paths, n, 3)."""
    logS = torch.log(S[:, :-1] / K)
    return torch.stack([torch.cummax(logS, 1).values, -torch.cummax(-logS, 1).values,
                        torch.cumsum(logS, 1) / torch.arange(1, logS.shape[1] + 1)], 2)


def torch_pnl(S, delta, payoff, cost, premium, impact=0.0):
    """cost may be a float or a per-path tensor of shape (paths,)."""
    c = cost.unsqueeze(1) if torch.is_tensor(cost) else cost
    gains = (delta * (S[:, 1:] - S[:, :-1])).sum(1)
    z = torch.zeros(S.shape[0], 1)
    trades = torch.abs(torch.cat([delta, z], 1) - torch.cat([z, delta], 1))
    costs = (c * S * trades + impact * S * trades ** 2).sum(1)
    return premium - payoff + gains - costs


def cvar_loss(pnl, w, alpha=0.95):
    """Rockafellar-Uryasev: CVaR_alpha(L) = min_w  w + E[(L-w)+]/(1-alpha), L = -pnl."""
    L = -pnl
    return (w + torch.relu(L - w).mean() / (1 - alpha)).squeeze()


def cvar_topk(pnl, alpha=0.95):
    """Empirical CVaR = mean of the worst (1-alpha) share of losses (used for per-regime CVaR in mixtures)."""
    L = -pnl; k = max(int(round((1 - alpha) * L.numel())), 1)
    return torch.topk(L, k).values.mean()


def _extra(S, K, n_extra):
    return running_features(S, K) if n_extra else None


def train(sim_fn, payoff_fn, K, T, cost, premium, impact=0.0, use_var=False, path_dependent=False,
          steps=1500, batch=10000, lr=2e-3, alpha=0.95, seed=0, verbose=True):
    """Single-regime training. sim_fn(n_paths, seed)->(S,V) numpy; payoff_fn(S)->numpy. Fresh paths every step."""
    torch.manual_seed(seed)
    n_extra = 3 if path_dependent else 0
    net = HedgeNet(use_var, n_extra=n_extra)
    w = nn.Parameter(torch.tensor([premium * 0.5], dtype=torch.float32))
    opt = torch.optim.Adam(list(net.parameters()) + [w], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(steps // 3, 1), gamma=0.3)
    for it in range(steps):
        Sn, Vn = sim_fn(batch, 10_000 + seed * 100_000 + it)
        S = torch.tensor(Sn, dtype=torch.float32); V = torch.tensor(Vn, dtype=torch.float32)
        P = torch.tensor(payoff_fn(Sn), dtype=torch.float32)
        d = net(S, V, K, T, _extra(S, K, n_extra))
        loss = cvar_loss(torch_pnl(S, d, P, cost, premium, impact), w, alpha)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and it % 250 == 0: print(f"  it {it:5d}  CVaR loss {loss.item():.4f}")
    return net


def train_mixture(sampler, make_sim, payoff_fn, K, T, premium, use_rv=False, nominal_sigma=0.2, impact=0.0,
                  steps=3000, batch=12000, lr=2e-3, alpha=0.95, seed=0, verbose=True):
    """Domain-randomised training. Each step: sampler(rng) -> list of Regime objects (same K, T, n_steps);
    equal share of the batch per regime, each regime's own cost. Loss = mean over regimes of
    CVaR95 / (sigma*S0*sqrt(T)) so that high-vol regimes do not dominate (a design choice - state it)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = HedgeNet(use_var=False, use_rv=use_rv, nominal_sigma=nominal_sigma)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(steps // 3, 1), gamma=0.3)
    for it in range(steps):
        regs = sampler(rng); per = batch // len(regs)
        Ss, Vs, cs, ps, scales = [], [], [], [], []
        for j, r in enumerate(regs):
            Sn, Vn = make_sim(r)(per, 10_000 + seed * 100_000 + it * 16 + j)
            Ss.append(Sn); Vs.append(Vn); ps.append(payoff_fn(Sn)); cs.append(np.full(per, r.cost))
            scales.append(r.sigma * 100.0 * np.sqrt(T))
        S = torch.tensor(np.concatenate(Ss), dtype=torch.float32); V = torch.tensor(np.concatenate(Vs), dtype=torch.float32)
        P = torch.tensor(np.concatenate(ps), dtype=torch.float32); C = torch.tensor(np.concatenate(cs), dtype=torch.float32)
        pnl = torch_pnl(S, net(S, V, K, T), P, C, premium, impact)
        loss = sum(cvar_topk(pnl[j * per:(j + 1) * per], alpha) / scales[j] for j in range(len(regs))) / len(regs)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and it % 250 == 0: print(f"  it {it:5d}  normalised CVaR {loss.item():.4f}")
    return net


@torch.no_grad()
def net_positions(net, S, V, K, T):
    S = torch.tensor(S, dtype=torch.float32); V = torch.tensor(V, dtype=torch.float32)
    return net(S, V, K, T, _extra(S, K, net.n_extra)).numpy().astype(np.float64)
