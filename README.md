# When does deep hedging beat analytic baselines?

**Research question:** across cost level, vol-of-vol, rebalancing frequency and maturity, where does a neural-net
hedger beat (a) BS delta, (b) Leland, (c) Whalley-Wilmott, measured by CVaR95 of hedging loss, with
paired-bootstrap confidence intervals? Negative results count.

## Run order
1. `pip install -r requirements.txt`
2. `python run_sanity.py`  -> net must recover ~BS delta with zero costs (pipeline bug check)
3. `python run_sweeps.py --quick`  -> small sweeps (minutes). Drop `--quick` for the real run.
   `--no-torch` runs baselines only.
4. Read `results/*.json`: `vs_best` gives NET WINS / NET LOSES / tie per regime.

## Fairness rules (these make the result credible)
- Baselines are TUNED on validation paths (WW risk-aversion, vol-inflation multiplier); textbook Leland is also reported.
- If `grid_edge_warning` is true, the baseline's tuning hit its grid boundary: widen the grid before trusting a "net wins".
- Same cost model, same test paths, same risk measure for every strategy. Training and test paths use different seeds.
- Net is trained with several seeds in the real run: report the spread, not the best seed.
- Two Heston nets: `net` (sees only price/time/position) and `net_var` (also sees variance). `net_var` uses
  information the baselines lack and variance is NOT observable in real markets, so don't call a win of
  `net_var` "beating Black-Scholes".

## Known limits (state these in class)
- Simulation only (GBM/Heston, zero rates, proportional costs). Not a statement about real markets yet.
- Baselines assume known sigma; under Heston that is deliberately misspecified (sigma=sqrt(v0)).
- Whalley-Wilmott/Leland formulas here were written from memory: verify them against the original papers.

## v2 additions
- `dh/payoffs.py`: european, asian (arithmetic), up-and-out barrier (discrete monitoring), floating-strike lookback.
- `simulate_merton` (jump-diffusion, martingale-corrected, total vol matched to `sigma`).
- Cost model: proportional + quadratic impact (`Regime.impact`). Fixed costs and lot sizes are NOT implemented (non-differentiable).
- Path-dependent payoffs: the net gets running max/min/mean of log-moneyness (`path_dependent=True`); without it the net cannot know its own payoff.
- `run_grid.py` (2D grids) and `plot_grids.py` (heatmaps: CVaR gap net vs best baseline, with win/loss/tie labels).

## Critical caveat on exotics
For asian/barrier/lookback the baselines hedge with the VANILLA call delta (a proxy), not the exotic's true delta.
They are weak by construction (lookback proxy CVaR is 3-4x the vanilla's). A net beating them proves little.
To make the exotic result meaningful, add a proper reference delta (e.g. nested Monte-Carlo / bump-and-reprice with
common random numbers) before claiming "the net wins on exotics". Results tagged `proxy_baselines: true`.
Also still missing: Heston-calibrated delta baseline for the Heston/Merton rows.

## v3: two studies, one pipeline
**Study A (regime map, `run_sweeps.py`, `run_grid.py`, `plot_grids.py`):** nets trained INSIDE each regime. Question: where does the net beat tuned baselines?
**Study B (robustness, `run_robustness.py`, `plot_robustness.py`):** everything calibrated/trained at a NOMINAL market
(Heston, sigma 0.20, xi 0.3, cost 10 bps, 30 steps, 30 days, European call), then evaluated FROZEN under shifted markets:
true vol, vol-of-vol, jump intensity, cost, and a combined 'crisis' cell. Question: how fast does each strategy degrade and can it degrade slower?

Strategies in Study B
- Frozen baselines: delta, Leland, tuned vol-inflated delta, Whalley-Wilmott, all assuming NOMINAL sigma and cost.
- `net_nominal`: trained at nominal only (the specialist). `net_nominal_shrunk`: 50/50 blend with plain delta.
- `net_random`: domain-randomised training (random vol 0.12-0.4, xi 0.2-1.8, jumps, cost 5-100 bps; each regime's CVaR
  normalised by sigma*S0*sqrt(T) so high-vol regimes do not dominate - a DESIGN CHOICE that can change results).
- `net_random_rv`: same plus a causal realised-volatility input (observable in real markets, unlike Heston variance).
- Reference only (not a competitor): delta using the TRUE sigma. `--oracle` also trains a specialist per stress cell.

Design rules
- Each test cell is priced at its own fair premium, so CVaR measures hedging error, not mispricing profit/loss
  (using the nominal premium made CVaR negative at low vol and ratios meaningless: found and fixed).
- Cells outside the randomised training range are flagged `in_support: false` (open markers in the plot). A robust net
  that only wins inside its training range has shown interpolation, not robustness.
- Verdicts use paired bootstrap CIs vs the best FROZEN baseline in that cell.

Baseline-only findings so far (hypotheses to test with the net, NOT results about the net)
- Misspecified VOLATILITY is the big lever: with true sigma known, delta would cut tail risk 11-47% vs the best frozen baseline at sigma 0.3-0.5.
- Misspecified vol-of-vol, jumps and cost move the frozen baselines relative to each other by only ~5-20%.
- So the most promising way to 'degrade slower' is an observable volatility signal (`net_random_rv`).
Still untested: every net result (PyTorch has not been run by the author).
