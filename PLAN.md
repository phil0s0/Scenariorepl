# Scenario-Based Stochastic Replenishment Optimization — Build Plan

This plan synthesizes two papers into one deployable system: Zalando's ZEOS
replenishment architecture as the backbone, with Amazon's exogenous-demand
paper informing data correction and a later RL upgrade path. The core
methodology is made explicit up front: the system is a **scenario-based
stochastic program solved by sample average approximation (SAA) over a
parametric policy class**. Everything else — forecasting, distribution
fitting, simulation — exists to generate and evaluate scenarios.

---

## 1. Mathematical framing: the scenario-based stochastic program

### 1.1 Decision vector

Per SKU, the policy is an extended (R, s, Q) rule with parameters

```
θ = (t₀, Q₀, s, Q, t_limit)
```

- `t₀`, `Q₀` — timing and size of the initial order (the only knobs exposed to users)
- `s` — reorder point, `Q` — reorder quantity for subsequent replenishment
- `t_limit` — last week in which reordering is allowed
- `R` (review period) is fixed operationally, not optimized

### 1.2 Scenarios

A **scenario** ω is one joint realization of all exogenous uncertainty over
the planning horizon T = 12 weeks (per SKU):

```
ω = ( d₁, …, d_T ;  L₁, L₂, … ;  r₁, r₂, … )
```

- **Demand path** `d_t = Q_t(u_t)` with `u_t ~ U(0,1)` — inverse-transform
  sampling from the per-SKU-week J-QPD quantile function (§4)
- **Lead times** `L_i` — nonparametric bootstrap from historical lead-time
  observations
- **Return delays** `r_j` (and return quantities via the return rate) —
  nonparametric bootstrap from historical return observations

**The load-bearing assumption is exogeneity** (the Amazon paper's central
argument): demand, prices, lead times, and return behavior do not respond to
our replenishment decisions. This is what licenses generating scenarios
*independently of the policy* — the same scenario set is valid for every
candidate θ, and (later, §9) historical trajectories are themselves valid
scenarios. If exogeneity fails (e.g., stockouts permanently shift demand to
competitors in a way the censoring correction doesn't capture), the whole
scenario framework needs revisiting.

### 1.3 Cost functional

The discrete-event simulator (§6) is a **deterministic** map

```
C(θ, ω) → total cost over the horizon
```

All randomness lives in the scenario ω; none lives in the simulator. Cost
components: holding, inbound handling, outbound handling, returns processing,
and lost sales, where

```
lost_sales_cost = (price − cost) × (1 − return_rate) × unmet_demand
```

### 1.4 The SAA problem

Fix a finite scenario set `Ω_N = {ω₁, …, ω_N}` (N ≈ 500 for optimization).
The optimization problem is

```
min over θ   of   ρ( C(θ, ω₁), …, C(θ, ω_N) )
```

where **ρ is the empirical 75th percentile** of the cost sample — the Q-PCTL
objective from Zalando's ablation, which beats the mean objective (Q-MEAN).
This is a CVaR-like robust criterion: it optimizes against the unfavorable
tail of the cost distribution rather than its average, which matters because
the cost distribution is asymmetric (understocking a winner is far more
costly than overstocking it).

Because `Ω_N` is **fixed across all candidate policies** (common random
numbers), the SAA objective is a deterministic function of θ. This is
essential: it removes Monte Carlo noise from the objective surface, so
gradient-free optimizers (SHGO, differential evolution) see a stable
landscape instead of a noisy one.

### 1.5 What this is — and what it is not

- It **is** simulation optimization over a *parametric policy class*: we
  search a 5-dimensional θ, and the policy reacts to realized state through
  the (s, Q) rule *inside* each scenario rollout.
- It is **not** two-stage stochastic programming with per-scenario recourse
  variables: there are no scenario-indexed decisions, so the problem stays
  5-dimensional regardless of N, and per-SKU independence keeps it
  embarrassingly parallel.
- The RL upgrade (§9) is the step from static parameters to *sequential
  policies* learned over the same scenario machinery — both papers flag this
  as the frontier.

---

## 2. Phase 0 — Data foundation & censoring correction

Build the SKU-week fact table: sales, stock on hand, in-transit, prices,
costs, returns, lead times.

**Correct demand censoring before anything else** (the Amazon paper's key
insight): observed sales = min(demand, inventory). Use
availability-corrected demand via traffic/view signals where available;
otherwise impute demand for out-of-stock weeks. Every scenario generated
downstream inherits the quality of this signal — a censored demand model
produces systematically pessimistic scenarios, which biases θ toward
under-ordering.

---

## 3. Phase 1 — Probabilistic demand forecast

Quantile regression with LightGBM: pinball loss per quantile, or a single
model with the quantile level as a feature. Features: demand lags, rolling
statistics, price/discount, seasonality, product hierarchy.

Then **conformalize** with CQR (conformalized quantile regression) on a
*temporal* calibration split to guarantee marginal coverage of the quantile
estimates.

Output: a small set of calibrated quantiles per SKU-week (e.g.
5 / 25 / 50 / 75 / 95).

---

## 4. Phase 2 — J-QPD distribution layer (the scenario sampling interface)

This is the improvement over Zalando's 39-raw-quantile representation, and
its **output contract is exactly what the scenario generator (§5) needs**: a
smooth, strictly increasing quantile function `Q_t(u)` per SKU-week that can
be inverted cheaply.

Fit a **Johnson Quantile-Parameterized Distribution** to 3–4 conformalized
quantiles per SKU-week. J-QPD-B with lower bound 0 fits demand naturally
(bounded support, no negative demand). Sampling is then inverse-transform:
`d = Q(u), u ~ U(0,1)`.

Validate fit quality against the raw quantiles; fall back to monotone
interpolation of the raw quantiles where J-QPD fitting fails.

---

## 5. Phase 3a — Scenario generation engine

The explicit new module. Responsibilities:

- **Scenario schema** (per SKU): demand path `d₁…d_T`, lead-time draws,
  return-delay draws, and the seed that produced them.
- **Fixed-seed generation**: scenarios are generated once, from recorded
  seeds, and are fully reproducible.
- **Common random numbers**: the *same* scenario set is reused across every
  policy candidate the optimizer evaluates (§1.4).
- **In-sample / out-of-sample discipline**:
  - `Ω_opt` — N ≈ 500 scenarios used *only* for optimization
  - `Ω_eval` — a *fresh* N ≥ 2000 scenarios used *only* for final policy
    evaluation and reporting

  Evaluating a policy on the scenarios it was optimized against overstates
  its performance (**SAA optimism bias** — the optimizer has partially fit
  the noise of `Ω_opt`). All reported numbers come from `Ω_eval`.

---

## 6. Phase 3b — Discrete-event cost evaluator

A weekly discrete-event simulation over the 12-week horizon (Zalando's
sensitivity analysis supports this horizon), implemented as the deterministic
functional `C(θ, ω)` of §1.3:

1. Consume the scenario's demand, lead-time, and return-delay realizations —
   no sampling inside the simulator.
2. Evolve stock: arrivals (orders placed per policy θ, delayed by sampled
   lead times), demand consumption, returns re-entering stock after sampled
   delays.
3. Accumulate cost components: holding, inbound, outbound, returns
   processing, lost sales (§1.3).

Determinism given (θ, ω) is a hard requirement and a unit-testable property.

---

## 7. Phase 4 — SAA optimization

Solve the §1.4 problem per SKU:

- **Decision space**: θ = (t₀, Q₀, s, Q, t_limit), with R fixed.
- **Optimizer**: gradient-free global — SHGO, or differential evolution /
  Optuna as pragmatic alternatives.
- **Objective**: empirical 75th percentile of `{C(θ, ωᵢ)}` over `Ω_opt`
  (Q-PCTL; the ablation shows it beats Q-MEAN).
- **Parallelism**: per-SKU independence ⇒ embarrassingly parallel.
- **Substrate**: batched JAX evaluation on GPU, sharded across SKUs via
  Kubernetes Indexed Jobs (§12).

**SAA diagnostics** (explicit additions):

- **Scenario-count sensitivity**: re-solve at N ∈ {250, 500, 1000}; θ* and
  the objective should stabilize as N grows.
- **Optimality-gap estimate**: solve on M independent scenario batches
  (e.g. M = 5 batches of 500); the spread of the M objective values bounds
  the SAA error. If the spread is large relative to the cost differences
  between candidate policies, increase N.
- Final candidate policies are re-scored on `Ω_eval` before any comparison
  or reporting.

---

## 8. Phase 5 — Backtesting & statistical validation

The walk-forward backtest is the **ultimate out-of-sample test**: real
historical trajectories are nature's scenarios, drawn from the true joint
distribution rather than the model's.

- **Design**: walk-forward over ≥ 12 monthly execution dates against
  baselines — current policy / human decisions, tuned (s, S), base-stock,
  myopic newsvendor.
- **Metrics**: GMV, GMV after fulfillment costs, availability, fill rate.
- **Inference**: merchant/SKU-level bootstrap (~2000 resamples) for CIs on
  uplifts, Wilson intervals for proportions, paired permutation tests.
- **Ablations that matter**: (1) point vs. probabilistic forecast,
  (2) mean vs. percentile objective.

---

## 9. Phase 6 — Deployment

- Expose only `(t₀, Q₀)` to users; keep `(s, Q, t_limit)` internal.
- Daily batch recommendations.
- Anti-leakage guardrails on feature freeze dates.
- **Forecast-WAPE monitoring** as the leading health indicator: Zalando
  found ρ ≈ −0.85 between forecast WAPE and profit uplift, so forecast
  degradation predicts policy degradation before the P&L shows it.

---

## 10. Phase 7 (optional) — RL upgrade path

Once the scenario engine (§5) and evaluator (§6) exist, the expensive part of
the Amazon approach is already built. Their exogeneity argument (§1.2) means
historical trajectories *are* valid simulation paths. A differentiable
simulator plus DirectBackprop-style policy learning then replaces the
one-shot SHGO optimization: **sequential policies instead of static
parameters**, trained over the identical scenario machinery — the scenario
engine is reused as-is. Both papers flag this as the frontier.

Two concrete routes, both over the same JAX simulator (§12.1):

- **DirectBackprop** — soft-relax the discrete branches of the evaluator and
  train a policy network by backpropagating through the rollout.
- **MCTS via mctx** — treat the exact (non-relaxed) simulator as a perfect
  model and search over order decisions at each review epoch (§12.4).

Run both against the SAA baseline inside the Phase 5 harness before
committing to either.

---

## 11. Dependency chain

```
censoring correction
      → calibrated quantiles (LightGBM + CQR)
      → J-QPD quantile functions
      → scenario generator (Ω_opt / Ω_eval, CRN, seeds)
      → deterministic evaluator C(θ, ω)
      → SAA optimizer (75th-percentile objective + diagnostics)
      → walk-forward backtest
      → deployment
      → (optional) RL over the same scenario engine
```

Each stage is independently testable: coverage tests for the conformal
quantiles, fit-quality tests for J-QPD, reproducibility/determinism tests for
the scenario generator and evaluator, SAA-gap diagnostics for the optimizer,
and statistical tests for the backtest.

---

## 12. Compute & execution architecture

Four components, each with one specific job:

| Component | Job |
|---|---|
| **JAX** | Implementation substrate for scenarios + evaluator: vectorized, jitted, differentiable |
| **GPU** | Batch axis exploitation: SKUs × candidates × scenarios in one kernel launch |
| **Kubernetes Indexed Jobs** | Orchestration of the embarrassingly parallel per-SKU sharding |
| **mctx** | Phase 7 sequential-policy search over the exact JAX simulator |

### 12.1 JAX evaluator core

Implement `C(θ, ω)` (§6) as a pure function:

- **Weekly DES as `lax.scan`** over T = 12 steps. State carried through the
  scan: on-hand stock, in-transit pipeline (a fixed-length vector indexed by
  weeks-to-arrival), pending-returns vector. Policy logic — the (s, Q)
  reorder rule, `t₀`/`Q₀` initial order, `t_limit` cutoff — expressed
  branch-free with `jnp.where`, so the whole rollout jit-compiles.
- **Scenario generation on-device.** The J-QPD inverse CDF (§4) is
  closed-form (a Johnson transform of normal quantiles), so
  `d_t = Q_t(u_t)` with `u_t` from `jax.random.uniform` is a few
  element-wise ops. Bootstrap lead-time/return-delay draws are
  `jax.random.choice` over historical pools. **Common random numbers become
  key discipline**: `fold_in(key, sku_id)` then `fold_in(·, scenario_id)` —
  every candidate θ sees identical scenarios by construction, and scenarios
  never need to be materialized to disk unless wanted for audit.
- **Three nested `vmap`s**: scenarios (N) × candidate policies (P) × SKUs
  (B). One jitted call produces a B × P × N cost tensor; the SAA objective
  is `jnp.percentile(costs, 75, axis=scenario_axis)`.
- **Determinism stays testable** (§6's hard requirement): fixed keys, no
  nondeterministic ops; assert bit-identical costs across repeated calls.
- **Differentiability for free**: the same code with soft relaxations of the
  `where`-branches (sigmoid instead of step for the reorder trigger) is the
  differentiable simulator Phase 7's DirectBackprop needs.

### 12.2 GPU batching

- The optimizer's entire population is scored in **one batched evaluation**:
  P candidates × N = 500 scenarios × B SKUs of 12-step scans per launch.
  Percentile reduction happens on-device; only the P objective values (or
  just the argmin) cross back to host.
- **Keep the optimization loop on-device where possible**: a JAX-native
  population method (e.g. evosax's differential evolution / CMA-ES) keeps
  the full optimize step inside jit. SHGO remains a host-side SciPy loop —
  acceptable, since each of its objective calls is still a single batched
  GPU evaluation, but the population-based route avoids per-iteration
  host↔device round-trips.
- **Sizing**: float32 throughout; shard size B is tuned to GPU memory
  against the B × P × N working set. The final `Ω_eval` re-scoring
  (N ≥ 2000, §5) is one extra batched pass over the surviving θ*.

### 12.3 Kubernetes Indexed Jobs

Per-SKU independence (§1.5) maps directly onto indexed completion:

- **Sharding**: a prep step writes a deterministic manifest
  (index → SKU list) plus per-shard inputs (J-QPD parameters, bootstrap
  pools, cost parameters) to object storage as parquet. One `Job` with
  `completionMode: Indexed`, `completions: K`,
  `parallelism: min(K, GPU quota)`. Each pod reads
  `JOB_COMPLETION_INDEX`, loads its shard, runs the §12.2 optimization,
  writes `θ*` + SAA diagnostics to `results/shard={index}/`.
- **Idempotence ⇒ cheap retries**: fixed seeds make every shard rerunnable
  with identical output, so `backoffLimitPerIndex` and spot/preemptible GPU
  nodes are safe. Failed indexes retry without touching completed ones.
- **Scheduling**: `nvidia.com/gpu: 1` per pod with the GPU pool's
  nodeSelector/tolerations; a CPU-only shard class for long-tail SKUs whose
  batch sizes don't justify GPU queue time.
- **Reuse of the pattern**: the walk-forward backtest (§8) runs as a second
  indexed Job with index = execution date × policy arm; SAA optimality-gap
  batches (§7) as index = batch id. The daily production run (§9) is a
  CronJob triggering the phase DAG (Argo Workflows or chained Jobs).

### 12.4 mctx for the Phase 7 sequential policy

mctx is JAX-native MCTS, so it composes directly with the jitted simulator —
no environment bridging, and search batches across SKUs/root states via
`vmap` on the same GPUs.

- **MDP framing** (per SKU): state = (week, on-hand, in-transit vector,
  pending returns); action = order quantity from a discretized grid
  (including 0 = no order); transitions = scenario draws from the §5 engine.
- **Perfect model, not learned**: in MuZero terms the "dynamics network" is
  replaced by the exact simulator step — no model learning stage. Use
  `mctx.stochastic_muzero_policy` with chance nodes over discretized demand
  outcomes (binned from the J-QPD quantiles), or root-sampled scenarios with
  `mctx.gumbel_muzero_policy` when the search budget per decision is small.
- **Two deployment modes**: (a) *receding-horizon planner* — run the search
  at each review epoch and execute the visit-count argmax; this is the
  strong sequential benchmark against static θ*; (b) *policy improvement
  operator* — distill visit distributions into a policy network
  (AlphaZero-style), amortizing search into an inference-time policy as
  cheap as the static rule.
- **Everything upstream carries over unchanged**: scenario engine, CRN key
  discipline, and the Ω_opt/Ω_eval split (§5) are identical; evaluation
  still happens in the Phase 5 harness against the SAA baseline.
