"""Solving for a policy: SHGO over a scenario-averaged, risk-aware objective.

This is the "Black-Box Optimization Engine" of the Presbitero et al. system.
The objective is

    theta*  =  argmin_theta  Q_75[ C(theta, omega) : omega in Omega_opt ]

-- the 75th percentile of the simulated cost distribution, not its mean.  The
cost distribution is asymmetric, so optimising the mean quietly accepts a fat
right tail; a high percentile is a tractable stand-in for a CVaR criterion.

Two practical points the paper does not dwell on but that matter if you actually
run this:

**Common random numbers make the objective deterministic.**  Because
``Omega_opt`` is fixed across every candidate, ``objective(theta)`` is a pure
function.  Re-drawing scenarios per evaluation would make it stochastic and
gradient-free global optimizers would chase the noise.

**Integer parameters need separate handling.**  ``t0`` and ``t_limit`` are week
indices.  Rounding them inside a continuous objective creates flat plateaus that
defeat SHGO's local refinement step, so we enumerate them on a small outer grid
and let SHGO work on the genuinely continuous ``(Q0, s, Q)``.  This is cleaner
and more honest than rounding, and the outer grid is small enough to be free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import shgo

from .des import CostParams, Policy, ScenarioSet, cost_percentile, simulate_costs
from .newsvendor import critical_fractile

__all__ = [
    "PolicyBounds",
    "OptimizationResult",
    "optimize_policy",
    "tune_order_up_to",
    "myopic_newsvendor_policy",
]


@dataclass(frozen=True)
class PolicyBounds:
    """Search space for ``theta``.

    ``t0 >= lead_time`` reflects the paper's constraint: an initial order cannot
    become sellable sooner than the merchant's replenishment lead time allows.
    """

    q0_max: float
    s_max: float
    q_max: float
    t0_choices: tuple
    t_limit_choices: tuple


def default_bounds(weekly_demand: float, horizon: int, lead_weeks: int = 2) -> PolicyBounds:
    """Bounds scaled to the SKU's own demand rate, so one setting fits all SKUs."""
    return PolicyBounds(
        q0_max=max(6.0 * weekly_demand, 10.0),
        s_max=max(4.0 * weekly_demand, 8.0),
        q_max=max(6.0 * weekly_demand, 10.0),
        t0_choices=(lead_weeks, lead_weeks + 1),
        t_limit_choices=(horizon - 4, horizon - 1),
    )


@dataclass
class OptimizationResult:
    policy: Policy
    objective: float
    n_evaluations: int


def optimize_policy(
    scenarios: ScenarioSet,
    costs: CostParams,
    bounds: PolicyBounds,
    initial_stock: float = 0.0,
    percentile: float | None = 75.0,
    review_period: int = 2,
    order_up_to: bool = False,
    sobol_n: int = 16,
    iters: int = 1,
    local_maxiter: int = 12,
) -> OptimizationResult:
    """Minimise a cost percentile over ``theta``, SHGO inside, grid outside."""
    counter = {"n": 0}

    def objective_factory(t0: int, t_limit: int):
        def objective(x):
            counter["n"] += 1
            policy = Policy(
                t0=t0,
                q0=float(x[0]),
                s=float(x[1]),
                q=float(x[2]),
                t_limit=t_limit,
                review_period=review_period,
                order_up_to=order_up_to,
            )
            sample = simulate_costs(policy, scenarios, costs, initial_stock)
            # percentile=None is the Q-MEAN ablation arm: optimise the average
            # cost instead of a tail percentile.
            return float(sample.mean()) if percentile is None else cost_percentile(sample, percentile)

        return objective

    box = [(0.0, bounds.q0_max), (0.0, bounds.s_max), (0.0, bounds.q_max)]
    # A coarse deterministic sweep first.  It costs almost nothing, guarantees a
    # valid answer, and gives SHGO something to beat -- SHGO occasionally returns
    # `fun=None` when its local refinement finds no acceptable minimiser, and a
    # per-SKU run that raises in the middle of a 200-SKU job is no fun at all.
    grid = np.array(
        np.meshgrid(
            np.linspace(0.0, bounds.q0_max, 4),
            np.linspace(0.0, bounds.s_max, 4),
            np.linspace(0.0, bounds.q_max, 4),
        )
    ).reshape(3, -1).T

    best = None
    for t0 in bounds.t0_choices:
        for t_limit in bounds.t_limit_choices:
            if t_limit < t0:
                continue
            objective = objective_factory(t0, t_limit)
            for point in grid:
                value = objective(point)
                if best is None or value < best[0]:
                    best = (float(value), t0, t_limit, np.asarray(point, dtype=float))

            result = shgo(
                objective,
                bounds=box,
                sampling_method="sobol",
                n=sobol_n,
                iters=iters,
                options={"maxfev": 600, "minimize_every_iter": False},
                minimizer_kwargs={"options": {"maxiter": local_maxiter}},
            )
            if result.fun is not None and result.x is not None and result.fun < best[0]:
                best = (float(result.fun), t0, t_limit, np.asarray(result.x, dtype=float))

    value, t0, t_limit, x = best
    policy = Policy(
        t0=t0,
        q0=float(x[0]),
        s=float(x[1]),
        q=float(x[2]),
        t_limit=t_limit,
        review_period=review_period,
        order_up_to=order_up_to,
    )
    return OptimizationResult(policy, value, counter["n"])


def tune_order_up_to(
    scenarios: ScenarioSet,
    costs: CostParams,
    initial_stock: float,
    s_grid: np.ndarray,
    order_up_to_grid: np.ndarray,
    percentile: float | None = 75.0,
    review_period: int = 2,
    horizon: int = 12,
    always_order: bool = False,
) -> Policy:
    """Grid-tune a classical order-up-to policy on the same scenarios.

    ``always_order=True`` gives a **periodic base-stock** policy (order up to
    ``S`` at every review regardless of position); otherwise it is a classical
    ``(s, S)``.  Both are tuned on exactly the same ``Omega_opt`` and the same
    objective as the extended policy, so the comparison isolates the policy class
    rather than the tuning effort -- the methodological parity the paper insists on.
    """
    best, best_value = None, np.inf
    for s_level in [np.inf] if always_order else s_grid:
        for target in order_up_to_grid:
            policy = Policy(
                t0=0,
                q0=0.0,
                s=float(s_level),
                q=float(target),
                t_limit=horizon - 1,
                review_period=review_period,
                order_up_to=True,
                require_no_in_transit=False,
            )
            sample = simulate_costs(policy, scenarios, costs, initial_stock)
            value = float(sample.mean()) if percentile is None else cost_percentile(sample, percentile)
            if np.isfinite(value) and value < best_value:
                best, best_value = policy, value
    if best is None:  # every candidate was non-finite -- should not happen
        raise RuntimeError("no feasible order-up-to policy found")
    return best


def myopic_newsvendor_policy(
    forecast_quantile_fn,
    costs: CostParams,
    lead_weeks: float,
    review_period: int = 2,
    horizon: int = 12,
) -> Policy:
    """The myopic newsvendor baseline -- tutorial 1's answer, used as a policy.

    At each review it targets the critical fractile of demand over the period it
    is responsible for (the review period plus the lead time), ignoring every
    multi-period effect.  Underage is the margin lost on an unserved unit;
    overage is the cost of holding a unit that did not sell.

    This is the literal bridge between the two tutorials: the closed-form answer
    from part A of tutorial 1, dropped into tutorial 2's simulator so it can be
    scored by the same cost functional as everything else.
    """
    tau = float(
        critical_fractile(costs.lost_sale_cost, costs.holding_fee * (review_period + lead_weeks))
    )
    target = float(forecast_quantile_fn(tau))
    return Policy(
        t0=0,
        q0=0.0,
        s=np.inf,
        q=target,
        t_limit=horizon - 1,
        review_period=review_period,
        order_up_to=True,
        require_no_in_transit=False,
    )
