"""Running the Presbitero-style ablation end to end, per SKU.

Each SKU is solved independently -- the paper's own assumption of no
cannibalisation between SKUs -- which makes the whole thing embarrassingly
parallel.  We exploit that with ``joblib`` here for the same reason the paper
reaches for Kubernetes: it is the cheapest speedup available.

The arms follow the paper's ablation table, plus its classical baselines:

``Q-PCTL``   probabilistic forecast, 75th-percentile cost objective (the full model)
``Q-MEAN``   probabilistic forecast, mean cost objective
``P-PCTL``   point forecast, 75th-percentile objective
``(s,S)``    grid-tuned classical order-up-to with a reorder point
``base-stock``  grid-tuned periodic order-up-to
``newsvendor``  the myopic single-period answer from tutorial 1

Every arm is tuned on the *same* ``Omega_opt`` with the *same* cost parameters,
so the comparison isolates the method rather than the tuning effort.

Every arm is then scored on two held-out scenario sets, and the difference
between them is the point of the exercise:

``Omega_eval``  fresh scenarios from the *fitted forecast*  -> isolates SAA optimism
``Omega_true``  scenarios from the *true DGP*               -> isolates forecast error
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .des import CostParams, Policy, cost_percentile, service_metrics, simulate_costs
from .optimize import (
    default_bounds,
    myopic_newsvendor_policy,
    optimize_policy,
    tune_order_up_to,
)
from .scenarios import (
    make_point_scenarios,
    make_scenarios_from_qpd,
    make_scenarios_from_truth,
)

__all__ = ["sku_rows", "run_sku", "run_ablation", "ARMS"]

ARMS = ["Q-PCTL", "Q-MEAN", "P-PCTL", "(s,S)", "base-stock", "newsvendor"]


def sku_rows(test: pd.DataFrame, p_id: int, l_id: int, horizon: int) -> np.ndarray:
    """Positional indices of one SKU's first ``horizon`` evaluation weeks."""
    mask = (test["P_ID"].to_numpy() == p_id) & (test["L_ID"].to_numpy() == l_id)
    idx = np.where(mask)[0]
    order = np.argsort(test["week"].to_numpy()[idx])
    return idx[order][:horizon]


def run_sku(
    world,
    p_id: int,
    l_id: int,
    horizon: int = 12,
    n_opt: int = 500,
    n_eval: int = 2000,
    costs: CostParams | None = None,
    review_period: int = 2,
    lead_mean: float = 1.0,
    seed: int = 0,
    arms: tuple | None = None,
) -> list[dict]:
    """Optimize and score the requested arms for a single SKU.

    ``arms`` defaults to all of :data:`ARMS`; pass a subset when you only need
    one (sweeps over forecast quality, for instance) and do not want to pay for
    an SHGO run per unused arm.
    """
    wanted = set(ARMS if arms is None else arms)
    costs = costs or CostParams()
    rows = sku_rows(world.test, p_id, l_id, horizon)
    if len(rows) < horizon:
        return []

    forecast = world.forecast
    median = forecast.q_median[rows]
    lam = world.test["lam"].to_numpy()[rows]
    dispersion = world.test["dispersion"].to_numpy()[rows]

    # "Current stock" at the start of the horizon, so the run is not dominated by
    # an unavoidable stock-out while the very first order is still in transit.
    initial_stock = float(1.5 * np.median(median))
    weekly = float(np.mean(median))
    lead_weeks = max(1, int(round(lead_mean)))
    bounds = default_bounds(weekly, horizon, lead_weeks)

    base = 1000 * (p_id + 1) + 7 * (l_id + 1) + seed
    omega_opt = make_scenarios_from_qpd(forecast, rows, n_opt, base + 1, lead_mean=lead_mean, label="opt")
    omega_eval = make_scenarios_from_qpd(forecast, rows, n_eval, base + 2, lead_mean=lead_mean, label="eval")
    omega_true = make_scenarios_from_truth(lam, dispersion, n_eval, base + 3, lead_mean=lead_mean, label="true")
    omega_point = make_point_scenarios(median, n_opt, base + 4, lead_mean=lead_mean)

    kw = dict(initial_stock=initial_stock, review_period=review_period)
    policies: dict[str, Policy] = {}

    if "Q-PCTL" in wanted:
        policies["Q-PCTL"] = optimize_policy(omega_opt, costs, bounds, percentile=75.0, **kw).policy
    if "Q-MEAN" in wanted:
        policies["Q-MEAN"] = optimize_policy(omega_opt, costs, bounds, percentile=None, **kw).policy
    if "P-PCTL" in wanted:
        policies["P-PCTL"] = optimize_policy(omega_point, costs, bounds, percentile=75.0, **kw).policy

    s_grid = np.linspace(0.0, 4.0 * weekly, 9)
    up_to_grid = np.linspace(weekly, 9.0 * weekly, 12)
    if "(s,S)" in wanted:
        policies["(s,S)"] = tune_order_up_to(
            omega_opt, costs, initial_stock, s_grid, up_to_grid,
            review_period=review_period, horizon=horizon,
        )
    if "base-stock" in wanted:
        policies["base-stock"] = tune_order_up_to(
            omega_opt, costs, initial_stock, s_grid, up_to_grid,
            review_period=review_period, horizon=horizon, always_order=True,
        )
    if "newsvendor" in wanted:
        policies["newsvendor"] = myopic_newsvendor_policy(
            lambda tau: np.mean(forecast.qpd.ppf(tau)[rows]) * (review_period + lead_mean),
            costs, lead_mean, review_period, horizon,
        )

    out = []
    for arm, policy in policies.items():
        row = {"P_ID": p_id, "L_ID": l_id, "arm": arm}
        in_sample = simulate_costs(policy, omega_opt, costs, initial_stock)
        row["cost_p75_opt"] = cost_percentile(in_sample, 75.0)
        for label, omega in (("eval", omega_eval), ("true", omega_true)):
            metrics = service_metrics(policy, omega, costs, initial_stock)
            for key, value in metrics.items():
                row[f"{key}_{label}"] = value
        row["theta"] = policy
        out.append(row)
    return out


def run_ablation(world, skus, n_jobs: int = -1, **kwargs) -> pd.DataFrame:
    """Run :func:`run_sku` across SKUs in parallel and stack the results."""
    batches = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_sku)(world, int(p), int(l), **kwargs) for p, l in skus
    )
    return pd.DataFrame([row for batch in batches for row in batch])
