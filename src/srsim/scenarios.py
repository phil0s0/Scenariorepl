"""Scenario generation: turning a predictive distribution into sample paths.

A *scenario* is one joint realisation of everything the replenishment policy
cannot control, over the planning horizon:

    omega = (d_1..d_T ;  L_1..L_T ;  r_1..r_T)

-- a demand path, the lead time an order placed in each week would experience,
and the delay before a returned unit comes back into stock.

Three properties are deliberate and all three matter:

**Common random numbers.**  The scenario set is generated *once* and reused for
every candidate policy the optimizer tries.  Without this the objective surface
is noisy and a gradient-free optimizer chases sampling noise instead of signal.

**A strict optimize/evaluate split.**  Policies are optimized against
``n_opt`` scenarios and reported on a *fresh* ``n_eval``.  Scoring a policy on
the scenarios it was tuned against overstates it -- the optimizer has partly
fitted the noise.

**A third, "true" scenario set.**  ``Omega_opt`` and ``Omega_eval`` are both
drawn from the *fitted forecast*, so the gap between them measures only that
optimism.  Because this is a simulation, we can also draw from the true
data-generating process, and the gap between ``Omega_eval`` and ``Omega_true``
then isolates a completely different error: the forecast being wrong.  Real
deployments cannot do this decomposition; that is the advantage of a tutorial
built on a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import nb_params, sample_lead_times

__all__ = ["ScenarioSet", "make_scenarios_from_qpd", "make_scenarios_from_truth", "make_point_scenarios"]


@dataclass(frozen=True)
class ScenarioSet:
    """``(n_scenarios, horizon)`` realisations of demand, lead time and return delay."""

    demand: np.ndarray
    lead_time: np.ndarray
    return_delay: np.ndarray
    seed: int
    label: str = ""

    @property
    def n_scenarios(self) -> int:
        return self.demand.shape[0]

    @property
    def horizon(self) -> int:
        return self.demand.shape[1]


def _exogenous(
    rng: np.random.Generator, n: int, horizon: int, lead_mean: float, lead_cv: float
):
    lead = sample_lead_times(rng, (n, horizon), mean=lead_mean, cv=lead_cv)
    delay = 1 + rng.poisson(1.0, size=(n, horizon))
    return lead, delay


def make_scenarios_from_qpd(
    forecast,
    row_index: np.ndarray,
    n_scenarios: int,
    seed: int,
    lead_mean: float = 2.0,
    lead_cv: float = 0.4,
    label: str = "",
) -> ScenarioSet:
    """Inverse-transform sampling of demand paths from a fitted J-QPD.

    ``row_index`` selects the ``horizon`` rows of the forecast that make up this
    SKU's planning horizon, in week order.  ``J_QPD_S`` exposes ``ppf`` but no
    ``rvs``, so sampling is ``Q(u)`` with ``u ~ U(0, 1)`` -- which is also why the
    quantile function is the right output contract for a forecast feeding an
    optimizer.

    One wrinkle worth knowing: ``J_QPD_S.ppf`` does not broadcast elementwise --
    see :func:`srsim.forecast.inverse_transform`, which handles it.
    """
    from .forecast import inverse_transform

    rng = np.random.default_rng(seed)
    horizon = len(row_index)
    u = rng.uniform(size=(n_scenarios, horizon))
    draws = inverse_transform(forecast, u, row_index)
    demand = np.clip(np.round(draws), 0.0, None)
    lead, delay = _exogenous(rng, n_scenarios, horizon, lead_mean, lead_cv)
    return ScenarioSet(demand, lead, delay, seed, label)


def make_scenarios_from_truth(
    lam: np.ndarray,
    dispersion: np.ndarray,
    n_scenarios: int,
    seed: int,
    lead_mean: float = 2.0,
    lead_cv: float = 0.4,
    label: str = "true",
) -> ScenarioSet:
    """Demand paths drawn from the true DGP -- the honest yardstick."""
    rng = np.random.default_rng(seed)
    horizon = len(lam)
    n_param, p_param = nb_params(np.asarray(lam), np.asarray(dispersion))
    demand = rng.negative_binomial(
        np.tile(n_param, (n_scenarios, 1)), np.tile(p_param, (n_scenarios, 1))
    ).astype(float)
    lead, delay = _exogenous(rng, n_scenarios, horizon, lead_mean, lead_cv)
    return ScenarioSet(demand, lead, delay, seed, label)


def make_point_scenarios(
    point_forecast: np.ndarray,
    n_scenarios: int,
    seed: int,
    lead_mean: float = 2.0,
    lead_cv: float = 0.4,
    label: str = "point",
) -> ScenarioSet:
    """Every scenario gets the *same* demand path: the point forecast.

    This is the ablation arm Presbitero et al. call P-PCTL.  Uncertainty in lead
    times and returns survives; uncertainty in demand is thrown away.  It is the
    single most informative comparison in the paper, and the cheapest to run.
    """
    rng = np.random.default_rng(seed)
    horizon = len(point_forecast)
    demand = np.tile(np.round(np.asarray(point_forecast, dtype=float)), (n_scenarios, 1))
    lead, delay = _exogenous(rng, n_scenarios, horizon, lead_mean, lead_cv)
    return ScenarioSet(np.clip(demand, 0.0, None), lead, delay, seed, label)
