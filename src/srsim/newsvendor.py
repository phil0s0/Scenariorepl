"""The single-period newsvendor, and the identity that makes it the right
starting point for a tutorial about decision-focused learning.

With underage cost ``c_u`` (margin lost on demand you could not serve) and
overage cost ``c_o`` (cost of a unit you stocked and did not sell), the cost of
ordering ``q`` against realised demand ``D`` is

    C(q, D) = c_u (D - q)^+ + c_o (q - D)^+

and, writing ``tau = c_u / (c_u + c_o)``,

    C(q, D) = (c_u + c_o) * pinball_tau(D, q)

*exactly* -- not approximately.  Minimising empirical decision cost over a class
of order policies is therefore **identical** to pinball quantile regression at
level ``tau`` over that same class.  Decision-focused learning for the
newsvendor is not a new algorithm; it is quantile regression, and you already
know how to do it.

That identity is the cleanest possible introduction to the idea, and mapping out
exactly where it stops holding is what motivates everything that follows.  See
:func:`equivalence_caveats`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .simulate import nb_params

__all__ = [
    "critical_fractile",
    "newsvendor_cost",
    "pinball_loss",
    "expected_cost_nb",
    "optimal_order_nb",
    "regret_nb",
    "equivalence_caveats",
]


def critical_fractile(underage, overage):
    """``tau = c_u / (c_u + c_o)`` -- the service level the cost ratio implies."""
    underage = np.asarray(underage, dtype=float)
    overage = np.asarray(overage, dtype=float)
    return underage / (underage + overage)


def newsvendor_cost(q, demand, underage, overage):
    """Realised cost of ordering ``q`` against realised ``demand``."""
    q = np.asarray(q, dtype=float)
    demand = np.asarray(demand, dtype=float)
    return np.asarray(underage) * np.maximum(demand - q, 0.0) + np.asarray(
        overage
    ) * np.maximum(q - demand, 0.0)


def pinball_loss(y, q, tau):
    """``tau (y - q)^+ + (1 - tau) (q - y)^+``.

    Scaled by ``c_u + c_o`` this *is* :func:`newsvendor_cost`; the notebook
    checks that numerically rather than asking the reader to take it on trust.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return tau * np.maximum(y - q, 0.0) + (1.0 - tau) * np.maximum(q - y, 0.0)


def _expected_shortfall_below(q: np.ndarray, lam: np.ndarray, dispersion: np.ndarray):
    """``E[(q - D)^+]`` under the true NB, via ``sum_{k<q} F(k)``.

    Uses the identity ``E[(q - D)^+] = sum_{k=0}^{q-1} F(k)``, which is exact for
    integer ``q`` and avoids any simulation error in the benchmark.
    """
    q = np.asarray(q, dtype=float)
    n, p = nb_params(lam, dispersion)
    q_max = int(np.max(q)) if q.size else 0
    if q_max <= 0:
        return np.zeros_like(q, dtype=float)
    ks = np.arange(q_max, dtype=float)  # 0 .. q_max-1
    # cdf_matrix[i, k] = F_i(k)
    cdf_matrix = stats.nbinom.cdf(ks[None, :], np.asarray(n)[:, None], np.asarray(p)[:, None])
    cumulative = np.cumsum(cdf_matrix, axis=1)
    # sum_{k=0}^{q-1} F(k) == cumulative[:, q-1], with 0 for q == 0
    idx = np.clip(q.astype(int) - 1, -1, q_max - 1)
    out = np.where(idx >= 0, cumulative[np.arange(len(q)), np.maximum(idx, 0)], 0.0)
    return out


def expected_cost_nb(q, lam, dispersion, underage, overage):
    """Exact ``E[C(q, D)]`` under the true negative binomial.

    Exact rather than Monte Carlo: the benchmark this tutorial reports should not
    have simulation noise of its own.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    dispersion = np.atleast_1d(np.asarray(dispersion, dtype=float))
    below = _expected_shortfall_below(q, lam, dispersion)  # E[(q - D)^+]
    above = lam - q + below  # E[(D - q)^+]
    return np.asarray(underage) * above + np.asarray(overage) * below


def optimal_order_nb(lam, dispersion, tau):
    """``q* = min{q : F(q) >= tau}`` under the true NB.

    Demand is a count, so the optimum is a *discrete* quantile.  ``nbinom.ppf``
    already returns exactly this, which is why the continuous ``J_QPD.ppf`` must
    be ceilinged to match it.
    """
    n, p = nb_params(lam, dispersion)
    return np.asarray(stats.nbinom.ppf(np.asarray(tau, dtype=float), n, p), dtype=float)


def regret_nb(q, lam, dispersion, underage, overage):
    """Expected cost of ordering ``q``, minus the best achievable expected cost.

    Zero exactly when ``q`` is the true critical-fractile order.  This is regret
    against a *model oracle* -- the best decision obtainable from the true
    conditional distribution -- not against a clairvoyant who knows the realised
    demand.
    """
    tau = critical_fractile(underage, overage)
    q_star = optimal_order_nb(lam, dispersion, tau)
    return expected_cost_nb(q, lam, dispersion, underage, overage) - expected_cost_nb(
        q_star, lam, dispersion, underage, overage
    )


def equivalence_caveats() -> list[str]:
    """Where "decision-focused training == quantile regression" stops being true.

    These six are the spine of the whole tutorial series: each one is a reason
    the simple answer fails, and between them they motivate the constrained
    problem in part B and the whole of tutorial 2.
    """
    return [
        "Cost must be exactly two-slope piecewise-linear in q. A fixed ordering "
        "cost, salvage value, or quantity discount breaks the pinball identity.",
        "Single period, no carryover. Once leftover stock survives into next "
        "week -- the entire premise of an (R,s,Q) policy -- per-period costs stop "
        "being separable. With lead time L the result generalises to the "
        "distribution of lead-time demand, not single-period demand.",
        "q must be unconstrained. Add a shared capacity across items and the "
        "per-item optimum is no longer its own fractile: it becomes water-filling "
        "against a common shadow price, q_j* = F_j^-1((c_u - mu)/(c_u + c_o)).",
        "Under misspecification pinball still returns the decision-optimal member "
        "of the model class, while fitting by MSE and plugging in does not. This "
        "is the real argument for decision-focused learning -- not that it is "
        "always better, but that it optimises the thing you care about.",
        "It delivers one fractile, not a distribution. Anything that needs whole "
        "sampled demand paths -- a multi-week simulation -- needs more than this.",
        "Demand is integer-valued, so the optimum is min{q : F(q) >= tau}. A "
        "continuous quantile must be ceilinged, not rounded.",
    ]
