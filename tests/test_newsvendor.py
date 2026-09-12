"""The newsvendor closed forms must agree with brute force, exactly."""

import numpy as np
import pytest
from scipy import stats

from srsim.newsvendor import (
    critical_fractile,
    equivalence_caveats,
    expected_cost_nb,
    newsvendor_cost,
    optimal_order_nb,
    pinball_loss,
    regret_nb,
)
from srsim.simulate import nb_params

LAM = np.array([4.0, 11.0, 27.0])
DISPERSION = np.array([0.25, 0.4, 0.15])
UNDERAGE, OVERAGE = 9.0, 2.0


def test_newsvendor_cost_is_scaled_pinball_loss():
    # The identity the whole of part A rests on.
    tau = critical_fractile(UNDERAGE, OVERAGE)
    demand = np.arange(0, 40, dtype=float)
    for q in (0.0, 7.0, 18.5, 39.0):
        assert np.allclose(
            newsvendor_cost(q, demand, UNDERAGE, OVERAGE),
            (UNDERAGE + OVERAGE) * pinball_loss(demand, q, tau),
        )


def test_expected_cost_matches_brute_force():
    grid = np.arange(0, 400)
    for lam, dispersion in zip(LAM, DISPERSION):
        n, p = nb_params(np.array([lam]), np.array([dispersion]))
        pmf = stats.nbinom.pmf(grid, n[0], p[0])
        for q in (0.0, 5.0, 20.0, 50.0):
            brute = float((pmf * newsvendor_cost(q, grid, UNDERAGE, OVERAGE)).sum())
            exact = expected_cost_nb([q], [lam], [dispersion], UNDERAGE, OVERAGE)[0]
            assert exact == pytest.approx(brute, rel=1e-6)


def test_optimal_order_minimises_expected_cost():
    grid = np.arange(0, 200, dtype=float)
    tau = critical_fractile(UNDERAGE, OVERAGE)
    q_star = optimal_order_nb(LAM, DISPERSION, tau)
    for i, (lam, dispersion) in enumerate(zip(LAM, DISPERSION)):
        costs = expected_cost_nb(
            grid, np.full_like(grid, lam), np.full_like(grid, dispersion), UNDERAGE, OVERAGE
        )
        assert grid[int(np.argmin(costs))] == pytest.approx(q_star[i])


def test_regret_is_zero_at_the_optimum_and_positive_elsewhere():
    tau = critical_fractile(UNDERAGE, OVERAGE)
    q_star = optimal_order_nb(LAM, DISPERSION, tau)
    assert np.allclose(regret_nb(q_star, LAM, DISPERSION, UNDERAGE, OVERAGE), 0.0)
    assert np.all(regret_nb(q_star + 6, LAM, DISPERSION, UNDERAGE, OVERAGE) > 0)
    assert np.all(regret_nb(np.maximum(q_star - 3, 0), LAM, DISPERSION, UNDERAGE, OVERAGE) > 0)


def test_ordering_the_mean_is_not_optimal_for_an_asymmetric_cost():
    # The point of part A: the mean is the wrong summary statistic.
    tau = critical_fractile(UNDERAGE, OVERAGE)
    mean_regret = regret_nb(np.round(LAM), LAM, DISPERSION, UNDERAGE, OVERAGE)
    star_regret = regret_nb(optimal_order_nb(LAM, DISPERSION, tau), LAM, DISPERSION, UNDERAGE, OVERAGE)
    assert np.all(mean_regret > star_regret)


def test_caveats_are_documented():
    assert len(equivalence_caveats()) == 6
