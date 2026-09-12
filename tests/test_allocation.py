"""The allocation LP is the riskiest piece of plumbing in the repo.

Three things must hold or tutorial 1 part B is quietly wrong: the LP must return
integral solutions (total unimodularity), the objective must come back as a
maximisation-sense **scalar** (a shape-(1,) value silently corrupts SPO+'s loss),
and the uncapacitated optimum must reproduce the newsvendor.
"""

import numpy as np
import pytest
from pyepo import EPO

from srsim.allocation import (
    TrancheAllocationModel,
    TrancheSpec,
    marginal_values,
    true_survival_matrix,
)
from srsim.newsvendor import critical_fractile, optimal_order_nb
from srsim.simulate import survival

SPEC = TrancheSpec(width=2, n_blocks=10)
LAM = np.array([4.0, 7.0, 11.0, 9.0])
DISPERSION = np.full(4, 0.25)
MARGIN = np.full(4, 8.0)
HOLDING = np.full(4, 2.0)


def _model(total_capacity=1e6, store_caps=(1e6, 1e6)):
    return TrancheAllocationModel(
        store_of_item=[0, 0, 1, 1],
        spec_width=SPEC.width,
        spec_blocks=SPEC.n_blocks,
        store_caps=list(store_caps),
        total_capacity=total_capacity,
    )


def test_survival_blocks_are_decreasing():
    s = true_survival_matrix(LAM, DISPERSION, SPEC)
    assert s.shape == (4, SPEC.n_blocks)
    assert np.all(np.diff(s, axis=1) <= 1e-12)


def test_marginal_values_decrease_and_cross_zero():
    v = marginal_values(true_survival_matrix(LAM, DISPERSION, SPEC), MARGIN, HOLDING)
    assert np.all(np.diff(v, axis=1) <= 1e-12)
    assert v[:, 0].min() > 0 and v[:, -1].max() < 0


def test_model_shape_and_sense():
    model = _model()
    assert model.modelSense == EPO.MAXIMIZE
    assert model.num_cost == 4 * SPEC.n_blocks


def test_solve_returns_scalar_objective_and_flat_solution():
    model = _model()
    v = marginal_values(true_survival_matrix(LAM, DISPERSION, SPEC), MARGIN, HOLDING)
    model.setObj(v.ravel())
    sol, obj = model.solve()
    # A shape-(1,) objective broadcasts to (batch, batch) inside SPOPlusFunc and
    # produces a wrong loss with no error, so this assertion is load-bearing.
    assert isinstance(obj, float)
    assert np.ndim(obj) == 0
    assert sol.shape == (model.num_cost,)
    assert obj == pytest.approx(float(v.ravel() @ sol))


def test_solutions_are_integral():
    # Laminar constraint family => totally unimodular => integral vertices.
    rng = np.random.default_rng(0)
    model = _model(total_capacity=30.0, store_caps=(20.0, 20.0))
    for _ in range(20):
        v = marginal_values(
            true_survival_matrix(rng.uniform(2, 18, 4), DISPERSION, SPEC), MARGIN, HOLDING
        )
        model.setObj(v.ravel())
        sol, _ = model.solve()
        assert np.allclose(sol, np.round(sol), atol=1e-7)


def test_capacity_is_respected():
    model = _model(total_capacity=18.0, store_caps=(12.0, 12.0))
    v = marginal_values(true_survival_matrix(LAM, DISPERSION, SPEC), MARGIN, HOLDING)
    model.setObj(v.ravel())
    sol, _ = model.solve()
    per_item = sol.reshape(4, SPEC.n_blocks).sum(axis=1)
    assert sol.sum() <= 18.0 + 1e-6
    assert per_item[:2].sum() <= 12.0 + 1e-6
    assert per_item[2:].sum() <= 12.0 + 1e-6


def test_uncapacitated_optimum_reproduces_the_newsvendor():
    # Drop the coupling and the LP must fall back to the critical fractile,
    # up to the resolution of the tranche grid.
    model = _model()
    v = marginal_values(true_survival_matrix(LAM, DISPERSION, SPEC), MARGIN, HOLDING)
    model.setObj(v.ravel())
    sol, _ = model.solve()
    allocated = sol.reshape(4, SPEC.n_blocks).sum(axis=1)
    tau = critical_fractile(MARGIN, HOLDING)
    q_star = optimal_order_nb(LAM, DISPERSION, tau)
    assert np.all(np.abs(allocated - q_star) <= SPEC.width)


def test_linearisation_is_exact_at_block_boundaries():
    # Filling whole blocks must reproduce the true expected profit of that stock
    # level, or the regret measured in tranche space means nothing.
    model = _model()
    v = marginal_values(true_survival_matrix(LAM, DISPERSION, SPEC), MARGIN, HOLDING)
    model.setObj(v.ravel())
    sol, _ = model.solve()
    per_item = sol.reshape(4, SPEC.n_blocks)
    for i in range(4):
        stocked = int(per_item[i].sum())
        units = np.arange(1, stocked + 1)
        exact = ((MARGIN[i] + HOLDING[i]) * survival(units, LAM[i], DISPERSION[i]) - HOLDING[i]).sum()
        assert float(v[i] @ per_item[i]) == pytest.approx(exact, rel=1e-9)
