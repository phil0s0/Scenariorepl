"""The DES must be deterministic and must conserve units.

Determinism given (theta, omega) is what licenses common random numbers, which is
what makes the optimizer's objective surface stable. It is a property worth a test
rather than a comment.
"""

import numpy as np
import pytest

from srsim.des import CostParams, Policy, cost_percentile, service_metrics, simulate_costs
from srsim.scenarios import (
    make_point_scenarios,
    make_scenarios_from_truth,
)

COSTS = CostParams()
LAM = np.full(12, 10.0)
DISPERSION = np.full(12, 0.25)


@pytest.fixture(scope="module")
def scenarios():
    return make_scenarios_from_truth(LAM, DISPERSION, 300, seed=3)


def test_scenario_shapes(scenarios):
    assert scenarios.demand.shape == (300, 12)
    assert scenarios.lead_time.shape == scenarios.demand.shape
    assert scenarios.n_scenarios == 300 and scenarios.horizon == 12


def test_scenarios_are_reproducible():
    a = make_scenarios_from_truth(LAM, DISPERSION, 50, seed=11)
    b = make_scenarios_from_truth(LAM, DISPERSION, 50, seed=11)
    assert np.array_equal(a.demand, b.demand)
    assert np.array_equal(a.lead_time, b.lead_time)


def test_point_scenarios_have_no_demand_uncertainty():
    point = make_point_scenarios(np.arange(1.0, 13.0), 40, seed=2)
    assert np.all(point.demand == point.demand[0])
    # but lead-time and return uncertainty survive
    assert point.lead_time.std() > 0


def test_evaluator_is_deterministic(scenarios):
    policy = Policy(t0=1, q0=40, s=12, q=25, t_limit=9)
    first = simulate_costs(policy, scenarios, COSTS, initial_stock=10.0)
    second = simulate_costs(policy, scenarios, COSTS, initial_stock=10.0)
    assert np.array_equal(first, second)  # bit-identical, not just close


def test_cost_components_sum_to_total(scenarios):
    policy = Policy(t0=1, q0=40, s=12, q=25, t_limit=9)
    total, parts = simulate_costs(policy, scenarios, COSTS, 10.0, return_components=True)
    named = [v for k, v in parts.items() if not k.startswith("_")]
    assert np.allclose(sum(named), total)


def test_ordering_nothing_loses_exactly_all_demand(scenarios):
    # The strongest available end-to-end check on the accounting.
    policy = Policy(t0=0, q0=0, s=0, q=0, t_limit=0)
    total = simulate_costs(policy, scenarios, COSTS, initial_stock=0.0)
    expected = sum(
        COSTS.discount**t * COSTS.lost_sale_cost * scenarios.demand[:, t] for t in range(12)
    )
    assert np.allclose(total, expected)


def test_more_stock_improves_service(scenarios):
    lean = service_metrics(Policy(2, 10, 5, 10, 10), scenarios, COSTS, 5.0)
    rich = service_metrics(Policy(2, 80, 40, 60, 10), scenarios, COSTS, 40.0)
    assert rich["fill_rate"] > lean["fill_rate"]
    assert rich["holding"] > lean["holding"]
    assert rich["lost_sales"] < lean["lost_sales"]


def test_order_up_to_mode_tracks_a_target(scenarios):
    fixed = service_metrics(Policy(0, 0, 30, 25, 11), scenarios, COSTS, 20.0)
    up_to = service_metrics(
        Policy(0, 0, 30, 60, 11, order_up_to=True, require_no_in_transit=False),
        scenarios, COSTS, 20.0,
    )
    assert up_to["fill_rate"] > fixed["fill_rate"]


def test_service_metrics_are_proportions(scenarios):
    metrics = service_metrics(Policy(1, 40, 20, 30, 10), scenarios, COSTS, 20.0)
    assert 0.0 <= metrics["fill_rate"] <= 1.0
    assert 0.0 <= metrics["availability"] <= 1.0


def test_cost_percentile_is_monotone_in_the_level(scenarios):
    sample = simulate_costs(Policy(1, 40, 20, 30, 10), scenarios, COSTS, 20.0)
    levels = [cost_percentile(sample, p) for p in (25, 50, 75, 90)]
    assert levels == sorted(levels)


def test_t_limit_stops_ordering():
    scen = make_point_scenarios(np.full(12, 10.0), 5, seed=1)
    early = simulate_costs(Policy(0, 20, 100, 30, 2), scen, COSTS, 5.0, return_components=True)[1]
    late = simulate_costs(Policy(0, 20, 100, 30, 11), scen, COSTS, 5.0, return_components=True)[1]
    assert early["inbound"].mean() < late["inbound"].mean()
