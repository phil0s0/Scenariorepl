"""The deterministic cost evaluator ``C(theta, omega)``.

A weekly discrete-event simulation of one SKU over a 12-week horizon, following
the event ordering in Presbitero et al.  The critical property is in the name:
given a policy ``theta`` and a scenario ``omega``, this function is
**deterministic**.  All randomness lives in the scenario.  That is what lets the
optimizer reuse one scenario set across every candidate policy and see a stable
objective surface instead of Monte Carlo noise.

Event ordering within a week, matching the paper:

1.  *Half* of this week's scheduled inbound arrivals and expected returns are
    added to on-hand stock -- the paper's way of saying that deliveries land at
    some point during the week rather than neatly at the start.
2.  Demand is realised and filled from available stock; anything unfilled is
    lost, not backordered.
3.  The remaining half of arrivals and returns is added.
4.  End-of-week stock is recorded and costs accrue.
5.  At a review point the policy decides whether to order.

The policy is the paper's extended ``(R, s, Q)``:

    theta = (t0, Q0, s, Q, t_limit)

``t0`` and ``Q0`` time and size an initial "kickstart" order, ``(s, Q)`` is the
ordinary reorder rule on a review period ``R``, and ``t_limit`` is an ordering
cutoff that stops the policy buying stock it cannot sell before the season ends.
Only ``(t0, Q0)`` are exposed to merchants in the real system.

Everything is vectorised over the scenario axis: the loop runs over the 12 weeks,
never over the (hundreds or thousands of) scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenarios import ScenarioSet

__all__ = ["CostParams", "Policy", "simulate_costs", "cost_percentile", "service_metrics"]


@dataclass(frozen=True)
class CostParams:
    """Unit economics of the replenishment problem."""

    holding_fee: float = 0.35      # per unit of end-of-week stock
    inbound_fee: float = 0.25      # per unit shipped in
    outbound_fee: float = 0.40     # per unit shipped to a customer
    returns_fee: float = 0.60      # per unit processed back
    price: float = 24.0
    unit_cost: float = 12.0
    return_rate: float = 0.25
    #: Weekly discount on costs; the paper mentions an exponential decay factor
    #: that damps late-horizon volatility.
    discount: float = 0.98

    @property
    def lost_sale_cost(self) -> float:
        """Margin forgone per unit of unmet demand.

        Only the fraction of demand that would have *stayed* sold counts, hence
        the ``(1 - return_rate)`` factor -- a returned unit was never margin.
        """
        return (self.price - self.unit_cost) * (1.0 - self.return_rate)


@dataclass(frozen=True)
class Policy:
    """Extended ``(R, s, Q)`` parameters.  ``R`` is operational, not optimized.

    ``order_up_to`` switches the reorder rule from "order a fixed ``q``" to
    "order enough to bring the inventory position up to ``q``".  That one flag is
    what lets the same evaluator run the classical baselines -- an ``(s, S)``
    policy is ``order_up_to=True``, and a periodic base-stock policy is the same
    thing with ``s`` set high enough that every review triggers an order.
    """

    t0: int
    q0: float
    s: float
    q: float
    t_limit: int
    review_period: int = 2
    order_up_to: bool = False
    #: The paper's extended (R,s,Q) triggers on end-of-week **on-hand stock**
    #: and only reorders when nothing is already in transit -- the guard is what
    #: stops a trigger that ignores the pipeline from ordering the same
    #: replenishment twice.  The two go together: with the guard on we compare
    #: on-hand against ``s``; with it off we compare the full inventory position,
    #: which is what classical order-up-to policies do.  Triggering on position
    #: *and* keeping the guard would double-count the pipeline and handicap the
    #: policy, which would rig the comparison.
    require_no_in_transit: bool = True

    def as_array(self) -> np.ndarray:
        return np.array([self.t0, self.q0, self.s, self.q, self.t_limit], dtype=float)


def simulate_costs(
    policy: Policy,
    scenarios: ScenarioSet,
    costs: CostParams,
    initial_stock: float = 0.0,
    return_components: bool = False,
):
    """Total discounted cost of ``policy`` under every scenario.

    Returns an array of shape ``(n_scenarios,)`` -- one cost sample per
    scenario, i.e. a draw from the cost distribution ``C(theta)``.
    """
    n, horizon = scenarios.demand.shape
    max_lead = int(scenarios.lead_time.max()) + 1
    max_delay = int(scenarios.return_delay.max()) + 1
    buffer = horizon + max_lead + max_delay + 2

    on_hand = np.full(n, float(initial_stock))
    pipeline = np.zeros((n, buffer))        # units arriving in week t
    returns_pipe = np.zeros((n, buffer))    # units re-entering stock in week t

    total = np.zeros(n)
    components = {k: np.zeros(n) for k in ("holding", "inbound", "outbound", "returns", "lost_sales")}
    demand_total = np.zeros(n)
    sales_total = np.zeros(n)
    weeks_in_stock = np.zeros(n)
    rows = np.arange(n)

    t0 = int(round(policy.t0))
    t_limit = int(round(policy.t_limit))
    q0 = max(float(policy.q0), 0.0)
    reorder_point = max(float(policy.s), 0.0)
    reorder_qty = max(float(policy.q), 0.0)

    for t in range(horizon):
        discount = costs.discount**t

        arrivals = pipeline[:, t]
        returning = returns_pipe[:, t]

        # 1. half of inbound and returns land before demand is served
        on_hand += 0.5 * (arrivals + returning)

        # 2. demand is realised; unmet demand is lost, not backordered
        demand = scenarios.demand[:, t]
        sales = np.minimum(demand, on_hand)
        unmet = demand - sales
        on_hand -= sales
        demand_total += demand
        sales_total += sales
        weeks_in_stock += (unmet <= 1e-9).astype(float)

        # 3. the rest lands after
        on_hand += 0.5 * (arrivals + returning)

        # 4. returns generated by this week's sales come back later
        returned = costs.return_rate * sales
        back_at = np.minimum(t + scenarios.return_delay[:, t], buffer - 1)
        np.add.at(returns_pipe, (rows, back_at), returned)

        # 5. costs accrue on end-of-week state
        components["holding"] += discount * costs.holding_fee * on_hand
        components["outbound"] += discount * costs.outbound_fee * sales
        components["returns"] += discount * costs.returns_fee * returned
        components["lost_sales"] += discount * costs.lost_sale_cost * unmet

        # 6. review and order.  Branch-free so the whole thing stays vectorised.
        in_transit = pipeline[:, t + 1 :].sum(axis=1)
        position = on_hand + in_transit

        is_review = (t % policy.review_period) == 0
        within_limit = t <= t_limit

        place_initial = (t == t0) and within_limit
        can_reorder = is_review and within_limit and (t > t0)
        if policy.require_no_in_transit:
            trigger_level = on_hand
            eligible = can_reorder & (trigger_level <= reorder_point) & (in_transit <= 1e-9)
        else:
            eligible = can_reorder & (position <= reorder_point)
        place_ongoing = eligible

        if policy.order_up_to:
            target = np.maximum(reorder_qty - position, 0.0)
        else:
            target = np.full(n, reorder_qty)
        order = np.where(place_ongoing, target, 0.0)
        if place_initial:
            order = np.full(n, q0)

        components["inbound"] += discount * costs.inbound_fee * order
        arrive_at = np.minimum(t + scenarios.lead_time[:, t], buffer - 1)
        np.add.at(pipeline, (rows, arrive_at), order)

    for value in components.values():
        total += value
    if return_components:
        components["_fill_rate"] = np.divide(
            sales_total, demand_total, out=np.ones_like(sales_total), where=demand_total > 0
        )
        components["_availability"] = weeks_in_stock / horizon
        return total, components
    return total


def cost_percentile(costs: np.ndarray, percentile: float = 75.0) -> float:
    """The risk-aware objective: a percentile of the cost distribution.

    Presbitero et al. minimise the 75th percentile rather than the mean, as a
    tractable stand-in for a CVaR-style criterion.  The cost distribution is
    asymmetric -- understocking a winner hurts far more than overstocking a dud --
    so optimising the mean quietly accepts a fat right tail.
    """
    return float(np.percentile(costs, percentile))


def service_metrics(policy: Policy, scenarios: ScenarioSet, costs: CostParams, initial_stock: float = 0.0) -> dict:
    """Cost and service KPIs for one policy, averaged over scenarios.

    ``fill_rate`` is the share of demand served from stock; ``availability`` is
    the share of weeks that ended with no unmet demand.  These are the operational
    counterparts to the cost objective, and the pair Presbitero et al. report.
    """
    total, comp = simulate_costs(policy, scenarios, costs, initial_stock, return_components=True)
    return {
        "cost_p75": cost_percentile(total, 75.0),
        "cost_mean": float(total.mean()),
        "fill_rate": float(comp["_fill_rate"].mean()),
        "availability": float(comp["_availability"].mean()),
        "holding": float(comp["holding"].mean()),
        "lost_sales": float(comp["lost_sales"].mean()),
    }
