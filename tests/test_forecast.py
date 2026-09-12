"""J-QPD properties, without paying for a cyclic-boosting fit.

The distribution layer is constructed directly from a quantile triplet here; the
expensive end-to-end fit is exercised by the notebooks.
"""

import numpy as np
import pytest
from cyclic_boosting.quantile_matching import J_QPD_S

from srsim.forecast import conformalize, make_forecast, sample_demand, survival_from_qpd

ALPHA = 0.1


@pytest.fixture(scope="module")
def forecast():
    # The third triplet is exactly log-symmetric (3 == sqrt(1 * 9)), which is the
    # case that makes a raw J_QPD_S return NaN; make_forecast must handle it.
    return make_forecast([2.0, 5.0, 1.0], [6.0, 12.0, 3.0], [14.0, 28.0, 9.0])


def test_log_symmetric_triplet_does_not_produce_nan():
    raw = J_QPD_S(ALPHA, np.array([1.0]), np.array([3.0]), np.array([9.0]), l=0.0)
    assert np.isnan(np.asarray(raw.ppf(0.5))).all()  # the library's behaviour
    guarded = make_forecast([1.0], [3.0], [9.0])
    assert np.all(np.isfinite(np.asarray(guarded.ppf(0.5))))


def test_qpd_recovers_its_pinned_quantiles(forecast):
    assert np.allclose(forecast.ppf(ALPHA), forecast.q_low, rtol=1e-4)
    assert np.allclose(forecast.ppf(0.5), forecast.q_median, rtol=1e-4)
    assert np.allclose(forecast.ppf(1 - ALPHA), forecast.q_high, rtol=1e-4)


def test_quantile_function_is_monotone(forecast):
    levels = np.linspace(0.01, 0.99, 60)
    curve = np.stack([np.asarray(forecast.ppf(u)) for u in levels])
    assert np.all(np.diff(curve, axis=0) >= -1e-9)


def test_support_is_non_negative(forecast):
    assert np.all(np.asarray(forecast.ppf(1e-4)) >= 0.0)


def test_order_up_to_ceilings_rather_than_rounds():
    # Demand is integer valued: the optimum is min{q : F(q) >= tau}. Rounding a
    # continuous quantile down under-serves; ceiling is correct.
    forecast = make_forecast([2.0], [6.0], [14.0])
    raw = float(np.asarray(forecast.ppf(0.7)))
    order = float(np.asarray(forecast.order_up_to(0.7)))
    assert order == pytest.approx(np.ceil(raw - 1e-9))
    assert order >= raw - 1e-9


def test_survival_matrix_is_decreasing_and_bounded(forecast):
    s = survival_from_qpd(forecast, np.array([1.0, 4.0, 8.0, 16.0, 32.0]))
    assert s.shape == (3, 5)
    assert np.all(np.diff(s, axis=1) <= 1e-9)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_sampling_matches_the_quantile_function(forecast):
    rng = np.random.default_rng(0)
    draws = sample_demand(forecast, 40_000, rng)
    assert draws.shape == (40_000, 3)
    assert np.all(draws >= 0)
    # the empirical median of the draws should sit near the pinned median
    for i, median in enumerate(forecast.q_median):
        assert np.median(draws[:, i]) == pytest.approx(median, abs=1.0)


def test_conformalize_widens_when_the_interval_is_too_narrow():
    rng = np.random.default_rng(1)
    n = 400
    narrow = make_forecast(np.full(n, 9.0), np.full(n, 10.05), np.full(n, 11.0))
    truth = rng.normal(10.0, 6.0, n)  # far wider than the predicted interval
    widened = conformalize(narrow, narrow, truth)
    assert np.all(widened.q_low <= narrow.q_low)
    assert np.all(widened.q_high >= narrow.q_high)
    before = np.mean((truth >= narrow.q_low) & (truth <= narrow.q_high))
    after = np.mean((truth >= widened.q_low) & (truth <= widened.q_high))
    assert after > before
    assert after == pytest.approx(1 - 2 * ALPHA, abs=0.05)


def test_conformalize_keeps_the_triplet_valid():
    # A degenerate triplet makes J_QPD_S return NaN silently rather than raise.
    n = 50
    tiny = make_forecast(np.full(n, 0.2), np.full(n, 0.42), np.full(n, 0.9))
    widened = conformalize(tiny, tiny, np.full(n, 0.4))
    assert np.all(widened.q_low > 0)
    assert np.all(widened.q_median > widened.q_low)
    assert np.all(widened.q_high > widened.q_median)
    assert np.all(np.isfinite(np.asarray(widened.ppf(0.3))))
