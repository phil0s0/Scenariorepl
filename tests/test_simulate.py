"""The simulator has to be reproducible and its ground truth has to be true."""

import numpy as np
import pytest
from scipy import stats

from srsim.simulate import (
    FEATURE_COLUMNS,
    PanelConfig,
    censor_observations,
    nb_params,
    sample_lead_times,
    simulate_panel,
    survival,
)

CONFIG = PanelConfig(n_products=25, n_locations=3, n_weeks=60, seed=7)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(CONFIG)


def test_reproducible(panel):
    again = simulate_panel(CONFIG)
    assert panel["sales"].equals(again["sales"])
    assert np.allclose(panel["lam"], again["lam"])


def test_seed_changes_the_world(panel):
    other = simulate_panel(PanelConfig(**{**CONFIG.__dict__, "seed": 8}))
    assert not np.allclose(panel["lam"].to_numpy()[:100], other["lam"].to_numpy()[:100])


def test_features_exclude_ground_truth():
    # If the true intensity leaks into the features, every regret number in the
    # tutorials is meaningless.
    assert "lam" not in FEATURE_COLUMNS
    assert "dispersion" not in FEATURE_COLUMNS


def test_realised_sales_track_true_intensity(panel):
    # The law of large numbers should hold to within Monte Carlo error.
    mean_lam = panel["lam"].mean()
    mean_sales = panel["sales"].mean()
    assert mean_sales == pytest.approx(mean_lam, rel=0.05)


def test_nb_params_reproduce_mean_and_variance(panel):
    lam = panel["lam"].to_numpy()[:500]
    dispersion = panel["dispersion"].to_numpy()[:500]
    n, p = nb_params(lam, dispersion)
    assert np.allclose(stats.nbinom.mean(n, p), lam)
    assert np.allclose(stats.nbinom.var(n, p), lam + dispersion * lam**2)


def test_survival_is_a_survival_function(panel):
    lam = panel["lam"].to_numpy()[:200]
    dispersion = panel["dispersion"].to_numpy()[:200]
    levels = np.arange(0, 40)
    values = np.stack([survival(q, lam, dispersion) for q in levels])
    assert np.all(np.diff(values, axis=0) <= 1e-12)       # non-increasing in q
    assert np.allclose(values[0], 1.0)                     # P(D >= 0) == 1
    assert np.all(values >= 0) and np.all(values <= 1.0)


def test_survival_matches_empirical_frequency():
    rng = np.random.default_rng(0)
    lam, dispersion = np.array([9.0]), np.array([0.25])
    n, p = nb_params(lam, dispersion)
    draws = rng.negative_binomial(n[0], p[0], size=200_000)
    for q in (3, 9, 15):
        assert survival(q, lam, dispersion)[0] == pytest.approx((draws >= q).mean(), abs=0.01)


def test_lead_times_are_positive_whole_weeks():
    rng = np.random.default_rng(1)
    lead = sample_lead_times(rng, 5000, mean=2.0, cv=0.4)
    assert lead.dtype.kind == "i"
    assert lead.min() >= 1
    assert lead.mean() == pytest.approx(2.5, abs=0.2)  # ceil() shifts the mean up


def test_censoring_never_exceeds_stock():
    demand = np.array([5.0, 12.0, 0.0])
    on_hand = np.array([8.0, 4.0, 3.0])
    assert np.array_equal(censor_observations(demand, on_hand), [5.0, 4.0, 0.0])
