"""A small weekly retail demand simulator with known ground truth.

The data-generating process is modelled on Felix Wick's
`demand_forecasting_simulation <https://github.com/FelixWick/demand_forecasting_simulation>`_
(EPL-2.0), which is the simulator behind the Blue Yonder / cyclic-boosting
demand-forecasting work.  We re-implement rather than vendor it, for four
reasons that matter for these tutorials:

1.  Upstream deletes the true intensity ``LAMBDA`` before writing its output.
    We *keep* it, along with the exact negative-binomial parameters of every
    row, because that is what makes true decision regret computable rather
    than merely estimable.
2.  Upstream is daily; replenishment review periods are weekly.
3.  Upstream has no inventory layer -- no lead times, no stock-out censoring.
4.  Upstream hardcodes its configuration and ends in an interactive shell.

What we keep is the shape of the process, which is the part that makes it a
worthwhile teaching example: everything is additive in ``log lambda`` (a
multiplicative demand model), effects are hierarchical, price enters through a
per-product elasticity, promotions are *deliberately confounded* with season
and product group, and the observation noise is negative-binomial with a
heteroscedastic dispersion rather than Poisson.

The single most important export is :func:`survival`.  It is not a debugging
convenience: ``P(D >= q)`` under the true model is literally the cost vector
that the allocation problem in tutorial 1 is trying to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "PanelConfig",
    "simulate_panel",
    "survival",
    "nb_params",
    "sample_lead_times",
    "censor_observations",
    "FEATURE_COLUMNS",
    "TRUTH_COLUMNS",
]

#: Columns a forecasting model is allowed to see.  ``lam`` is deliberately absent:
#: the whole regret benchmark is vacuous if the true intensity leaks into features.
FEATURE_COLUMNS = [
    "P_ID",
    "L_ID",
    "PG_ID",
    "week_of_year",
    "price_ratio",
    "promo",
    "trend",
    "age",
]

#: Ground-truth columns.  Available to the *evaluator*, never to a model.
TRUTH_COLUMNS = ["lam", "dispersion"]


@dataclass(frozen=True)
class PanelConfig:
    """Size and shape of the simulated world."""

    n_products: int = 150
    n_locations: int = 4
    n_weeks: int = 104
    n_groups: int = 8
    seed: int = 20250101

    #: Unit economics.  Margin and holding cost per unit per week, drawn per SKU.
    margin_mean: float = 8.0
    holding_mean: float = 2.0

    #: Share of products with a finite lifecycle (launch, peak, decline) rather
    #: than steady-state demand.  This matters more than it looks: the extra
    #: parameters of the paper's extended (R,s,Q) policy -- an initial order
    #: (t0, Q0) and an ordering cutoff t_limit -- only earn their keep when demand
    #: has a shape to be positioned against.  On steady-state demand a plain
    #: base-stock policy is very hard to beat, and should be.
    lifecycle_fraction: float = 0.45


def _group_offsets(rng: np.random.Generator, keys: np.ndarray, sigma: float) -> np.ndarray:
    """One scalar Gaussian offset per distinct key, broadcast back over rows.

    This mirrors upstream's ``groupby(...).apply(gaussian_noise, sigma)``: these
    are genuine group-level random effects, not per-row noise.
    """
    uniq, inverse = np.unique(keys, return_inverse=True)
    return rng.normal(0.0, sigma, size=uniq.size)[inverse]


def simulate_panel(config: PanelConfig | None = None, **overrides) -> pd.DataFrame:
    """Simulate a weekly SKU x location panel with its ground truth attached.

    Returns one row per (product, location, week) with observed ``sales`` and the
    true negative-binomial parameters (``lam``, ``dispersion``) that generated it.
    Fully deterministic given ``config.seed``.
    """
    config = (config or PanelConfig())
    if overrides:
        config = PanelConfig(**{**config.__dict__, **overrides})
    rng = np.random.default_rng(config.seed)

    n_p, n_l, n_w = config.n_products, config.n_locations, config.n_weeks

    # --- static entity attributes -------------------------------------------------
    product_group = rng.integers(0, config.n_groups, size=n_p)
    # Elasticity: lognormal around 1.5, clipped, varying by product and group.
    log_elast = np.log(1.5) + _group_offsets(rng, product_group, 0.3) + rng.normal(0, 0.3, n_p)
    elasticity = np.clip(np.exp(log_elast), 0.0, 3.0)
    normal_price = np.round(10.0 + rng.uniform(0, 5, n_p) * rng.exponential(1.2, n_p), 2)
    normal_price = np.clip(normal_price, 1.0, 100.0)

    # Per-SKU unit economics, used by the newsvendor and the allocation LP.
    margin = np.round(config.margin_mean * rng.lognormal(0.0, 0.25, n_p), 2)
    holding = np.round(config.holding_mean * rng.lognormal(0.0, 0.25, n_p), 2)

    # Seasonality only affects a random subset -- half get annual, half semi-annual.
    seasonal_products = rng.random(n_p) < 0.5
    seasonal_amp = np.where(seasonal_products, rng.exponential(0.5, n_p).clip(max=1.5), 0.0)
    seasonal_half = rng.random(n_p) < 0.5
    # Trend affects a small minority in each direction.
    trend_dir = rng.choice([-1.0, 0.0, 1.0], size=n_p, p=[0.05, 0.90, 0.05])

    # Product lifecycles: a launch week and a life length, with a hump-shaped
    # intensity profile in between.  Launches are spread so that a good share of
    # products are in their decline phase during the evaluation window.
    has_lifecycle = rng.random(n_p) < config.lifecycle_fraction
    launch_week = rng.integers(-30, max(n_w - 25, 1), size=n_p)
    life_length = rng.integers(70, 130, size=n_p)

    # --- the panel grid -----------------------------------------------------------
    p_idx = np.repeat(np.arange(n_p), n_l * n_w)
    l_idx = np.tile(np.repeat(np.arange(n_l), n_w), n_p)
    w_idx = np.tile(np.arange(n_w), n_p * n_l)

    df = pd.DataFrame(
        {
            "P_ID": p_idx,
            "L_ID": l_idx,
            "week": w_idx,
            "PG_ID": product_group[p_idx],
        }
    )
    df["week_of_year"] = df["week"] % 52
    df["trend"] = df["week"] / max(n_w - 1, 1)

    # --- log-additive demand intensity --------------------------------------------
    log_lam = np.full(len(df), 2.2)
    log_lam += _group_offsets(rng, df["PG_ID"].to_numpy(), 0.5)
    log_lam += _group_offsets(rng, df["L_ID"].to_numpy(), 0.5)
    log_lam += _group_offsets(rng, df["P_ID"].to_numpy(), 0.3)
    log_lam += _group_offsets(rng, df["P_ID"].to_numpy() * n_l + df["L_ID"].to_numpy(), 0.1)

    # Seasonality: annual or semi-annual sine, per product.
    period = np.where(seasonal_half[p_idx], 26.0, 52.0)
    log_lam += seasonal_amp[p_idx] * np.sin(2 * np.pi * df["week_of_year"].to_numpy() / period)

    # Trend, linear in normalised week index.
    log_lam += trend_dir[p_idx] * 1.2 * (df["trend"].to_numpy() - 0.5)

    # Lifecycle multiplier: 0 before launch and after end of life, humped between.
    age_weeks = df["week"].to_numpy() - launch_week[p_idx]
    life = life_length[p_idx].astype(float)
    frac = np.clip(age_weeks / life, 0.0, 1.0)
    # Beta-shaped hump peaking early in life, normalised to a maximum of 1.
    a, b = 1.6, 2.4
    peak = ((a - 1) / (a + b - 2)) if (a > 1 and b > 1) else 0.5
    shape = (frac**(a - 1)) * ((1 - frac) ** (b - 1))
    shape = shape / ((peak ** (a - 1)) * ((1 - peak) ** (b - 1)))
    alive = (age_weeks >= 0) & (age_weeks <= life)
    multiplier = np.where(has_lifecycle[p_idx], np.clip(shape, 0.08, 1.0), 1.0)
    log_lam += np.log(multiplier)
    df["age"] = np.clip(age_weeks, 0, 200)
    df["has_lifecycle"] = has_lifecycle[p_idx]

    # Products that have not launched yet, or are past end of life, simply are
    # not in the data -- exactly as a real sales table has no rows for them.
    # (Upstream does the same thing with its `restrict_pl_ranges` step.) Keeping
    # them as near-zero rows would just add noise the forecaster cannot use.
    in_range = (~has_lifecycle[p_idx]) | alive
    keep = np.asarray(in_range)
    df = df.loc[keep].reset_index(drop=True)
    log_lam = log_lam[keep]
    p_idx = p_idx[keep]

    # --- promotions, deliberately confounded --------------------------------------
    # Promotion probability rises with week-of-year AND with product group, so a
    # model that ignores either will attribute promo lift to season, or vice versa.
    promo_prob = (
        df["week_of_year"].to_numpy() / 51.0 + df["PG_ID"].to_numpy() / (config.n_groups - 1)
    ) / 8.0
    promo = (rng.random(len(df)) < promo_prob).astype(np.int8)
    df["promo"] = promo

    # Price: promotions discount, with per-row variation. Non-promo weeks sit at
    # the normal price, so `price_ratio` is exactly 1.0 there.
    discount = np.where(promo == 1, rng.uniform(0.65, 0.9, len(df)), 1.0)
    df["normal_price"] = normal_price[p_idx]
    df["price"] = np.round(df["normal_price"].to_numpy() * discount, 2)
    df["price_ratio"] = df["price"] / df["normal_price"]

    # Price effect, exactly upstream's form: relative discount times elasticity.
    log_lam += (1.0 - df["price_ratio"].to_numpy()) * elasticity[p_idx]
    # A direct promo uplift on top of the price effect, for a subset of promotions.
    direct_promo = (promo == 1) & (rng.random(len(df)) < 0.25)
    log_lam += np.where(direct_promo, rng.normal(0.5, 0.1, len(df)), 0.0)

    log_lam = np.clip(log_lam, -2.0, 5.0)
    lam = np.exp(log_lam)

    # --- negative-binomial observation noise (NB2) --------------------------------
    # var = lam + dispersion * lam^2, heteroscedastic by hierarchy and promotion.
    dispersion = np.full(len(df), 0.25)
    dispersion += _group_offsets(rng, df["PG_ID"].to_numpy(), 0.05)
    dispersion += _group_offsets(rng, df["P_ID"].to_numpy(), 0.05)
    dispersion += np.where(promo == 1, 0.10, 0.0)
    dispersion = np.clip(dispersion, 0.02, 1.0)

    n_param, p_param = nb_params(lam, dispersion)
    df["sales"] = rng.negative_binomial(n_param, p_param)

    df["lam"] = lam
    df["dispersion"] = dispersion
    df["margin"] = margin[p_idx]
    df["holding"] = holding[p_idx]
    df["elasticity"] = elasticity[p_idx]

    return df


def nb_params(lam: np.ndarray, dispersion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NB2 mean/dispersion -> scipy's ``(n, p)`` parameterisation.

    With ``var = lam + dispersion * lam**2`` we have ``n = 1 / dispersion`` and
    ``p = n / (n + lam)``, which gives ``mean == lam`` exactly.
    """
    lam = np.asarray(lam, dtype=float)
    dispersion = np.asarray(dispersion, dtype=float)
    n = 1.0 / dispersion
    p = n / (n + lam)
    return n, p


def survival(q, lam, dispersion) -> np.ndarray:
    """``P(D >= q)`` under the true demand distribution.

    This is the object the allocation problem in tutorial 1 predicts: the
    marginal value of the unit that takes stock from ``q-1`` to ``q`` is
    ``margin * P(D >= q) - holding``.  Demand is integer-valued, so the
    survival function is evaluated with ``sf(q - 1)``, not ``sf(q)``.
    """
    n, p = nb_params(lam, dispersion)
    return stats.nbinom.sf(np.asarray(q, dtype=float) - 1, n, p)


def sample_lead_times(
    rng: np.random.Generator, size, mean: float = 2.0, cv: float = 0.4
) -> np.ndarray:
    """Replenishment lead times in weeks, gamma-distributed.

    Presbitero et al. model replenishment lead times with a gamma distribution
    parameterised by user-specified base values; ``mean`` and ``cv`` are that
    parameterisation. Draws are rounded up to whole weeks -- an order that
    arrives mid-week is only usable from the following review point.
    """
    shape = 1.0 / (cv**2)
    scale = mean / shape
    return np.ceil(rng.gamma(shape, scale, size=size)).astype(int)


def censor_observations(demand: np.ndarray, on_hand: np.ndarray) -> np.ndarray:
    """``observed = min(demand, on_hand)`` -- what a sales log actually records.

    Fitting a demand model to censored sales biases it low, which biases every
    downstream order quantity low, which causes more censoring.  Tutorial 1 turns
    this on in one short section to show the effect; it is off by default.
    """
    return np.minimum(np.asarray(demand), np.asarray(on_hand))
