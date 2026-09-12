"""Probabilistic demand forecasts: cyclic-boosting quantiles -> a J-QPD.

The output contract is the thing to keep in mind: every downstream consumer in
these tutorials wants a *quantile function* ``Q(u)`` per SKU-week, not a point
forecast.

- The newsvendor (tutorial 1, part A) reads one point off it: ``Q(tau)``.
- The allocation LP (tutorial 1, part B) reads a whole discretised survival
  curve off it.
- The scenario generator (tutorial 2) inverts it: ``d = Q(u)``, ``u ~ U(0,1)``.

A Johnson Quantile-Parameterized Distribution (Hadlock & Bickel 2017) is a good
fit for that contract: it is parameterised directly by a symmetric-percentile
triplet, its quantile function is closed form, and the semi-bounded variant
``J_QPD_S`` with lower bound 0 respects the fact that demand cannot be negative.

Two routes to the triplet are provided, because the difference between them is
worth teaching:

``fit_qpd_independent``
    Three independently-fitted quantile models.  Simple and transparent, but
    nothing stops the fitted quantiles from *crossing* -- a predicted 10th
    percentile above the predicted median.  We measure how often it happens and
    repair it by sorting.
``fit_qpd_chain``
    ``QPD_RegressorChain``, which conditions each model on the previous one so
    the triplet is ordered by construction.  Note the non-obvious calibration:
    for a triplet at ``alpha``, the lower model must be trained at quantile
    ``2 * alpha`` and the upper at ``2 * (1 - alpha) - 1``.

Monotonicity is not cosmetic here.  The allocation LP in part B is only
well-posed -- "fill the cheap tranches first" -- if the marginal-value vector is
decreasing, and that follows from the survival function being decreasing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from cyclic_boosting import flags
from cyclic_boosting.pipelines import (
    pipeline_CBAdditiveQuantileRegressor,
    pipeline_CBMultiplicativeQuantileRegressor,
)
from cyclic_boosting.quantile_matching import J_QPD_S, QPD_RegressorChain

#: Minimum separation between the J-QPD's three pinned quantiles, and between the
#: lowest of them and the distribution's lower bound.
_FLOOR = 1e-2

#: Minimum distance from exact log-symmetry of the quantile triplet.  ``J_QPD_S``
#: computes ``delta = sinh(arccosh((H - L) / (2 * min(B - L, H - B)))) / c`` in log
#: space.  When the median is exactly the geometric mean of the outer quantiles the
#: argument of ``arccosh`` is exactly 1, so ``delta == 0`` and the shape parameter
#: divides by zero -- the object is constructed without complaint and every ``ppf``
#: call then returns NaN.  A lognormal-looking forecast lands on precisely that
#: knife edge, so we nudge the median off it.
_LOG_SYMMETRY_EPS = 1e-3


def _break_log_symmetry(q_low, q_median, q_high):
    """Nudge the median away from the geometric mean of the outer quantiles."""
    q_low = np.asarray(q_low, dtype=float)
    q_median = np.asarray(q_median, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    imbalance = np.log(q_low) + np.log(q_high) - 2 * np.log(q_median)
    degenerate = np.abs(imbalance) < _LOG_SYMMETRY_EPS
    if np.any(degenerate):
        nudged = q_median * (1.0 + _LOG_SYMMETRY_EPS)
        q_median = np.where(degenerate, np.minimum(nudged, q_high - _FLOOR), q_median)
    return q_low, q_median, q_high

__all__ = [
    "QPDForecast",
    "feature_properties",
    "feature_groups",
    "fit_qpd_independent",
    "fit_qpd_chain",
    "conformalize",
    "make_forecast",
    "sample_demand",
    "inverse_transform",
    "survival_from_qpd",
]


def feature_properties() -> dict:
    """Per-column cyclic-boosting flags.

    Exactly one of CONTINUOUS / ORDERED / UNORDERED must be set per feature.
    ``week_of_year`` is marked seasonal so the smoother wraps December into
    January instead of treating them as maximally distant bins.
    """
    return {
        "P_ID": flags.IS_UNORDERED,
        "L_ID": flags.IS_UNORDERED,
        "PG_ID": flags.IS_UNORDERED,
        "week_of_year": flags.IS_CONTINUOUS | flags.IS_SEASONAL,
        "price_ratio": flags.IS_CONTINUOUS,
        "promo": flags.IS_ORDERED,
        "trend": flags.IS_CONTINUOUS,
        "age": flags.IS_CONTINUOUS,
    }


def feature_groups() -> list:
    """Main effects plus the interactions the DGP actually contains.

    ``(P_ID, L_ID)`` picks up the product-by-location random effect that the
    simulator puts in.  A ``(PG_ID, week_of_year)`` interaction would help the
    model further disentangle the promotion effect from the seasonality it is
    confounded with, but it triples fitting time for a modest gain, so it is
    left out here and flagged in the notebook as the obvious next thing to add.
    """
    return [
        "P_ID",
        "L_ID",
        "PG_ID",
        "week_of_year",
        "price_ratio",
        "promo",
        "trend",
        "age",
        ("P_ID", "L_ID"),
    ]


@dataclass
class QPDForecast:
    """A fitted predictive distribution over a set of rows.

    ``qpd`` is vectorised: ``qpd.ppf(0.9)`` returns one value per row, and
    ``qpd.ppf(u)`` with ``u`` of shape ``(k,)`` returns shape ``(k, n_rows)``.
    """

    qpd: J_QPD_S
    alpha: float
    q_low: np.ndarray
    q_median: np.ndarray
    q_high: np.ndarray
    crossing_rate: float

    def ppf(self, u):
        return self.qpd.ppf(u)

    def order_up_to(self, tau) -> np.ndarray:
        """Smallest integer order quantity meeting service level ``tau``.

        Demand is integer-valued, so this is ``ceil``, never ``round``: rounding
        a quantile down silently under-serves by up to half a unit of demand.
        """
        return np.ceil(np.asarray(self.qpd.ppf(tau), dtype=float) - 1e-9)


def conformalize(
    forecast: QPDForecast,
    calib_forecast: QPDForecast,
    calib_y: np.ndarray,
) -> QPDForecast:
    """Split-conformal (CQR) widening of the predicted interval.

    Quantile regression gives no coverage guarantee: fitted quantiles are
    typically too narrow out of sample, especially out of *time*.  Conformalized
    quantile regression fixes this with one scalar, computed on a held-out
    calibration split:

        E_i = max(q_low_i - y_i, y_i - q_high_i)

    and the interval is widened by the ``(1 - alpha)(1 + 1/n)`` empirical
    quantile of those scores.  This is the step Presbitero et al. describe as
    "conformal calibration", and it is what makes the quantiles that feed the
    optimizer mean what they say.

    Note the honest limitation: this restores *marginal* coverage of the
    ``[alpha, 1 - alpha]`` interval, not conditional coverage, and it shifts the
    two outer quantiles without re-fitting the median.
    """
    alpha = forecast.alpha
    scores = np.maximum(calib_forecast.q_low - calib_y, calib_y - calib_forecast.q_high)
    n = len(scores)
    level = min(1.0, (1.0 - 2 * alpha) * (1.0 + 1.0 / n))
    pad = float(np.quantile(scores, level))

    # The lower quantile must stay strictly inside the support: J_QPD_S is
    # degenerate (and silently returns NaN) if qv_low touches the lower bound l.
    q_low = np.clip(forecast.q_low - pad, _FLOOR, None)
    q_high = np.maximum(forecast.q_high + pad, q_low + 2 * _FLOOR)
    q_median = np.clip(forecast.q_median, q_low + _FLOOR, q_high - _FLOOR)
    q_low, q_median, q_high = _break_log_symmetry(q_low, q_median, q_high)

    qpd = J_QPD_S(alpha, q_low, q_median, q_high, l=0.0)
    return QPDForecast(qpd, alpha, q_low, q_median, q_high, forecast.crossing_rate)


def _quantile_pipeline(quantile: float, multiplicative: bool, max_iter: int):
    factory = (
        pipeline_CBMultiplicativeQuantileRegressor
        if multiplicative
        else pipeline_CBAdditiveQuantileRegressor
    )
    return factory(
        quantile=quantile,
        feature_properties=feature_properties(),
        feature_groups=feature_groups(),
        maximal_iterations=max_iter,
    )


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    from .simulate import FEATURE_COLUMNS

    return df[FEATURE_COLUMNS].reset_index(drop=True)


def fit_qpd_independent(
    train: pd.DataFrame,
    predict_on: pd.DataFrame,
    alpha: float = 0.1,
    target: str = "sales",
    max_iter: int = 10,
) -> QPDForecast:
    """Three independent quantile fits at ``alpha``, ``0.5``, ``1 - alpha``.

    Quantile crossing is measured and then repaired by sorting the triplet,
    which is the projection onto the ordered set -- the minimal change that
    makes the J-QPD constructible.
    """
    x_train, y = _prep(train), train[target].to_numpy(dtype=float)
    x_pred = _prep(predict_on)

    preds = []
    for q in (alpha, 0.5, 1.0 - alpha):
        model = _quantile_pipeline(q, multiplicative=True, max_iter=max_iter)
        model.fit(x_train.copy(), y)
        preds.append(np.asarray(model.predict(x_pred.copy()), dtype=float))

    q_low, q_median, q_high = preds
    crossing = float(np.mean((q_low > q_median) | (q_high < q_median)))

    # Project onto the ordered set, and keep the triplet strictly increasing so
    # the J-QPD is non-degenerate.
    stacked = np.sort(np.vstack(preds), axis=0)
    q_low, q_median, q_high = stacked
    # Strictly increasing, and strictly above the lower bound l = 0 -- J_QPD_S
    # returns NaN rather than raising if either condition is violated.
    q_low = np.maximum(q_low, _FLOOR)
    q_median = np.maximum(q_median, q_low + _FLOOR)
    q_high = np.maximum(q_high, q_median + _FLOOR)
    q_low, q_median, q_high = _break_log_symmetry(q_low, q_median, q_high)

    qpd = J_QPD_S(alpha, q_low, q_median, q_high, l=0.0)
    return QPDForecast(qpd, alpha, q_low, q_median, q_high, crossing)


def fit_qpd_chain(
    train: pd.DataFrame,
    predict_on: pd.DataFrame,
    alpha: float = 0.1,
    target: str = "sales",
    max_iter: int = 10,
) -> QPDForecast:
    """``QPD_RegressorChain``: the triplet is ordered by construction.

    Each stage is conditioned on the previous one, which is why the sub-models
    are trained at ``2 * alpha`` and ``2 * (1 - alpha) - 1`` rather than at
    ``alpha`` and ``1 - alpha``.
    """
    x_train, y = _prep(train), train[target].to_numpy(dtype=float)
    x_pred = _prep(predict_on)

    chain = QPD_RegressorChain(
        est_median=_quantile_pipeline(0.5, multiplicative=False, max_iter=max_iter),
        est_lowq=_quantile_pipeline(2 * alpha, multiplicative=False, max_iter=max_iter),
        est_highq=_quantile_pipeline(2 * (1 - alpha) - 1, multiplicative=False, max_iter=max_iter),
        bound="S",
        alpha=alpha,
        l=0.0,
    )
    chain.fit(x_train.copy(), y)
    q_low, q_median, q_high, qpd = chain.predict(x_pred.copy())
    q_low = np.asarray(q_low, dtype=float)
    q_median = np.asarray(q_median, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    return QPDForecast(qpd, alpha, q_low, q_median, q_high, 0.0)


def survival_from_qpd(forecast: QPDForecast, levels: np.ndarray) -> np.ndarray:
    """``P(D >= q)`` implied by the fitted J-QPD, for each row and each level.

    Returns shape ``(n_rows, n_levels)``.  Uses ``cdf(q - 1)`` to match the
    integer convention in :func:`srsim.simulate.survival`, and clips into
    ``[0, 1]`` because the J-QPD's closed form can overshoot slightly in the
    far tails.
    """
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    out = np.empty((len(forecast.q_median), levels.size), dtype=float)
    for j, q in enumerate(levels):
        out[:, j] = 1.0 - np.asarray(forecast.qpd.cdf(max(q - 1.0, 0.0)), dtype=float)
    return np.clip(out, 0.0, 1.0)


def inverse_transform(forecast: QPDForecast, u: np.ndarray, index=None, chunk: int = 32) -> np.ndarray:
    """Evaluate ``Q_i(u[:, i])`` row-wise -- the elementwise inverse transform.

    This exists because ``J_QPD_S.ppf`` does **not** broadcast elementwise.  Given
    ``k`` probabilities and ``m`` parameter sets it returns the full ``(k, m)``
    cross product, so ``ppf(u)`` on a ``(n_draws, m)`` array silently returns
    ``(n_draws * m, m)`` rather than the ``(n_draws, m)`` you wanted -- a bug that
    produces plausible-looking numbers.  We take the matching diagonal entries,
    in column chunks so the cross product never blows up memory.
    """
    index = np.arange(len(forecast.q_median)) if index is None else np.asarray(index)
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n_draws, n_cols = u.shape
    if n_cols != len(index):
        raise ValueError(f"u has {n_cols} columns but index selects {len(index)} rows")

    out = np.empty((n_draws, n_cols), dtype=float)
    for start in range(0, n_cols, chunk):
        stop = min(start + chunk, n_cols)
        rows = index[start:stop]
        width = stop - start
        sub = J_QPD_S(
            forecast.alpha,
            forecast.q_low[rows],
            forecast.q_median[rows],
            forecast.q_high[rows],
            l=0.0,
        )
        grid = np.asarray(sub.ppf(u[:, start:stop].ravel()), dtype=float)
        pick = np.tile(np.arange(width), n_draws)
        out[:, start:stop] = grid[np.arange(n_draws * width), pick].reshape(n_draws, width)
    return out


def sample_demand(forecast: QPDForecast, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-transform sampling from the fitted J-QPD.

    ``J_QPD_S`` exposes ``ppf`` and ``cdf`` but no ``rvs``, so sampling is
    ``Q(u)`` with ``u ~ U(0, 1)``.  Returns shape ``(n_scenarios, n_rows)``,
    rounded to integers because demand is a count.
    """
    u = rng.uniform(size=(n_scenarios, len(forecast.q_median)))
    return np.clip(np.round(inverse_transform(forecast, u)), 0.0, None)


def make_forecast(q_low, q_median, q_high, alpha: float = 0.1) -> QPDForecast:
    """Build a :class:`QPDForecast` from a quantile triplet, guarding the edge cases."""
    q_low = np.maximum(np.asarray(q_low, dtype=float), _FLOOR)
    q_median = np.maximum(np.asarray(q_median, dtype=float), q_low + _FLOOR)
    q_high = np.maximum(np.asarray(q_high, dtype=float), q_median + _FLOOR)
    q_low, q_median, q_high = _break_log_symmetry(q_low, q_median, q_high)
    return QPDForecast(
        J_QPD_S(alpha, q_low, q_median, q_high, l=0.0), alpha, q_low, q_median, q_high, 0.0
    )
