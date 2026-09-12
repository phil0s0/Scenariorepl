"""The shared, cached "build the world" step.

Both notebooks need the same thing: a simulated panel, a temporal split, and a
calibrated J-QPD forecast for the evaluation weeks.  Fitting that costs a couple
of minutes, so it is cached on disk with ``joblib``.

The point of putting it here rather than in a notebook is that **notebook 2 must
not depend on notebook 1 having been run**.  Both call :func:`build_world` with
the same config and get the same object; whichever runs first pays for the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory

from .forecast import QPDForecast, conformalize, fit_qpd_independent
from .simulate import PanelConfig, simulate_panel

__all__ = ["WorldConfig", "World", "build_world", "CACHE_DIR"]

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
_memory = Memory(location=str(CACHE_DIR), verbose=0)


@dataclass(frozen=True)
class WorldConfig:
    """Everything that determines the simulated world and its forecast."""

    n_products: int = 60
    n_locations: int = 3
    n_weeks: int = 104
    seed: int = 20250101
    #: Weeks [0, train_weeks) fit the model.
    train_weeks: int = 70
    #: Weeks [train_weeks, train_weeks + calib_weeks) calibrate the intervals.
    calib_weeks: int = 8
    #: Symmetric-percentile triplet level: the J-QPD is pinned at alpha, 0.5, 1-alpha.
    alpha: float = 0.1
    max_iter: int = 6

    @property
    def panel(self) -> PanelConfig:
        return PanelConfig(
            n_products=self.n_products,
            n_locations=self.n_locations,
            n_weeks=self.n_weeks,
            seed=self.seed,
        )

    @property
    def eval_start(self) -> int:
        return self.train_weeks + self.calib_weeks

    def as_key(self) -> tuple:
        return (
            self.n_products,
            self.n_locations,
            self.n_weeks,
            self.seed,
            self.train_weeks,
            self.calib_weeks,
            self.alpha,
            self.max_iter,
        )


@dataclass
class World:
    """A simulated panel plus a calibrated predictive distribution over its tail."""

    config: WorldConfig
    panel: pd.DataFrame
    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    #: Raw (un-conformalized) forecast on the evaluation rows.
    raw: QPDForecast
    #: Conformalized forecast on the evaluation rows -- the one to use.
    forecast: QPDForecast
    #: Diagnostics recorded at fit time.
    diagnostics: dict = field(default_factory=dict)


def _interval_coverage(fc: QPDForecast, y: np.ndarray) -> float:
    return float(np.mean((y >= fc.q_low) & (y <= fc.q_high)))


@_memory.cache(ignore=["verbose"])
def _build(key: tuple, verbose: bool = False) -> dict:
    cfg = WorldConfig(
        n_products=key[0],
        n_locations=key[1],
        n_weeks=key[2],
        seed=key[3],
        train_weeks=key[4],
        calib_weeks=key[5],
        alpha=key[6],
        max_iter=key[7],
    )
    panel = simulate_panel(cfg.panel)
    train = panel[panel.week < cfg.train_weeks].reset_index(drop=True)
    calib = panel[
        (panel.week >= cfg.train_weeks) & (panel.week < cfg.eval_start)
    ].reset_index(drop=True)
    test = panel[panel.week >= cfg.eval_start].reset_index(drop=True)

    # One fit, two prediction sets: the calibration rows set the conformal
    # padding, the evaluation rows are what everything downstream consumes.
    both = pd.concat([calib, test], ignore_index=True)
    fc_both = fit_qpd_independent(
        train, both, alpha=cfg.alpha, max_iter=cfg.max_iter
    )
    n_calib = len(calib)

    def _slice(fc: QPDForecast, lo: int, hi: int) -> QPDForecast:
        from cyclic_boosting.quantile_matching import J_QPD_S

        ql, qm, qh = fc.q_low[lo:hi], fc.q_median[lo:hi], fc.q_high[lo:hi]
        return QPDForecast(
            J_QPD_S(fc.alpha, ql, qm, qh, l=0.0), fc.alpha, ql, qm, qh, fc.crossing_rate
        )

    fc_calib = _slice(fc_both, 0, n_calib)
    fc_test = _slice(fc_both, n_calib, len(both))
    fc_conf = conformalize(fc_test, fc_calib, calib["sales"].to_numpy(dtype=float))

    diagnostics = {
        "crossing_rate": fc_both.crossing_rate,
        "coverage_raw": _interval_coverage(fc_test, test["sales"].to_numpy(dtype=float)),
        "coverage_conformal": _interval_coverage(
            fc_conf, test["sales"].to_numpy(dtype=float)
        ),
        "nominal_coverage": 1 - 2 * cfg.alpha,
        "conformal_pad": float(np.mean(fc_conf.q_high - fc_test.q_high)),
        "wape": float(
            np.abs(fc_test.q_median - test["sales"].to_numpy()).sum()
            / test["sales"].to_numpy().sum()
        ),
        # The WAPE an oracle would score, knowing the true intensity exactly.
        # Anything at or near this floor is irreducible observation noise, not
        # model error -- which is worth knowing before reading too much into a
        # WAPE number, or into a correlation between WAPE and business uplift.
        # Guard: a degenerate triplet makes J_QPD_S return NaN silently, which
        # would only surface much later as a NaN cost in the optimizer.
        "finite_triplet": bool(
            np.all(np.isfinite(fc_conf.q_low))
            and np.all(fc_conf.q_low > 0)
            and np.all(fc_conf.q_median > fc_conf.q_low)
            and np.all(fc_conf.q_high > fc_conf.q_median)
        ),
        "wape_oracle": float(
            np.abs(test["lam"].to_numpy() - test["sales"].to_numpy()).sum()
            / test["sales"].to_numpy().sum()
        ),
    }
    return {
        "panel": panel,
        "train": train,
        "calib": calib,
        "test": test,
        "raw_triplet": (fc_test.q_low, fc_test.q_median, fc_test.q_high),
        "conf_triplet": (fc_conf.q_low, fc_conf.q_median, fc_conf.q_high),
        "crossing_rate": fc_both.crossing_rate,
        "diagnostics": diagnostics,
    }


def build_world(config: WorldConfig | None = None, verbose: bool = True, **overrides) -> World:
    """Simulate, split, fit and calibrate.  Cached on disk by config."""
    config = config or WorldConfig()
    if overrides:
        config = replace(config, **overrides)
    from cyclic_boosting.quantile_matching import J_QPD_S

    payload = _build(config.as_key(), verbose=verbose)

    def _mk(triplet) -> QPDForecast:
        ql, qm, qh = triplet
        return QPDForecast(
            J_QPD_S(config.alpha, ql, qm, qh, l=0.0),
            config.alpha,
            ql,
            qm,
            qh,
            payload["crossing_rate"],
        )

    return World(
        config=config,
        panel=payload["panel"],
        train=payload["train"],
        calib=payload["calib"],
        test=payload["test"],
        raw=_mk(payload["raw_triplet"]),
        forecast=_mk(payload["conf_triplet"]),
        diagnostics=payload["diagnostics"],
    )
