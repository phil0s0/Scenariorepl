"""A capacitated allocation problem that PyEPO can actually learn on.

Getting inventory into PyEPO's frame takes one non-obvious step, and hiding it
would teach the wrong lesson.  PyEPO/SPO+ assumes

    min_w  c(x)^T w   subject to  w in S,

with uncertainty in the **objective coefficients** and a **fixed** feasible set.
Inventory resists that twice over:

1.  Demand naturally enters the *constraints* (inventory balance), not the
    objective.  That rules out multi-period lot-sizing outright -- regret is not
    even well-defined when the feasible set moves with the uncertainty.
2.  The value of stock is *concave* in quantity: each extra unit is less likely
    to sell than the last.  PyEPO needs linearity.

The fix for (2) is to linearise the concave value on a grid of **tranches**.
Stock level ``q`` for item ``j`` is built up out of ``B`` blocks, and the value
of block ``b`` is the average marginal value of the units inside it:

    v[j, b] = (margin + holding) * s[j, b] - holding,
    s[j, b] = mean over units q in block b of  P(D_j >= q).

Two things fall out of this, and they are why the tutorial is built this way:

*   **The cost vector is an affine transform of a survival function.**  That is
    exactly what a J-QPD predicts, so part B predicts the same object as part A
    rather than changing the subject.
*   **The uncoupled optimum is the newsvendor.**  Drop the capacity constraints
    and the LP fills blocks while ``v > 0``, i.e. while
    ``P(D >= q) >= holding / (margin + holding)`` -- the critical fractile with
    ``c_u = margin``, ``c_o = holding``.  Part A is the special case.

The LP is

    max_x  sum_{j,b} v[j, b] x[j, b]
    s.t.   sum_{j,b} x[j, b] <= total_capacity            (one shared container)
           sum_{j in store s, b} x[j, b] <= store_cap[s]  (shelf space per store)
           0 <= x[j, b] <= width[b]

whose constraint matrix is a laminar family (items nested in stores, plus the
grand total) and therefore totally unimodular, so the LP relaxation already has
integral vertices for integer capacities and widths.  We assert that rather than
trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyepo import EPO
from pyepo.model.opt import optModel
from scipy.optimize import linprog

__all__ = [
    "TrancheSpec",
    "TrancheAllocationModel",
    "marginal_values",
    "true_survival_matrix",
    "forecast_survival_matrix",
]


@dataclass(frozen=True)
class TrancheSpec:
    """The stock-level grid the concave value is linearised on.

    ``width`` units per block, ``n_blocks`` blocks, so the grid reaches
    ``width * n_blocks`` units.  Block ``b`` covers unit levels
    ``b*width + 1 ... (b+1)*width``.
    """

    width: int = 4
    n_blocks: int = 5

    @property
    def unit_levels(self) -> np.ndarray:
        """Every unit level on the grid, shape ``(n_blocks, width)``."""
        return (np.arange(self.n_blocks * self.width) + 1).reshape(self.n_blocks, self.width)

    @property
    def widths(self) -> np.ndarray:
        return np.full(self.n_blocks, float(self.width))

    @property
    def max_units(self) -> int:
        return self.width * self.n_blocks


def marginal_values(survival: np.ndarray, margin: np.ndarray, holding: np.ndarray) -> np.ndarray:
    """``v = (margin + holding) * s - holding``.

    ``survival`` is ``(n_items, n_blocks)``; ``margin`` and ``holding`` are
    per-item, shape ``(n_items,)``.  Positive where the block is worth stocking.
    """
    margin = np.asarray(margin, dtype=float)[:, None]
    holding = np.asarray(holding, dtype=float)[:, None]
    return (margin + holding) * np.asarray(survival, dtype=float) - holding


def true_survival_matrix(lam: np.ndarray, dispersion: np.ndarray, spec: TrancheSpec) -> np.ndarray:
    """Block-averaged ``P(D >= q)`` under the true NB.  Shape ``(n_items, n_blocks)``."""
    from .simulate import survival

    levels = spec.unit_levels  # (n_blocks, width)
    flat = levels.ravel()[None, :]  # (1, n_blocks*width)
    s = survival(flat, np.asarray(lam)[:, None], np.asarray(dispersion)[:, None])
    return s.reshape(len(lam), spec.n_blocks, spec.width).mean(axis=2)


def forecast_survival_matrix(forecast, spec: TrancheSpec, index=None) -> np.ndarray:
    """Block-averaged ``P(D >= q)`` implied by a fitted J-QPD."""
    from .forecast import survival_from_qpd

    s = survival_from_qpd(forecast, spec.unit_levels.ravel().astype(float))
    if index is not None:
        s = s[index]
    n_items = s.shape[0]
    return s.reshape(n_items, spec.n_blocks, spec.width).mean(axis=2)


class TrancheAllocationModel(optModel):
    """PyEPO model for the capacitated tranche allocation LP.

    Deliberately backed by ``scipy.optimize.linprog`` rather than a commercial
    solver: PyEPO hard-depends on no solver at all, and ``linprog`` is a
    *stateless function*, so there is no live solver object to trip over PyEPO
    2.x's constructor-argument snapshotting.

    Two implementation details that are easy to get wrong and silent when you do:

    *   ``solve`` must return a **scalar** objective.  A shape-``(1,)`` value
        broadcasts to ``(batch, batch)`` inside ``SPOPlusFunc.forward`` and still
        reduces to a scalar loss -- wrong, with no error.
    *   ``modelSense`` is a **class attribute**, so it stays out of the captured
        init config, and the objective is returned in *maximisation* sense
        (``c @ sol``), not as ``-res.fun``.
    """

    modelSense = EPO.MAXIMIZE

    def __init__(self, store_of_item, spec_width, spec_blocks, store_caps, total_capacity):
        # Plain, deep-copyable, dill-safe values only.
        self.store_of_item = np.asarray(store_of_item, dtype=int)
        self.spec_width = int(spec_width)
        self.spec_blocks = int(spec_blocks)
        self.store_caps = np.asarray(store_caps, dtype=float)
        self.total_capacity = float(total_capacity)
        self.n_items = len(self.store_of_item)
        self._c = None
        super().__init__()  # must be last: the base __init__ calls _getModel()

    def _getModel(self):
        n_items, n_blocks = self.n_items, self.spec_blocks
        n_var = n_items * n_blocks

        rows = [np.ones(n_var)]
        rhs = [self.total_capacity]
        for store, cap in enumerate(self.store_caps):
            row = np.zeros(n_var)
            members = np.where(self.store_of_item == store)[0]
            for j in members:
                row[j * n_blocks : (j + 1) * n_blocks] = 1.0
            rows.append(row)
            rhs.append(float(cap))

        upper = np.tile(np.full(n_blocks, float(self.spec_width)), n_items)
        model = {
            "A_ub": np.vstack(rows),
            "b_ub": np.asarray(rhs, dtype=float),
            "bounds": np.stack([np.zeros(n_var), upper], axis=1),
        }
        return model, list(range(n_var))

    def setObj(self, c) -> None:
        c = np.asarray(c, dtype=float).ravel()
        if c.size != self.num_cost:
            raise ValueError(f"expected {self.num_cost} coefficients, got {c.size}")
        self._c = c

    def solve(self):
        if self._c is None:
            raise RuntimeError("setObj must be called before solve")
        res = linprog(
            c=-self._c,  # linprog minimises; we maximise value
            A_ub=self._model["A_ub"],
            b_ub=self._model["b_ub"],
            bounds=self._model["bounds"],
            method="highs",
        )
        if not res.success:  # x = 0 is always feasible, so this is an assertion
            raise RuntimeError(f"LP failed: {res.message}")
        sol = np.asarray(res.x, dtype=float)
        return sol, float(self._c @ sol)
