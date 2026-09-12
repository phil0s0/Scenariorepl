"""Decision-focused learning on the tranche allocation problem.

Three arms, all solving the *same* LP with the *same* capacities, differing only
in where the cost vector comes from:

``plug-in``    read the survival curve straight off the fitted J-QPD.  No learning
               at all -- this is classical predict-then-optimize.
``two-stage``  train a predictor to match the true cost vector by MSE, then
               optimize.  The textbook "fit accurately, then plug in".
``SPO+``       train the *same* predictor with the optimizer in the loop, using
               the Smart Predict-then-Optimize surrogate loss.

The two trained arms share an architecture, so the comparison isolates the loss.

A note on honesty, because it is easy to rig this experiment: we deliberately use
a **linear** head against a demand process that is exponential in its features.
That misspecification is the regime where decision-focused learning is supposed
to help -- a predictor that cannot be right everywhere should at least be right
where the decision is sensitive.  With a well-specified predictor the picture
changes, and often reverses: two-stage MSE is asymptotically optimal and SPO+ is a
surrogate.  The notebook runs both and reports what actually happens.

The predictor is monotone by construction.  The LP is only well-posed -- "fill the
cheap blocks first" -- if marginal value decreases with stock level, so the
network predicts non-negative decrements and accumulates them, rather than
predicting ``B`` free numbers and hoping.  That is the decision structure
dictating the model architecture, which is the whole idea in miniature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .allocation import TrancheSpec, marginal_values, true_survival_matrix

__all__ = [
    "InstanceSet",
    "FEATURE_NAMES",
    "build_instances",
    "TrancheNet",
    "train_two_stage",
    "train_spo",
]

#: Per-item features.  ``margin`` and ``holding`` are also read directly by the
#: output head, which is why their positions are pinned.
FEATURE_NAMES = [
    "log_hist_mean",
    "price_ratio",
    "promo",
    "woy_sin",
    "woy_cos",
    "log_age",
    "margin",
    "holding",
]
_MARGIN = FEATURE_NAMES.index("margin")
_HOLDING = FEATURE_NAMES.index("holding")


@dataclass
class InstanceSet:
    """A batch of allocation instances sharing one feasible set."""

    feats: np.ndarray          # (n_instances, n_items * n_features)
    costs_true: np.ndarray     # (n_instances, n_items * n_blocks)
    costs_plugin: np.ndarray   # (n_instances, n_items * n_blocks)
    store_of_item: np.ndarray  # (n_items,)
    spec: TrancheSpec
    n_items: int

    @property
    def n_features(self) -> int:
        return len(FEATURE_NAMES)


def _hist_mean(train) -> dict:
    grouped = train.groupby(["P_ID", "L_ID"])["sales"].mean()
    return grouped.to_dict()


def build_instances(
    world,
    spec: TrancheSpec,
    n_instances: int,
    n_items: int,
    rng: np.random.Generator,
    frame: str = "test",
) -> InstanceSet:
    """Assemble allocation instances by sampling baskets of SKUs per week.

    Slot ``j`` is permanently assigned to store ``j % n_locations``, because
    PyEPO holds **one** ``optModel`` instance for a whole dataset -- the feasible
    set cannot vary from instance to instance.  Each instance then fills slot
    ``j`` with a SKU drawn from that slot's store in the chosen week.
    """
    data = getattr(world, frame)
    hist = _hist_mean(world.train)
    n_stores = int(world.panel["L_ID"].max()) + 1
    store_of_item = np.arange(n_items) % n_stores

    weeks = np.sort(data["week"].unique())
    by_week_store: dict = {}
    for week in weeks:
        for store in range(n_stores):
            idx = np.where(
                (data["week"].to_numpy() == week) & (data["L_ID"].to_numpy() == store)
            )[0]
            if len(idx):
                by_week_store[(week, store)] = idx

    usable = [w for w in weeks if all((w, s) in by_week_store for s in range(n_stores))]
    if not usable:
        raise ValueError("no week has SKUs in every store")

    forecast = world.forecast
    plugin_all = _plugin_survival(forecast, spec)

    feats = np.zeros((n_instances, n_items * len(FEATURE_NAMES)), dtype=np.float32)
    costs_true = np.zeros((n_instances, n_items * spec.n_blocks), dtype=np.float32)
    costs_plugin = np.zeros_like(costs_true)

    lam = data["lam"].to_numpy()
    dispersion = data["dispersion"].to_numpy()
    margin = data["margin"].to_numpy()
    holding = data["holding"].to_numpy()
    price_ratio = data["price_ratio"].to_numpy()
    promo = data["promo"].to_numpy()
    woy = data["week_of_year"].to_numpy()
    age = data["age"].to_numpy()
    pid = data["P_ID"].to_numpy()
    lid = data["L_ID"].to_numpy()

    for i in range(n_instances):
        week = usable[rng.integers(len(usable))]
        rows = np.array(
            [rng.choice(by_week_store[(week, s)]) for s in store_of_item], dtype=int
        )
        block = np.stack(
            [
                np.log1p([hist.get((pid[r], lid[r]), 0.0) for r in rows]),
                price_ratio[rows],
                promo[rows].astype(float),
                np.sin(2 * np.pi * woy[rows] / 52.0),
                np.cos(2 * np.pi * woy[rows] / 52.0),
                np.log1p(age[rows]),
                margin[rows],
                holding[rows],
            ],
            axis=1,
        )
        feats[i] = block.astype(np.float32).ravel()
        surv_true = true_survival_matrix(lam[rows], dispersion[rows], spec)
        costs_true[i] = marginal_values(surv_true, margin[rows], holding[rows]).ravel()
        costs_plugin[i] = marginal_values(plugin_all[rows], margin[rows], holding[rows]).ravel()

    return InstanceSet(feats, costs_true, costs_plugin, store_of_item, spec, n_items)


def _plugin_survival(forecast, spec: TrancheSpec) -> np.ndarray:
    from .forecast import survival_from_qpd

    s = survival_from_qpd(forecast, spec.unit_levels.ravel().astype(float))
    return s.reshape(s.shape[0], spec.n_blocks, spec.width).mean(axis=2)


class TrancheNet(nn.Module):
    """Per-item predictor of the block marginal-value vector.

    The same parameters are applied to every item -- slots are interchangeable --
    and the output is forced to be decreasing in the block index:

        u = softplus(raw) >= 0,   s = exp(-cumsum(u)) in (0, 1] and decreasing,
        v = (margin + holding) * s - holding.

    ``s`` is a survival curve by construction, so the network cannot predict
    something the LP would misinterpret.
    """

    def __init__(self, n_items: int, n_blocks: int, hidden: int = 0, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_items = n_items
        self.n_blocks = n_blocks
        self.n_features = len(FEATURE_NAMES)
        if hidden:
            self.head = nn.Sequential(
                nn.Linear(self.n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_blocks)
            )
        else:
            self.head = nn.Linear(self.n_features, n_blocks)
        self.register_buffer("feat_mean", torch.zeros(self.n_features))
        self.register_buffer("feat_std", torch.ones(self.n_features))

    def fit_scaler(self, feats: np.ndarray) -> None:
        flat = torch.as_tensor(feats, dtype=torch.float32).view(-1, self.n_features)
        self.feat_mean.copy_(flat.mean(dim=0))
        self.feat_std.copy_(flat.std(dim=0).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyEPO's metrics hand over a single un-batched feature row, while the
        # training loop hands over batches; accept either.
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        items = x.view(batch, self.n_items, self.n_features)
        margin = items[..., _MARGIN]
        holding = items[..., _HOLDING]
        scaled = (items - self.feat_mean) / self.feat_std
        raw = self.head(scaled)
        decrements = torch.nn.functional.softplus(raw)
        survival = torch.exp(-torch.cumsum(decrements, dim=-1))
        value = (margin + holding).unsqueeze(-1) * survival - holding.unsqueeze(-1)
        out = value.reshape(batch, self.n_items * self.n_blocks)
        return out.squeeze(0) if squeeze else out


def _loader(dataset, batch_size: int, shuffle: bool):
    from pyepo.data.dataset import optDataLoader

    return optDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_two_stage(
    model: TrancheNet,
    dataset,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-2,
) -> list[float]:
    """Fit the predictor by MSE against the true cost vector, ignoring the LP."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = _loader(dataset, batch_size, True)
    history = []
    for _ in range(epochs):
        running = 0.0
        for x, c, _w, _z in loader:
            optimizer.zero_grad()
            loss = ((model(x) - c) ** 2).mean()
            loss.backward()
            optimizer.step()
            running += float(loss)
        history.append(running / max(len(loader), 1))
    return history


def train_spo(
    model: TrancheNet,
    dataset,
    optmodel,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-2,
) -> list[float]:
    """Fit the predictor with SPO+, which puts the LP inside the loss."""
    from pyepo.func import SPOPlus

    criterion = SPOPlus(optmodel, processes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = _loader(dataset, batch_size, True)
    history = []
    for _ in range(epochs):
        running = 0.0
        for x, c, w, z in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), c, w, z)
            loss.backward()
            optimizer.step()
            running += float(loss)
        history.append(running / max(len(loader), 1))
    return history
