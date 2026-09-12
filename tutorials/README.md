# Tutorials

Two executable notebooks that build up the ideas in [`../PLAN.md`](../PLAN.md)
from scratch, on a simulated retail world small enough to run on a laptop.

| | | runtime |
|---|---|---|
| **[01 — Smart Predict-then-Optimize](01_pyepo_spo.ipynb)** | Decision-focused learning with [PyEPO](https://github.com/khalil-research/PyEPO). A newsvendor warm-up, then a capacitated allocation LP trained with SPO+. | ~2 min |
| **[02 — Presbitero / ZEOS](02_presbitero_zeos.ipynb)** | Simulation-assisted stochastic optimization: an extended `(R,s,Q)` policy, a discrete-event cost simulator, SHGO, and the connections back to notebook 1. | ~6 min |

Read them in order. Notebook 2 does not *depend* on notebook 1 having been run —
the expensive fitting step is shared and cached — but it assumes its ideas.

## The through-line

Both notebooks ask the same question: **how do you make a forecast serve a
decision?** They answer it in two different places.

- Notebook 1 pushes the decision into the **forecaster's loss function**. Its
  central result is that a predictor trained with the optimizer in the loop can
  make better decisions while scoring *worse* on forecast accuracy.
- Notebook 2 leaves the forecaster alone and pushes the full predictive
  **distribution into a simulation-based optimizer**. Its central result is that
  replacing that distribution with a point forecast costs ~50% in realised cost —
  the single largest effect measured in either notebook, and the same finding as
  the paper it reproduces.

Notebook 2 closes by mapping the two approaches against each other and pointing
at where they merge: a differentiable simulator, which is what `PLAN.md` §10 and
§12.1 describe.

## Running them

```bash
pip install -e ".[tutorials]"          # from the repository root
jupyter lab tutorials/
```

Everything here runs on CPU. If you do not want PyEPO's PyTorch dependency to
drag in the CUDA wheels, install torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[tutorials]"
```

Python **3.9–3.11** (cyclic-boosting pins `<3.12`). Exactly resolved versions are
in [`requirements.txt`](requirements.txt). No optimization solver is needed —
the PyEPO model is backed by `scipy.optimize.linprog`.

The first run fits three cyclic-boosting quantile models (~2 minutes) and caches
the result under `.cache/`; everything afterwards is instant. The notebooks are
committed **with their outputs**, so every number and plot in the prose is one
that actually ran.

### Sources

The notebooks are generated from the `.py` files beside them, which are the
version-controlled source:

```bash
python tools/build_notebook.py tutorials/01_pyepo_spo.py tutorials/01_pyepo_spo.ipynb
python -m nbconvert --to notebook --execute --inplace tutorials/01_pyepo_spo.ipynb
```

## The supporting package

Everything the notebooks import lives in [`../src/srsim`](../src/srsim), so the
notebooks stay readable and the machinery stays testable (`pytest` from the
repository root; 43 tests, ~4 seconds).

| module | what it is |
|---|---|
| `simulate.py` | the demand simulator, and `survival()` — the true cost vector |
| `forecast.py` | cyclic-boosting quantiles → J-QPD, with conformal calibration |
| `pipeline.py` | `build_world()`: simulate, split, fit, calibrate — cached |
| `newsvendor.py` | critical fractile, the pinball identity, exact regret |
| `allocation.py` | the tranche linearization and the PyEPO `optModel` |
| `scenarios.py` | common random numbers, `Ω_opt` / `Ω_eval` / `Ω_true` |
| `des.py` | the deterministic evaluator `C(θ, ω)` |
| `optimize.py` | SHGO driver and classical baselines |
| `experiment.py` | the per-SKU ablation, parallel across cores |
| `dfl.py` | decision-focused learning: instances, predictor, training loops |

## Credits

- **Presbitero, A. et al.** (2025), *A practical approach to replenishment
  optimization with extended (R,s,Q) policy and probabilistic models*,
  *Scientific Reports* **15**:44225.
  [doi:10.1038/s41598-025-32537-2](https://doi.org/10.1038/s41598-025-32537-2).
  ⚠️ Carries a published correction (*Sci Rep* **16**:4211, 30 Jan 2026) — some
  figures differ from the original PDF. Quote the corrected record.
- **Tang, B. & Khalil, E. B.**, *PyEPO: A PyTorch-based End-to-End
  Predict-then-Optimize Library*. https://github.com/khalil-research/PyEPO
- **Elmachtoub, A. N. & Grigas, P.** (2022), *Smart "Predict, then Optimize"*,
  *Management Science* 68(1). The SPO+ loss.
- **Wick, F. et al.**, *Cyclic Boosting* ([arXiv:2002.03425](https://arxiv.org/abs/2002.03425))
  and *Demand Forecasting of Individual Probability Density Functions with
  Machine Learning* ([arXiv:2009.07052](https://arxiv.org/abs/2009.07052)).
- **Hadlock, C. C. & Bickel, J. E.** (2017), *Johnson Quantile-Parameterized
  Distributions*, *Decision Analysis* 14(1).
- The simulator is a re-implementation of
  [FelixWick/demand_forecasting_simulation](https://github.com/FelixWick/demand_forecasting_simulation)
  (EPL-2.0), adapted to weekly cadence and to retain the ground-truth demand
  parameters that the regret benchmark needs.
