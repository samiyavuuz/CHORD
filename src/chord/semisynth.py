"""Semi-synthetic benchmark on CPTAC-dev, used to select the CHORD settings (README, "Hyperparameter Selection").

Each scenario draws a target cohort from source-dev with a chosen class mix and OOD share, then adds a known shift
delta_true = alpha * s0 * d to every slide (ID and OOD) in input space: y = x + delta_true (no re-normalisation, so
delta_true is exactly the shift to recover). s0 = mean source-train class separation |mu_S,c - mu_S|.
Directions d: random (isotropic). Every scenario has its own seed, so all methods see identical cohorts.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from chord.pipeline import Cohort
from chord.source import SourceModel

SEED = 20261027
# Class mixes (ID slides per class), sized so that rho = 0.9 stays feasible on CPTAC-dev (GBM ~70 slides, OOD ~1000).
MIXES = {"balanced": {0: 30, 1: 30, 2: 30}, "lung": {0: 45, 1: 45}, "gbm": {2: 60}, "skewed": {0: 54, 1: 18, 2: 18}}
ALPHAS = (0.0, 0.25, 0.5, 1.0)
RHOS = (0.0, 0.3, 0.6, 0.9)
DIRECTIONS = ("random",)
N_REPS = 10


@dataclass
class Scenario:
    direction: str
    alpha: float
    rho: float
    mix: str
    rep: int
    y: np.ndarray            # shifted input-space embeddings
    role: np.ndarray
    id_class: np.ndarray
    delta_true: np.ndarray   # injected shift
    diagnosis: np.ndarray


def class_separation(model: SourceModel) -> float:
    return float(np.mean(np.linalg.norm(model.class_means - model.mu, axis=1)))



def _draw_indices(rng: np.random.Generator, dev: Cohort, mix: dict, rho: float, ood_keep=None) -> np.ndarray:
    idx = [rng.choice(np.flatnonzero((dev.role == "id") & (dev.id_class == c)), n, replace=False) for c, n in mix.items()]
    n_id = sum(mix.values())
    n_ood = int(round(rho / (1 - rho) * n_id))
    ood_pool = np.flatnonzero((dev.role != "id") & (True if ood_keep is None else ood_keep))
    if n_ood > len(ood_pool):
        raise ValueError(f"rho={rho}: need {n_ood} OOD slides, only {len(ood_pool)} available")
    idx.append(rng.choice(ood_pool, n_ood, replace=False))
    return np.concatenate(idx)


def scenarios(model: SourceModel, dev: Cohort, seed: int = SEED, ood_keep=None, mixes=None, alphas=None):
    """dev: the pool to draw from (source-dev).
    ood_keep: boolean mask restricting the OOD pool.
    mixes / alphas: subsets of MIXES names / ALPHAS (default all)."""
    s0 = class_separation(model)
    grid = itertools.product(enumerate(DIRECTIONS), enumerate(ALPHAS), enumerate(RHOS), enumerate(MIXES), range(N_REPS))
    for (di, direction), (ai, alpha), (ri, rho), (mi, mix), rep in grid:
        if (mixes is not None and mix not in mixes) or (alphas is not None and alpha not in alphas):
            continue
        rng = np.random.default_rng([seed, di, ai, ri, mi, rep])
        idx = _draw_indices(rng, dev, MIXES[mix], rho, ood_keep)
        d = rng.standard_normal(dev.x.shape[1])  # direction == "random"
        delta = alpha * s0 * d / np.linalg.norm(d)
        y = dev.x[idx] + delta
        yield Scenario(direction, alpha, rho, mix, rep, y, dev.role[idx], dev.id_class[idx], delta, dev.diagnosis[idx])
