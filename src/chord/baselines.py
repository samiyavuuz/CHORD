"""Shift-correction baselines. Each returns feature-space vectors for a target cohort given in input space.

No correction, per-cohort centering, CORAL, MNN shift, and the label-informed reference (uses the true target labels;
a reference, not a competing method).
"""

import numpy as np

from chord.source import SourceModel, l2

ORGAN_OF_CLASS = {0: "lung", 1: "lung", 2: "brain"}
CLASS_OF_CLASS = {0: 0, 1: 1, 2: 2}


def none(model: SourceModel, y: np.ndarray) -> np.ndarray:
    return model.features(y)


def origin(model: SourceModel):
    """Where the source reference set is centred in feature space (the source mean). Target features of the
    centering / CORAL baselines are placed around this point so they match the reference set."""
    return model.mu


def shifted(model: SourceModel, y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Features after removing a shift delta."""
    return l2(y - delta)


def centering(model: SourceModel, y: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Per-cohort centering: the cohort mean (or `mean`, fitted on other slides) is moved onto the source mean."""
    m = y.mean(0) if mean is None else mean
    return l2(y - m + model.mu)


def _shrink(c: np.ndarray, a: float) -> np.ndarray:
    d = c.shape[0]
    return (1 - a) * np.trace(c) / d * np.eye(d) + a * c


def _sqrtm(c: np.ndarray, inverse: bool = False) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    if w.min() <= 0:
        raise ValueError("covariance not positive definite after shrinkage")
    return (v * w ** (-0.5 if inverse else 0.5)) @ v.T


def coral_fit(y: np.ndarray, source_cov: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    """(mean, A) from a cohort: centre on its own mean, map its covariance onto the source-train covariance."""
    mean = y.mean(0)
    A = _sqrtm(_shrink(np.cov((y - mean).T), a), inverse=True) @ _sqrtm(_shrink(source_cov, a))
    return mean, A


def coral_apply(y: np.ndarray, mean: np.ndarray, A: np.ndarray, at=0.0) -> np.ndarray:
    """at: origin(model), the point the re-coloured cohort is centred on."""
    return l2((y - mean) @ A + at)


def coral(model: SourceModel, y: np.ndarray, source_cov: np.ndarray, a: float) -> np.ndarray:
    """Centre the cohort on its own mean and map its covariance onto the source-train covariance (shrinkage a)."""
    return coral_apply(y, *coral_fit(y, source_cov, a), at=origin(model))


MNN_K = 20  # mnnCorrect default (Haghverdi et al., 2018)


def mnn_pairs(model: SourceModel, y: np.ndarray, k: int = MNN_K) -> tuple[np.ndarray, np.ndarray]:
    """(target index, source-train index) of the mutual k-nearest-neighbour pairs (cosine)."""
    sim = model.features(y) @ model.bank.T
    t2s = np.argpartition(-sim, k - 1, axis=1)[:, :k]          # k nearest source slides of each target slide
    s2t = np.argpartition(-sim, k - 1, axis=0)[:k]             # k nearest target slides of each source slide
    s_has = np.zeros(sim.shape, bool)
    s_has[s2t, np.arange(sim.shape[1])] = True
    j, i = np.nonzero(np.take_along_axis(s_has, t2s, axis=1))
    return j, t2s[j, i]


def mnn_shift_delta(model: SourceModel, y: np.ndarray, k: int = MNN_K) -> np.ndarray:
    """MNN-style shift: mean of (target - source) over mutual k-nearest-neighbour pairs between the target cohort and
    the source-train set, applied as one translation (class-agnostic, no target labels)."""
    j, i = mnn_pairs(model, y, k)
    if len(j) == 0:
        return np.zeros(y.shape[1])
    return (y[j] - model.x_train[i]).mean(0)


def mnn_shift(model: SourceModel, y: np.ndarray, k: int = MNN_K) -> np.ndarray:
    return shifted(model, y, mnn_shift_delta(model, y, k))


def oracle_class_shifts(model: SourceModel, y: np.ndarray, id_class: np.ndarray) -> dict:
    """Label-informed shift per present class: (delta_c, n_c), delta_c = mean of class-c ID slides - mu_S,c."""
    return {c: (y[id_class == c].mean(0) - model.class_means[c], int((id_class == c).sum()))
            for c in np.unique(id_class[id_class >= 0])}


def oracle_shared(model: SourceModel, y: np.ndarray, id_class: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One delta: count-weighted average of the class shifts (same pooling as CHORD)."""
    shifts = oracle_class_shifts(model, y, id_class)
    n = sum(k for _, k in shifts.values())
    delta = sum(d * k for d, k in shifts.values()) / n
    return model.features(y, delta), delta


def oracle_grouped(model: SourceModel, y: np.ndarray, id_class: np.ndarray, group_of_class: dict) -> np.ndarray:
    """One delta per group (organ or class); each slide gets a posterior-weighted mix of the present groups.
    Posteriors come from the features after the shared label-informed correction."""
    return shifted(model, y, oracle_grouped_delta(model, y, id_class, group_of_class))


def oracle_grouped_delta(model: SourceModel, y: np.ndarray, id_class: np.ndarray, group_of_class: dict) -> np.ndarray:
    """Per-slide label-informed shift used by oracle_grouped."""
    return oracle_apply(model, oracle_fit(model, y, id_class, group_of_class), y)


def oracle_fit(model: SourceModel, y: np.ndarray, id_class: np.ndarray, group_of_class: dict) -> dict:
    """Label-informed reference fitted on labelled slides: shared shift and one shift per present group."""
    shifts = oracle_class_shifts(model, y, id_class)
    groups = sorted({group_of_class[c] for c in shifts})
    delta_g = []
    for g in groups:
        members = [c for c in shifts if group_of_class[c] == g]
        n = sum(shifts[c][1] for c in members)
        delta_g.append(sum(shifts[c][0] * shifts[c][1] for c in members) / n)
    n = sum(k for _, k in shifts.values())
    shared = sum(d * k for d, k in shifts.values()) / n
    return {"shared": shared, "groups": groups, "delta_g": np.stack(delta_g), "group_of": group_of_class}


def oracle_apply(model: SourceModel, params: dict, y: np.ndarray) -> np.ndarray:
    """Per-slide shift for any slides: posterior (after the shared correction) mix of the fitted group shifts."""
    proba = model.clf.predict_proba(model.features(y, params["shared"]))
    g_of = params["group_of"]
    w = np.stack([proba[:, [c for c in model.classes if g_of[c] == g]].sum(1) for g in params["groups"]], 1)
    w = w / w.sum(1, keepdims=True)
    return w @ params["delta_g"]
