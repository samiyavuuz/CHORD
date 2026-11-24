"""CHORD (paper, Section 3): class-aware, target-label-free estimate of the cohort shift.

One pass, given the current shift estimate delta (input space; one vector, or one per slide):
  1. u_j = l2(y_j - delta_j)
  2. yhat_j = h(u_j)                                         (source classifier)
  3. C_c = {j : yhat_j = c and max_i sim(u_j, bank_i) >= g}  (bank restricted to source-train class c)
     class c is present only if |C_c| >= n_min
  4. S_c = top max(n_min, ceil(r |C_c|)) of C_c by mean top-k similarity to source-train class c
  5. delta_c from the raw y over S_c:
       class_mean: mean_{j in S_c} y_j - mu_S,c
       matched:    mean_{j in S_c} (y_j - mean of j's k nearest source-train class-c slides, input space)
  6. combine:
       shared: delta = sum_c |S_c| delta_c / sum_c |S_c|                     (one vector for all slides)
       organ / class: delta_g = count-weighted delta_c within group g; slide j gets
                      sum_g w_jg delta_g over present groups, w_jg = classifier posterior mass of g at u_j
                      (from step 1 of the same pass), renormalised over present groups
Passes start from delta = 0; T passes reuse the previous delta for steps 1-4. If no class is present, delta is left
unchanged. The paper's setting (SETTINGS) is one pass with the shared combination; the other options are the
variants and ablations of the paper's Table 3.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from chord.source import KNN_K, SourceModel

# CHORD setting of the paper, selected on the CPTAC-dev semi-synthetic benchmark (README, "Hyperparameter Selection"):
# estimator, n_min, r (run_method_dev.py), T (run_iterations_dev.py), grouping (run_variants_dev.py).
SETTINGS = {"estimator": "matched", "n_min": 20, "r": 0.10, "T": 1, "grouping": "shared"}


def frozen(**overrides) -> dict:
    """estimate_shift keyword arguments of the paper's setting, with optional overrides for ablations."""
    return {**SETTINGS, **overrides}


ESTIMATORS = ("class_mean", "matched")
RANKS = ("knn", "random")  # ranking of C_c for S_c; "knn" is the paper's setting, "random" an ablation
GROUPINGS = {"shared": {0: "all", 1: "all", 2: "all"},
             "organ": {0: "lung", 1: "lung", 2: "brain"},
             "class": {0: "LUAD", 1: "LUSC", 2: "GBM"}}
# Result labels of CHORD and of the label-informed reference with the same grouping
GROUP = SETTINGS["grouping"]
OURS, REFERENCE = f"ours {GROUP}", f"V2 reference {GROUP}"


@dataclass
class ShiftEstimate:
    delta: np.ndarray                               # (d,) for shared, (n, d) for organ / class
    selected: dict = field(default_factory=dict)    # class -> indices of S_c (last pass)
    candidates: dict = field(default_factory=dict)  # class -> |C_c| (last pass)
    history: list = field(default_factory=list)     # delta after every pass
    selections: list = field(default_factory=list)  # S_c of every pass
    passes: list = field(default_factory=list)      # per pass: combine parameters (None if no class present)


def _class_shifts(model, y, u, yhat, n_min, r, estimator, k, presence=True, rank="knn", rng=None, gate=True):
    shifts, selected, candidates = {}, {}, {}
    for c in model.classes:
        cand = np.flatnonzero(yhat == c)
        if len(cand) == 0:
            candidates[c] = 0
            continue
        sim = model.similarity(u[cand], cls=c)
        passing = sim.max(1) >= model.gate if gate else np.ones(len(cand), bool)
        candidates[c] = int(passing.sum())
        if candidates[c] < (n_min if presence else 1):
            continue
        cand, sim = cand[passing], sim[passing]
        n_sel = min(len(cand), max(n_min, math.ceil(r * len(cand))))
        if rank == "knn":
            key = SourceModel.topk_mean(sim, k)
        else:
            key = rng.random(len(cand))
        order = np.argsort(-key, kind="stable")[:n_sel]
        chosen = cand[order]
        if estimator == "class_mean":
            per_slide = y[chosen] - model.class_means[c]
        else:
            x_c = model.x_train[model.bank_labels == c]
            nbrs = np.argpartition(-sim[order], min(k, sim.shape[1]) - 1, axis=1)[:, :k]
            per_slide = y[chosen] - x_c[nbrs].mean(1)
        shifts[c] = (per_slide.mean(0), len(chosen))
        selected[c] = chosen
    return shifts, selected, candidates


def _weighted(shifts: dict, members) -> np.ndarray:
    n = sum(shifts[c][1] for c in members)
    return sum(shifts[c][0] * shifts[c][1] for c in members) / n


def one_pass(model: SourceModel, y: np.ndarray, delta: np.ndarray, n_min: int, r: float, estimator: str,
             grouping: str = "shared", k: int = KNN_K, presence: bool = True,
             rank: str = "knn", rng=None, gate: bool = True) -> tuple[np.ndarray, dict, dict, dict | None]:
    if estimator not in ESTIMATORS or rank not in RANKS:
        raise ValueError((estimator, rank))
    group_of = GROUPINGS[grouping]
    u = model.features(y, delta)
    yhat = model.predict(u)
    shifts, selected, candidates = _class_shifts(model, y, u, yhat, n_min, r, estimator, k, presence, rank, rng, gate)
    if not shifts:
        return delta, selected, candidates, None
    if grouping == "shared":
        d = _weighted(shifts, shifts)
        return d, selected, candidates, {"grouping": "shared", "delta": d}
    groups = sorted({group_of[c] for c in shifts})
    delta_g = np.stack([_weighted(shifts, [c for c in shifts if group_of[c] == g]) for g in groups])
    params = {"grouping": grouping, "groups": groups, "delta_g": delta_g}
    return _mix(model, u, params), selected, candidates, params


def _mix(model: SourceModel, u: np.ndarray, params: dict) -> np.ndarray:
    """Per-slide delta: posterior-weighted mix of the present groups' shifts."""
    group_of = GROUPINGS[params["grouping"]]
    proba = model.clf.predict_proba(u)
    w = np.stack([proba[:, [c for c in model.classes if group_of[c] == g]].sum(1) for g in params["groups"]], 1)
    total = w.sum(1, keepdims=True)
    w = np.where(total > 0, w / np.where(total > 0, total, 1), 1 / len(params["groups"]))
    return w @ params["delta_g"]


def estimate_shift(model: SourceModel, y: np.ndarray, n_min: int, r: float, estimator: str = "class_mean",
                   T: int = 1, grouping: str = "shared", presence: bool = True,
                   k: int = KNN_K, rank: str = "knn", seed: int = 0, gate: bool = True) -> ShiftEstimate:
    """Estimate the cohort shift of unlabelled input-space slides y (paper, Section 3).

    k: neighbours for ranking and matching. Ablations (paper, Table 3): gate=False keeps every slide predicted as c as
    a candidate (no similarity gate); presence=False drops the count threshold (a class contributes if it has any
    candidate); rank="random" (with seed) replaces the kNN ranking."""
    rng = np.random.default_rng(seed)
    delta = np.zeros(y.shape[1])
    est = ShiftEstimate(delta)
    for _ in range(T):
        delta, est.selected, est.candidates, params = one_pass(model, y, delta, n_min, r, estimator, grouping, k=k,
                                                               presence=presence, rank=rank, rng=rng, gate=gate)
        est.history.append(delta)
        est.selections.append(est.selected)
        est.passes.append(params)
    est.delta = delta
    return est


def apply_shift(model: SourceModel, est: ShiftEstimate, y_new: np.ndarray) -> np.ndarray:
    """Replay the learned passes on new slides (held-out evaluation). Reproduces est.delta on the adaptation slides."""
    delta = np.zeros(y_new.shape[1])
    for params in est.passes:
        if params is None:
            continue
        if params["grouping"] == "shared":
            delta = params["delta"]
        else:
            delta = _mix(model, model.features(y_new, delta), params)
    return delta
