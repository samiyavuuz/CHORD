"""Shared entry points: the calibrated source model (CPTAC) and the evaluation cohorts (input space).
Source splits are named "CPTAC-train/val/dev/test"; external cohorts by their name."""

from dataclasses import dataclass

import numpy as np

from chord.data import SOURCE, load_embeddings, read_manifest_with_splits
from chord.source import SourceModel, to_input_space

EXTERNAL = ("TCGA", "PLCO", "EBRAINS")  # target cohorts
COHORTS = (f"{SOURCE}-test", *EXTERNAL)  # CPTAC-test: same-cohort control
# Index of each cohort's random-number stream (bootstrap resamples, held-out splits), fixed per cohort so that the
# random draws do not depend on the order or number of cohorts.
SEED_INDEX = {f"{SOURCE}-test": 0, "TCGA": 1, "PLCO": 3, "EBRAINS": 4}
LOOKALIKE = "HNSCC"  # head-and-neck squamous near-OOD class of CPTAC (failure analysis)


def src(split: str) -> str:
    return f"{SOURCE}-{split}"


@dataclass
class Cohort:
    name: str
    x: np.ndarray          # input space, l2(TITAN)
    role: np.ndarray       # id / near / far
    id_class: np.ndarray   # 0..2 for ID, -1 otherwise (evaluation only)
    diagnosis: np.ndarray
    patient_id: np.ndarray


def _active_manifest():
    m = read_manifest_with_splits()
    return m[m["role"] != "excluded"].reset_index(drop=True)


def load_source_model() -> SourceModel:
    """Fit on source-train ID slides, calibrate the gate on source-val ID slides."""
    m = _active_manifest()
    t = m[m["cohort"] == SOURCE]
    tr, va = t[t["split"] == "train"], t[t["split"] == "val"]
    assert (tr["role"] == "id").all() and (va["role"] == "id").all()
    model = SourceModel(to_input_space(load_embeddings(tr)), tr["id_class"].to_numpy())
    model.calibrate(to_input_space(load_embeddings(va)))
    return model


def load_cohort(name: str) -> Cohort:
    m = _active_manifest()
    if name in (src("train"), src("val"), src("dev"), src("test")):
        sel = (m["cohort"] == SOURCE) & (m["split"] == name.split("-")[1])
    elif name == "PLCO":  # H&E slides only; the IHC-stained slides are not used
        sel = (m["cohort"] == "PLCO") & (m["diagnosis"] != "IHC")
    elif name in EXTERNAL:
        sel = m["cohort"] == name
    else:  # includes the whole source cohort, which would leak training slides
        raise ValueError(name)
    c = m[sel]
    return Cohort(name, to_input_space(load_embeddings(c)), c["role"].to_numpy(), c["id_class"].to_numpy(),
                  c["diagnosis"].to_numpy(), c["patient_id"].to_numpy())
