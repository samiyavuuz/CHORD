"""Patient-level CPTAC splits: ID classes -> train/val/dev/test (50/15/15/20 %), OOD cancers -> dev/test (50/50).

Patients are shuffled within each diagnosis with one fixed seed. Writes data/splits.csv (cohort, slide_id, split).
"""

import numpy as np
import pandas as pd

from chord.data import REPO_ROOT, SOURCE, SPLITS_PATH, read_manifest

SEED = 20261026
ID_FRACTIONS = {"train": 0.50, "val": 0.15, "dev": 0.15, "test": 0.20}
OOD_FRACTIONS = {"dev": 0.50, "test": 0.50}
SPLIT_ORDER = ["train", "val", "dev", "test"]


def assign(patients: np.ndarray, fractions: dict, rng: np.random.Generator) -> dict:
    """Shuffle patients and cut at rounded cumulative fractions."""
    order = rng.permutation(np.sort(patients))
    bounds = np.round(np.cumsum(list(fractions.values())) * len(order)).astype(int)
    bounds[-1] = len(order)
    out, start = {}, 0
    for name, stop in zip(fractions, bounds):
        out.update({p: name for p in order[start:stop]})
        start = stop
    return out


def main():
    m = read_manifest()
    tcga = m[(m["cohort"] == SOURCE) & (m["role"] != "excluded")].copy()
    if (tcga.groupby("patient_id")["diagnosis"].nunique() > 1).any():
        raise ValueError(f"a {SOURCE} patient has more than one diagnosis; splits would leak")

    rng = np.random.default_rng(SEED)
    patient_split = {}
    for dx in sorted(tcga["diagnosis"].unique()):
        g = tcga[tcga["diagnosis"] == dx]
        role = g["role"].iloc[0]
        if g["role"].nunique() != 1:
            raise ValueError(f"{dx}: mixed roles")
        fractions = ID_FRACTIONS if role == "id" else OOD_FRACTIONS
        patient_split.update(assign(g["patient_id"].unique(), fractions, rng))

    tcga["split"] = tcga["patient_id"].map(patient_split)
    assert tcga["split"].notna().all(), "slide without a split"
    assert (tcga.groupby("patient_id")["split"].nunique() == 1).all(), "patient in more than one split"
    assert set(tcga.loc[tcga["role"] != "id", "split"]) <= set(OOD_FRACTIONS), "OOD slide in train/val"

    tcga[["cohort", "slide_id", "split"]].to_csv(SPLITS_PATH, index=False)

    cols = [s for s in SPLIT_ORDER if s in set(tcga["split"])]
    slides = pd.crosstab([tcga["role"], tcga["diagnosis"]], tcga["split"])[cols]
    patients = tcga.groupby(["role", "diagnosis", "split"])["patient_id"].nunique().unstack(fill_value=0)[cols]
    print("== Slides per split\n" + slides.to_string())
    print("\n== Patients per split\n" + patients.to_string())
    print("\n== Patient fraction per split (ID classes)")
    print((patients.loc["id"].div(patients.loc["id"].sum(axis=1), axis=0)).round(3).to_string())
    print(f"\nwrote {SPLITS_PATH.relative_to(REPO_ROOT)} ({len(tcga)} {SOURCE} slides, seed {SEED})")


if __name__ == "__main__":
    main()
