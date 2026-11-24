"""Read-only access to the TITAN slide embeddings and the slide manifest."""

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# TITAN slide embeddings, <cohort>/<subset>/{data,filenames,labels}.npy; shipped in data/titan, overridable.
TITAN_ROOT = Path(os.environ.get("CHORD_TITAN_ROOT", REPO_ROOT / "data" / "titan"))
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
# Source cohort (labelled ID training data) and its patient-level splits (scripts/make_splits.py).
SOURCE = "CPTAC"
SPLITS_PATH = REPO_ROOT / "data" / "splits.csv"
RESULTS_DIR = REPO_ROOT / "results"
EMB_DIM = 768

ROLES = ("id", "near", "far", "excluded")


@functools.cache
def load_subset(subset: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Embeddings and slide ids of one `<cohort>/<subset>` folder, validated and read-only."""
    folder = TITAN_ROOT / subset
    emb = np.load(folder / "data.npy", allow_pickle=True)
    if emb.dtype == object:
        emb = np.stack(emb)
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    names = tuple(str(x).removesuffix(".h5") for x in np.load(folder / "filenames.npy", allow_pickle=True))
    if emb.ndim != 2 or emb.shape[1] != EMB_DIM:
        raise ValueError(f"{subset}: unexpected embedding shape {emb.shape}")
    if len(names) != len(emb):
        raise ValueError(f"{subset}: {len(emb)} embeddings but {len(names)} filenames")
    if not np.isfinite(emb).all():
        raise ValueError(f"{subset}: non-finite embedding values")
    emb.setflags(write=False)
    return emb, names


def read_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"site": str, "patient_id": str}, keep_default_na=False)


def read_manifest_with_splits() -> pd.DataFrame:
    """Manifest with a `split` column (train / val / dev / test for the source cohort, empty elsewhere)."""
    m = read_manifest()
    s = pd.read_csv(SPLITS_PATH, keep_default_na=False)
    out = m.merge(s, on=["cohort", "slide_id"], how="left", validate="one_to_one")
    out["split"] = out["split"].fillna("")
    return out


def load_embeddings(manifest: pd.DataFrame) -> np.ndarray:
    """Raw embeddings for the manifest rows, in row order; checks every slide id against its source file."""
    out = np.empty((len(manifest), EMB_DIM), dtype=np.float32)
    rows = manifest["source_row"].to_numpy()
    ids = manifest["slide_id"].to_numpy()
    for subset, idx in manifest.groupby("source_subset", sort=False).indices.items():
        emb, names = load_subset(subset)
        for i in idx:
            if names[rows[i]] != ids[i]:
                raise ValueError(f"{subset} row {rows[i]}: expected {ids[i]}, found {names[rows[i]]}")
        out[idx] = emb[rows[idx]]
    return out
