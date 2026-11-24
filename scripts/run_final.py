"""CHORD on the real cohorts with the paper's setting (chord.method.SETTINGS), compared with no correction.

Transductive: fit on the whole unlabelled cohort, evaluate on the same cohort.
Held-out patients (paper, Section 4.1): split patients 50/50 (N_SPLITS seeded repeats), fit on half A without labels,
apply to half B, evaluate on B.
Diagnostics (paper, "What CHORD selects"): per class, gated candidates, selected slides, purity and OOD slides selected.
Writes RESULTS_DIR/final_transductive.csv, final_heldout.csv, final_diagnostics.csv.
"""

import numpy as np
import pandas as pd

from chord import baselines as B
from chord.data import RESULTS_DIR
from chord.method import OURS, apply_shift, estimate_shift, frozen
from chord.metrics import evaluate_cohort
from chord.pipeline import COHORTS, SEED_INDEX, load_cohort, load_source_model

RES = RESULTS_DIR
N_SPLITS, SEED = 5, 20261028


def diagnostics(model, c, est):
    rows = []
    for cls in model.classes:
        sel = est.selected.get(cls, np.array([], int))
        rows.append({"cohort": c.name, "class": int(cls), "present": cls in est.selected,
                     "candidates": est.candidates.get(cls, 0), "selected": len(sel),
                     "purity": float(np.mean((c.role[sel] == "id") & (c.id_class[sel] == cls))) if len(sel) else np.nan,
                     "sel_near": int((c.role[sel] == "near").sum()), "sel_far": int((c.role[sel] == "far").sum())})
    return rows


def heldout(model, c, ci):
    rows = []
    patients = np.unique(c.patient_id)
    for rep in range(N_SPLITS):
        rng = np.random.default_rng([SEED, ci, rep])
        half_a = set(rng.permutation(patients)[: len(patients) // 2])
        a = np.array([p in half_a for p in c.patient_id])
        b = ~a
        est = estimate_shift(model, c.x[a], **frozen())
        out = {"V0 none": B.none(model, c.x[b]), OURS: model.features(c.x[b], apply_shift(model, est, c.x[b]))}
        for name, u in out.items():
            rows.append({"cohort": c.name, "rep": rep, "method": name, "n_fit": int(a.sum()), "n_eval": int(b.sum()),
                         **evaluate_cohort(model, u, c.role[b], c.id_class[b])})
    return rows


def show(df, title):
    cols = ["bacc", "knn_near_auroc", "knn_near_fpr95", "knn_far_auroc", "knn_far_fpr95"]
    print(f"\n######## {title}")
    for name, g in df.groupby("cohort", sort=False):
        print(f"\n== {name}")
        print(g.groupby("method", sort=False)[cols].mean().to_string())


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    model = load_source_model()
    trans, held, diag = [], [], []
    for name in COHORTS:
        ci = SEED_INDEX[name]
        c = load_cohort(name)
        est = estimate_shift(model, c.x, **frozen())
        for m, u in {"V0 none": B.none(model, c.x), OURS: model.features(c.x, est.delta)}.items():
            trans.append({"cohort": name, "method": m, **evaluate_cohort(model, u, c.role, c.id_class)})
        diag += diagnostics(model, c, est)
        held += heldout(model, c, ci)
    RES.mkdir(exist_ok=True)
    trans, held, diag = pd.DataFrame(trans), pd.DataFrame(held), pd.DataFrame(diag)
    trans.to_csv(RES / "final_transductive.csv", index=False)
    held.to_csv(RES / "final_heldout.csv", index=False)
    diag.to_csv(RES / "final_diagnostics.csv", index=False)

    show(trans, "TRANSDUCTIVE")
    show(held, f"HELD-OUT PATIENTS (mean of {N_SPLITS} splits)")
    print("\n######## CHORD DIAGNOSTICS (transductive)")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
