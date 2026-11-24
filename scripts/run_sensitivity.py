"""Ablations and sensitivity of CHORD on the real cohorts (paper, Table 3 and Section 4; transductive).

One factor is changed at a time from the paper's setting (chord.method.SETTINGS; gate percentile 1, kNN ranking),
which was selected on the CPTAC-dev semi-synthetic benchmark; this sweep is not used for selection.
Writes RESULTS_DIR/sensitivity.csv (balanced accuracy, classes present, AUROC of every detector, near / far).
"""

import numpy as np
import pandas as pd

from chord.data import RESULTS_DIR
from chord.detectors import DETECTORS, Detectors
from chord.method import estimate_shift, frozen
from chord.metrics import auroc, balanced_accuracy
from chord.pipeline import COHORTS, load_cohort, load_source_model, src

OUT = RESULTS_DIR / "sensitivity.csv"
FROZEN = {**frozen(), "presence": True, "gate": True, "rank": "knn", "seed": 0}
GATE_Q, N_RANDOM = 0.01, 10
# Table 3: r = 1 (no strict selection), random ranking, no gate, no count threshold, class-mean estimator, organ-wise
# shifts, second pass; Section 4: ranges of n_min, r (<= 0.2) and the gate percentile.
SWEEPS = {"r": (0.05, 0.10, 0.20, 1.0), "n_min": (10, 20, 30, 50, 100), "T": (1, 2),
          "gate_q": (0.0, 0.05, 0.10, 0.25), "rank": ("random",), "grouping": ("shared", "organ"),
          "estimator": ("class_mean",), "presence": (False,), "gate": (False,)}
SWEEPS = {f: tuple(v for v in vals if v != FROZEN.get(f)) for f, vals in SWEEPS.items()}  # frozen value = row "frozen"


def evaluate(model, det, c, u):
    is_id = c.role == "id"
    out = {"bacc": balanced_accuracy(c.id_class[is_id], model.predict(u[is_id]))}
    sc = det.scores(u)
    for k in DETECTORS:
        for g in ("near", "far"):
            out[f"{k}|{g}"] = auroc(sc[k][is_id], sc[k][c.role == g])
    return out


def main():
    model = load_source_model()
    va = load_cohort(src("val"))
    det = Detectors(model)
    val_nn = model.similarity(model.features(va.x)).max(1)
    assert np.isclose(np.quantile(val_nn, GATE_Q), model.gate)
    configs = [("frozen", "-", {})] + [(f, v, {f: v}) for f, vals in SWEEPS.items() for v in vals]
    rows = []
    for name in COHORTS:
        c = load_cohort(name)
        rows.append({"cohort": name, "factor": "V0 none", "value": "-", **evaluate(model, det, c, model.features(c.x))})
        for factor, value, change in configs:
            p = {**FROZEN, **{k: v for k, v in change.items() if k in FROZEN}}
            model.gate = float(np.quantile(val_nn, change.get("gate_q", GATE_Q)))
            seeds = range(N_RANDOM) if p["rank"] == "random" else (0,)
            for seed in seeds:
                est = estimate_shift(model, c.x, p["n_min"], p["r"], p["estimator"], T=p["T"], grouping=p["grouping"],
                                     presence=p["presence"], rank=p["rank"], seed=seed, gate=p["gate"])
                u = model.features(c.x, est.delta)
                rows.append({"cohort": name, "factor": factor, "value": str(value), "seed": seed,
                             "present": "".join(str(int(q)) for q in sorted(est.selected)),
                             "n_selected": sum(len(s) for s in est.selected.values()), **evaluate(model, det, c, u)})
            model.gate = float(np.quantile(val_nn, GATE_Q))
        print(f"{name} done", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.float_format", "{:.3f}".format)
    for col in ("kNN|near", "kNN|far", "CARef|near", "MSP|near", "bacc"):
        print(f"\n== {col} (random ranking: mean over {N_RANDOM} seeds)")
        print(df.pivot_table(index=["factor", "value"], columns="cohort", values=col, sort=False)[list(COHORTS)].to_string())
    print("\n== classes declared present")
    print(df[df.seed.fillna(0) == 0].pivot_table(index=["factor", "value"], columns="cohort", values="present",
                                                 aggfunc="first", sort=False)[list(COHORTS)].to_string())


if __name__ == "__main__":
    main()
