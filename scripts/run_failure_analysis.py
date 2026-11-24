"""Failure analysis of CHORD: false class presence (paper, Discussion). Semi-synthetic, additive shift.

Pool: source-dev (same scenarios as development), absent-class mixes (GBM only, lung only), alpha = 1. Two OOD pools per scenario:
with HNSCC (default) and without HNSCC (head-and-neck squamous cell carcinoma removed; same OOD share and slide totals).
Records false/missed presences, selection purity, the diagnoses of selected OOD slides and the OOD metrics of
CHORD vs no correction. Writes RESULTS_DIR/failure.csv and RESULTS_DIR/failure_selected_ood.csv.
"""

import collections

import numpy as np
import pandas as pd

from chord.data import RESULTS_DIR
from chord.method import estimate_shift, frozen
from chord.metrics import evaluate_cohort
from chord.pipeline import LOOKALIKE, load_cohort, load_source_model, src
from chord.semisynth import MIXES, SEED, scenarios

RES = RESULTS_DIR
POOLS = {src("dev"): SEED}  # CPTAC-dev pool; absent-class mixes at the largest shift
MIXES_RUN, ALPHAS_RUN = ("gbm", "lung"), (1.0,)


def main():
    model = load_source_model()
    rows, sel_rows = [], []
    for pool, seed in POOLS.items():
        c = load_cohort(pool)
        for cond, keep in ((f"with {LOOKALIKE}", None), (f"without {LOOKALIKE}", c.diagnosis != LOOKALIKE)):
            for sc in scenarios(model, c, seed=seed, ood_keep=keep, mixes=MIXES_RUN, alphas=ALPHAS_RUN):
                est = estimate_shift(model, sc.y, **frozen())
                truth = set(MIXES[sc.mix])
                present = {int(k) for k in est.selected}
                sel = [(int(k), j) for k, js in est.selected.items() for j in js]
                purity = np.mean([sc.role[j] == "id" and sc.id_class[j] == k for k, j in sel]) if sel else np.nan
                ood_dx = collections.Counter(sc.diagnosis[j] for k, j in sel if sc.role[j] != "id")
                for dx, n in ood_dx.items():
                    sel_rows.append({"pool": pool, "condition": cond, "mix": sc.mix, "rho": sc.rho, "alpha": sc.alpha,
                                     "direction": sc.direction, "rep": sc.rep, "diagnosis": dx, "n_selected": n})
                base = {"pool": pool, "condition": cond, "direction": sc.direction, "alpha": sc.alpha, "rho": sc.rho,
                        "mix": sc.mix, "rep": sc.rep, "false_present": len(present - truth),
                        "false_classes": "".join(str(k) for k in sorted(present - truth)),
                        "missed_present": len(truth - present), "purity": purity, "n_selected": len(sel)}
                r_ours = evaluate_cohort(model, model.features(sc.y, est.delta), sc.role, sc.id_class)
                r_v0 = evaluate_cohort(model, model.features(sc.y), sc.role, sc.id_class)
                rows.append({**base, **{f"ours_{k}": v for k, v in r_ours.items()}, **{f"v0_{k}": v for k, v in r_v0.items()}})
            print(f"{pool} / {cond} done", flush=True)
    df, sel = pd.DataFrame(rows), pd.DataFrame(sel_rows)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "failure.csv", index=False)
    sel.to_csv(RES / "failure_selected_ood.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    df["any_false"] = df["false_present"] > 0
    df["near_gain"] = df["ours_knn_near_auroc"] - df["v0_knn_near_auroc"]
    for stat in ("any_false", "purity", "near_gain"):
        print(f"\n== {stat} by pool x condition x mix (rows) and rho (cols); mean over alpha, directions, reps")
        print(df.pivot_table(index=["pool", "condition", "mix"], columns="rho", values=stat).to_string())
    print("\n== which classes are falsely declared present (counts of scenarios)")
    print(df[df["any_false"]].groupby(["pool", "condition", "mix"])["false_classes"].value_counts().unstack(fill_value=0).to_string())
    print("\n== OOD diagnoses among selected slides (total slides over scenarios), absent-class mixes only")
    s = sel[sel["mix"].isin(["gbm", "lung"])]
    print(s.groupby(["pool", "condition", "mix", "diagnosis"])["n_selected"].sum().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
