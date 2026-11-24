"""Step 6: CHORD (one pass) on the CPTAC-dev semi-synthetic benchmark; selection of estimator, n_min and r.

Grid: estimator x n_min x r, T = 1. Same scenarios as Step 5 (seeded per scenario).
Selection rule (fixed before running): lowest mean shift-recovery error (units of s0) over all scenarios.
Settings within TIE_TOL of the best count as tied; among tied settings prefer class_mean, then larger n_min,
then smaller r. Writes RESULTS_DIR/step6_method_dev.csv.
"""

import itertools

import numpy as np
import pandas as pd

from chord.data import REPO_ROOT, RESULTS_DIR
from chord.method import ESTIMATORS, estimate_shift
from chord.metrics import evaluate_cohort
from chord.pipeline import load_cohort, load_source_model, src
from chord.semisynth import MIXES, class_separation, scenarios

OUT = RESULTS_DIR / "step6_method_dev.csv"
STEP5 = RESULTS_DIR / "step5_semisynth.csv"
N_MIN = (20, 30, 50)
R = (0.10, 0.25, 0.50)
TIE_TOL = 0.005


def selection_stats(est, role, id_class, mix):
    sel = [(c, j) for c, js in est.selected.items() for j in js]
    purity = np.mean([role[j] == "id" and id_class[j] == c for c, j in sel]) if sel else np.nan
    present = set(int(c) for c in est.selected)
    truth = set(MIXES[mix])
    return {"purity": purity, "n_selected": len(sel), "n_present": len(present),
            "false_present": len(present - truth), "missed_present": len(truth - present)}


def main():
    model = load_source_model()
    dev = load_cohort(src("dev"))
    s0 = class_separation(model)
    rows = []
    for sc in scenarios(model, dev):
        for estimator, n_min, r in itertools.product(ESTIMATORS, N_MIN, R):
            est = estimate_shift(model, sc.y, n_min, r, estimator, T=1)
            rows.append({"direction": sc.direction, "alpha": sc.alpha, "rho": sc.rho, "mix": sc.mix, "rep": sc.rep,
                         "estimator": estimator, "n_min": n_min, "r": r,
                         "shift_err_s0": float(np.linalg.norm(est.delta - sc.delta_true)) / s0,
                         **selection_stats(est, sc.role, sc.id_class, sc.mix),
                         **evaluate_cohort(model, model.features(sc.y, est.delta), sc.role, sc.id_class)})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
