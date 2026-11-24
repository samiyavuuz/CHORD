"""Step 7: number of passes T of CHORD on the CPTAC-dev semi-synthetic benchmark.

Step 6 settings (chord.method.SETTINGS). One run of 10 passes per scenario; the estimate after pass t equals the
T = t estimate. Selection rule (fixed before running): lowest mean shift-recovery error over all scenarios; ties
within TIE_TOL go to the smaller T. Writes RESULTS_DIR/step7_iterations_dev.csv (one row per scenario x pass).
"""

import numpy as np
import pandas as pd

from chord.data import REPO_ROOT, RESULTS_DIR
from chord.method import SETTINGS, estimate_shift
from chord.metrics import evaluate_cohort
from chord.pipeline import load_cohort, load_source_model, src
from chord.semisynth import MIXES, class_separation, scenarios

OUT = RESULTS_DIR / "step7_iterations_dev.csv"
ESTIMATOR, N_MIN, R = (SETTINGS[k] for k in ("estimator", "n_min", "r"))  # selected in Step 6
T_MAX = 10
T_GRID = (1, 2, 3, 5, 10)
TIE_TOL = 0.005


def main():
    model = load_source_model()
    dev = load_cohort(src("dev"))
    s0 = class_separation(model)
    rows = []
    for sc in scenarios(model, dev):
        est = estimate_shift(model, sc.y, N_MIN, R, ESTIMATOR, T=T_MAX)
        prev = np.zeros_like(sc.delta_true)
        for t, (delta, selected) in enumerate(zip(est.history, est.selections), start=1):
            sel = [(c, j) for c, js in selected.items() for j in js]
            present = {int(c) for c in selected}
            row = {"direction": sc.direction, "alpha": sc.alpha, "rho": sc.rho, "mix": sc.mix, "rep": sc.rep, "T": t,
                   "shift_err_s0": float(np.linalg.norm(delta - sc.delta_true)) / s0,
                   "step_s0": float(np.linalg.norm(delta - prev)) / s0,
                   "purity": np.mean([sc.role[j] == "id" and sc.id_class[j] == c for c, j in sel]) if sel else np.nan,
                   "false_present": len(present - set(MIXES[sc.mix])),
                   "missed_present": len(set(MIXES[sc.mix]) - present)}
            if t in T_GRID:
                row.update(evaluate_cohort(model, model.features(sc.y, delta), sc.role, sc.id_class))
            rows.append(row)
            prev = delta
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    g = df[df["T"].isin(T_GRID)]
    summary = g.groupby("T").agg(mean_err=("shift_err_s0", "mean"), purity=("purity", "mean"),
                                 false_present=("false_present", "mean"), knn_near=("knn_near_auroc", "mean"),
                                 knn_far=("knn_far_auroc", "mean"), msp_near=("msp_near_auroc", "mean"))
    summary["worst_cell_err"] = g.groupby(["T", "direction", "alpha", "rho", "mix"])["shift_err_s0"].mean().groupby("T").max()
    print("== Per T (means over all scenarios)\n" + summary.to_string())
    best = summary["mean_err"].min()
    chosen = int(summary.index[summary["mean_err"] <= best + TIE_TOL].min())
    print(f"\nbest mean error {best:.3f}; CHOSEN T = {chosen}")

    print("\n== Shift error by alpha x T\n" + df.pivot_table(index="alpha", columns="T", values="shift_err_s0").to_string())
    print("\n== Shift error by mix x rho, alpha = 1.0, T in {1, chosen}")
    a1 = df[(df["alpha"] == 1.0) & df["T"].isin({1, chosen})]
    print(a1.pivot_table(index=["mix", "T"], columns="rho", values="shift_err_s0").to_string())
    print("\n== Purity by mix x rho, alpha = 1.0, T in {1, chosen}")
    print(a1.pivot_table(index=["mix", "T"], columns="rho", values="purity").to_string())
    print("\n== Convergence: mean step size |delta_t - delta_(t-1)| / s0 per pass")
    print(df.groupby("T")["step_s0"].mean().to_frame().T.to_string())
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
