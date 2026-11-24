"""Step 7b: shared vs organ-wise vs class-wise CHORD shift on the CPTAC-dev semi-synthetic benchmark.

Steps 6-7 settings (chord.method.SETTINGS). Same scenarios as Steps 5-7.
Error per scenario = mean over ID slides of |delta_j - delta_true| / s0 (equals the Step 7 error for shared).
Selection rule (fixed before running): lowest mean error over all scenarios; ties within TIE_TOL go to the
simpler version in the order shared, organ, class. Writes RESULTS_DIR/step7b_variants_dev.csv.
"""

import numpy as np
import pandas as pd

from chord.data import REPO_ROOT, RESULTS_DIR
from chord.method import SETTINGS, estimate_shift
from chord.metrics import evaluate_cohort
from chord.pipeline import load_cohort, load_source_model, src
from chord.semisynth import MIXES, class_separation, scenarios

OUT = RESULTS_DIR / "step7b_variants_dev.csv"
STEP5 = RESULTS_DIR / "step5_semisynth.csv"
ESTIMATOR, N_MIN, R, T = (SETTINGS[k] for k in ("estimator", "n_min", "r", "T"))  # Steps 6-7
VARIANTS = ("shared", "organ", "class")  # order = simplicity for tie-breaking
TIE_TOL = 0.005


def main():
    model = load_source_model()
    dev = load_cohort(src("dev"))
    s0 = class_separation(model)
    rows = []
    for sc in scenarios(model, dev):
        is_id = sc.role == "id"
        for v in VARIANTS:
            est = estimate_shift(model, sc.y, N_MIN, R, ESTIMATOR, T=T, grouping=v)
            d = np.broadcast_to(est.delta, sc.y.shape)
            present = {int(c) for c in est.selected}
            rows.append({"direction": sc.direction, "alpha": sc.alpha, "rho": sc.rho, "mix": sc.mix, "rep": sc.rep,
                         "variant": v,
                         "id_err_s0": float(np.linalg.norm(d[is_id] - sc.delta_true, axis=1).mean()) / s0,
                         "false_present": len(present - set(MIXES[sc.mix])),
                         **evaluate_cohort(model, model.features(sc.y, est.delta), sc.role, sc.id_class)})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    cells = ["direction", "alpha", "rho", "mix"]
    summ = df.groupby("variant").agg(mean_err=("id_err_s0", "mean"), knn_near=("knn_near_auroc", "mean"),
                                     knn_far=("knn_far_auroc", "mean"), msp_near=("msp_near_auroc", "mean"),
                                     bacc=("bacc", "mean"))
    summ["worst_cell_err"] = df.groupby(["variant"] + cells)["id_err_s0"].mean().groupby("variant").max()
    summ = summ.loc[list(VARIANTS)]
    print(f"== Per variant (means over all {df.groupby('variant').size().iloc[0]} scenarios)\n" + summ.to_string())
    best = summ["mean_err"].min()
    chosen = next(v for v in VARIANTS if summ.loc[v, "mean_err"] <= best + TIE_TOL)
    print(f"\nbest mean error {best:.3f}; CHOSEN variant = {chosen}")

    ref = pd.read_csv(STEP5)
    ref = ref[ref["method"].isin(["V0 none", "V2 oracle shared", "V2 oracle organ"])].rename(columns={"method": "variant"})
    both = pd.concat([ref, df.assign(variant="ours " + df["variant"])])
    sub = both[both["rho"] > 0]
    for m in ("knn_near_auroc", "msp_near_auroc", "knn_far_auroc"):
        print(f"\n== {m} by mix x rho (mean over alpha, directions)")
        print(sub.pivot_table(index=["mix", "variant"], columns="rho", values=m).to_string())
    print("\n== ID shift error by mix x rho (alpha = 1.0)")
    print(df[df["alpha"] == 1.0].pivot_table(index=["mix", "variant"], columns="rho", values="id_err_s0").to_string())
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
