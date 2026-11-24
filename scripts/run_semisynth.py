"""Step 5: semi-synthetic benchmark (CPTAC-dev) for the baselines and the label-informed reference.

Per scenario: shift-recovery error |delta_hat - delta_true| (for methods that estimate one translation), reported in
units of s0, plus ID balanced accuracy and near/far OOD metrics. Writes RESULTS_DIR/step5_semisynth.csv.
"""

import numpy as np
import pandas as pd

from chord import baselines as B
from chord.data import REPO_ROOT, RESULTS_DIR
from chord.metrics import evaluate_cohort
from chord.pipeline import load_cohort, load_source_model, src
from chord.semisynth import class_separation, scenarios

OUT = RESULTS_DIR / "step5_semisynth.csv"
CORAL_A = 0.1  # main CORAL setting (paper)


def methods(model, sc):
    """name -> (features, estimated translation or None)."""
    true = np.where(sc.role == "id", sc.id_class, -1)  # label-informed reference only
    u_or, d_or = B.oracle_shared(model, sc.y, true)
    return {
        "V0 none": (B.none(model, sc.y), np.zeros_like(sc.delta_true)),
        "V1a centering": (B.centering(model, sc.y), sc.y.mean(0) - model.mu),
        f"V1b CORAL a={CORAL_A}": (B.coral(model, sc.y, model.cov, CORAL_A), None),
        "V2 oracle shared": (u_or, d_or),
        "V2 oracle organ": (B.oracle_grouped(model, sc.y, true, B.ORGAN_OF_CLASS), None),
        "V2 oracle class": (B.oracle_grouped(model, sc.y, true, B.CLASS_OF_CLASS), None),
    }


def main():
    model = load_source_model()
    dev = load_cohort(src("dev"))
    s0 = class_separation(model)
    print(f"s0 (mean source class separation) = {s0:.4f}; "
          f"dev pool: id={(dev.role == 'id').sum()} near={(dev.role == 'near').sum()} far={(dev.role == 'far').sum()}")

    rows = []
    for sc in scenarios(model, dev):
        for name, (u, d_hat) in methods(model, sc).items():
            err = float(np.linalg.norm(d_hat - sc.delta_true)) / s0 if d_hat is not None else np.nan
            rows.append({"direction": sc.direction, "alpha": sc.alpha, "rho": sc.rho, "mix": sc.mix, "rep": sc.rep,
                         "method": name, "n": len(sc.y), "shift_err_s0": err,
                         **evaluate_cohort(model, u, sc.role, sc.id_class)})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    est = df[df["method"].isin(["V1a centering", "V2 oracle shared"])]
    for alpha in (0.0, 1.0):
        print(f"\n== Shift-recovery error in units of s0 (alpha={alpha}; mean over directions and reps)")
        print(est[est["alpha"] == alpha].pivot_table(index=["method", "mix"], columns="rho", values="shift_err_s0").to_string())
    for metric in ("knn_near_auroc", "knn_far_auroc", "knn_far_fpr95", "msp_near_auroc", "bacc"):
        print(f"\n== {metric} (alpha=1.0; rho>0; mean over directions and reps)")
        sub = df[(df["alpha"] == 1.0) & (df["rho"] > 0)]
        print(sub.pivot_table(index=["mix", "method"], columns="rho", values=metric).to_string())
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
