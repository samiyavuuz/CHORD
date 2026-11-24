"""Step 6 summary: apply the selection rule of run_method_dev.py and compare with the Step 5 baselines."""

import pandas as pd

from chord.data import RESULTS_DIR

DEV = RESULTS_DIR / "step6_method_dev.csv"
STEP5 = RESULTS_DIR / "step5_semisynth.csv"
TIE_TOL = 0.005
KEYS = ["estimator", "n_min", "r"]


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    df = pd.read_csv(DEV)
    cells = ["direction", "alpha", "rho", "mix"]

    per_cell = df.groupby(KEYS + cells)["shift_err_s0"].mean()
    table = df.groupby(KEYS).agg(mean_err=("shift_err_s0", "mean"), purity=("purity", "mean"),
                                 false_present=("false_present", "mean"), missed_present=("missed_present", "mean"),
                                 knn_near=("knn_near_auroc", "mean"), knn_far=("knn_far_auroc", "mean"))
    table["worst_cell_err"] = per_cell.groupby(KEYS).max()
    table = table.sort_values("mean_err")
    print(f"== All settings (means over all {len(df) // len(table)} scenarios; worst_cell_err = max over cells of the per-cell mean)")
    print(table.to_string())

    best = table["mean_err"].min()
    tied = table[table["mean_err"] <= best + TIE_TOL].reset_index()
    tied = tied.sort_values(["estimator", "n_min", "r"], ascending=[True, False, True], key=lambda s: s.map(
        {"class_mean": 0, "matched": 1}) if s.name == "estimator" else s)
    chosen = tied.iloc[0]
    print(f"\nbest mean error {best:.3f}; tied within {TIE_TOL}: {len(tied)} setting(s)")
    print(f"CHOSEN: estimator={chosen['estimator']} n_min={int(chosen['n_min'])} r={chosen['r']}")

    ref = pd.read_csv(STEP5)
    ref = ref[ref["method"].isin(["V1a centering", "V2 oracle shared"])]
    sel = df[(df["estimator"] == chosen["estimator"]) & (df["n_min"] == chosen["n_min"]) & (df["r"] == chosen["r"])]
    both = pd.concat([ref[cells + ["method", "shift_err_s0"]],
                      sel[cells + ["shift_err_s0"]].assign(method="V3 ours (chosen)")])
    print("\n== Shift-recovery error (s0 units) by mix x rho, alpha = 1.0")
    print(both[both["alpha"] == 1.0].pivot_table(index=["mix", "method"], columns="rho", values="shift_err_s0").to_string())
    print("\n== Shift-recovery error (s0 units) by alpha (mean over mixes, rho, directions)")
    print(both.pivot_table(index="method", columns="alpha", values="shift_err_s0").to_string())
    print("\n== Chosen setting: purity / false presence by mix x rho (alpha = 1.0)")
    s1 = sel[sel["alpha"] == 1.0]
    print(s1.pivot_table(index="mix", columns="rho", values="purity").to_string())
    print(s1.pivot_table(index="mix", columns="rho", values="false_present").to_string())


if __name__ == "__main__":
    main()
