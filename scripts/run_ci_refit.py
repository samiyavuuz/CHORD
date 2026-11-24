"""Refitting bootstrap (paper, Protocol): CHORD refitted on every resample, kNN AUROC vs no correction.

Each replicate resamples patients with replacement (all slides of a drawn patient), refits CHORD on the resampled
cohort (transductive) and evaluates it on that same resample; both methods use identical resamples (paired). The main
intervals (run_detectors.py, run_mean_of_ten.py) keep the fitted correction fixed instead.
Writes RESULTS_DIR/refit_raw.csv, refit_ci.csv, refit_diff.csv.
"""

import numpy as np
import pandas as pd

from chord import baselines as B
from chord.data import RESULTS_DIR
from chord.method import OURS, estimate_shift, frozen
from chord.metrics import evaluate_cohort
from chord.pipeline import COHORTS, SEED_INDEX, load_cohort, load_source_model

RES = RESULTS_DIR
N_BOOT, SEED = 50, 20261102  # 50 refitted resamples per cohort (paper, Protocol)
METRICS = ("knn_near_auroc", "knn_far_auroc")
OTHERS = ("V0 none",)


def fit_all(model, y):
    est = estimate_shift(model, y, **frozen())
    feats = {"V0 none": B.none(model, y), OURS: model.features(y, est.delta)}
    return feats, "".join(str(int(k)) for k in sorted(est.selected))


def main():
    model = load_source_model()
    raw = []
    for name in COHORTS:
        ci = SEED_INDEX[name]
        c = load_cohort(name)
        patients, inv = np.unique(c.patient_id, return_inverse=True)
        slides_of = [np.flatnonzero(inv == p) for p in range(len(patients))]
        rng = np.random.default_rng([SEED, ci])
        for b in range(N_BOOT):
            idx = np.concatenate([slides_of[p] for p in rng.integers(len(patients), size=len(patients))])
            y, role, cls = c.x[idx], c.role[idx], c.id_class[idx]
            try:
                feats, present = fit_all(model, y)
            except Exception as e:  # recorded, never silently dropped
                raw.append({"cohort": name, "rep": b, "method": "FAILED", "error": repr(e)})
                continue
            for m, u in feats.items():
                raw.append({"cohort": name, "rep": b, "method": m, "ours_present": present,
                            **evaluate_cohort(model, u, role, cls)})
        print(f"{name}: {N_BOOT} refitted resamples", flush=True)
    raw = pd.DataFrame(raw)
    RES.mkdir(exist_ok=True)
    raw.to_csv(RES / "refit_raw.csv", index=False)

    point = pd.read_csv(RES / "final_transductive.csv")
    rows, diffs = [], []
    for name in COHORTS:
        r = raw[raw["cohort"] == name]
        for m in (OURS,) + OTHERS:
            g = r[r["method"] == m]
            for k in METRICS:
                v = g[k].dropna()
                p = point[(point["cohort"] == name) & (point["method"] == m)][k]
                rows.append({"cohort": name, "method": m, "metric": k, "value": float(p.iloc[0]) if len(p) else np.nan,
                             "lo": np.percentile(v, 2.5) if len(v) else np.nan, "hi": np.percentile(v, 97.5) if len(v) else np.nan,
                             "valid": len(v), "total": len(g)})
        a = r[r["method"] == OURS].set_index("rep")
        for o in OTHERS:
            bo = r[r["method"] == o].set_index("rep")
            for k in METRICS:
                d = (a[k] - bo[k]).dropna()
                lo, hi = np.percentile(d, [2.5, 97.5])
                diffs.append({"cohort": name, "pair": f"{OURS} - {o}", "metric": k, "mean_diff": d.mean(), "lo": lo, "hi": hi,
                              "valid": len(d), "significant": bool(lo > 0 or hi < 0)})
    rows, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    rows.to_csv(RES / "refit_ci.csv", index=False)
    diffs.to_csv(RES / "refit_diff.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n== failed replicates:", int((raw["method"] == "FAILED").sum()))
    print("== classes declared present by CHORD (counts over resamples)")
    print(raw[raw["method"] == OURS].groupby("cohort")["ours_present"].value_counts().unstack(fill_value=0).to_string())
    fmt = rows.assign(txt=rows.apply(lambda x: f"{x.value:.3f} [{x.lo:.3f}, {x.hi:.3f}] n={x.valid}", axis=1))
    for k in METRICS:
        print(f"\n== refit CI, {k}")
        print(fmt[fmt.metric == k].pivot(index="method", columns="cohort", values="txt")[list(COHORTS)].to_string())
    dfmt = diffs.assign(txt=diffs.apply(lambda x: f"{x.mean_diff:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}]{' *' if x.significant else ''}", axis=1))
    for k in METRICS:
        print(f"\n== refit paired differences, {k}")
        print(dfmt[dfmt.metric == k].pivot(index="pair", columns="cohort", values="txt")[list(COHORTS)].to_string())


if __name__ == "__main__":
    main()
