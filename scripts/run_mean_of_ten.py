"""Mean AUROC over the ten detectors of the paper (Table 2, "Mean of ten") with bootstrap CIs, and each detector's
near-OOD AUROC on CPTAC-dev without correction (the paper reports CARef as the strongest there).

Mean of ten: mean AUROC over chord.detectors.DETECTORS, per correction, transductive, with the patient-level conditional
bootstrap of the paired differences CHORD - X (corrections fitted once; N_BOOT resamples).
Writes results/detector_dev.csv, det_mean10.csv, det_mean10_ci.csv.
"""

import numpy as np
import pandas as pd

from chord.data import RESULTS_DIR
from chord.detectors import DETECTORS, Detectors
from chord.method import OURS
from chord.metrics import auroc
from chord.pipeline import COHORTS, SEED_INDEX, load_cohort, load_source_model, src

from run_detectors import CORRECTIONS, auroc_cols, corrections

N_BOOT, SEED = 1000, 20261101
PAIRS = ((OURS, "V0 none"), (OURS, "V1a centering"), (OURS, "V1b CORAL a=0.1"), (OURS, "V1c MNN"))


def dev_ranking(model, det):
    """Near-OOD AUROC of every detector on CPTAC-dev (no correction), highest first."""
    dev = load_cohort(src("dev"))
    s = det.scores(model.features(dev.x))
    near = {k: auroc(s[k][dev.role == "id"], s[k][dev.role == "near"]) for k in DETECTORS}
    ranked = sorted(DETECTORS, key=lambda k: -near[k])
    return pd.DataFrame({"detector": ranked, "dev_near_auroc": [near[k] for k in ranked]}), ranked


def main():
    model = load_source_model()
    det = Detectors(model)
    dev, top = dev_ranking(model, det)  # columns in CPTAC-dev order
    print("CPTAC-dev near-OOD AUROC:", ", ".join(f"{k} {v:.3f}" for k, v in zip(dev.detector, dev.dev_near_auroc)))
    rows, cis = [], []
    for name in COHORTS:
        ci = SEED_INDEX[name]
        c = load_cohort(name)
        S = {}
        for corr, u in corrections(model, c).items():
            sc = det.scores(u)
            S[corr] = np.stack([sc[k] for k in top], 1)
        patients, inv = np.unique(c.patient_id, return_inverse=True)
        slides_of = [np.flatnonzero(inv == p) for p in range(len(patients))]
        for g in ("near", "far"):
            keep = (c.role == "id") | (c.role == g)
            if (c.role == g).sum() == 0:
                continue
            point = {corr: auroc_cols(S[corr][keep], c.role[keep] == g) for corr in CORRECTIONS}
            for corr in CORRECTIONS:
                rows.append({"cohort": name, "split": g, "correction": corr, "mean_auroc": point[corr].mean(),
                             **{f"auroc_{k}": v for k, v in zip(top, point[corr])}})
            rng = np.random.default_rng([SEED, ci, g == "near"])
            boot = {corr: [] for corr in CORRECTIONS}
            for _ in range(N_BOOT):
                idx = np.concatenate([slides_of[p] for p in rng.integers(len(patients), size=len(patients))])
                r = c.role[idx]
                k2 = (r == "id") | (r == g)
                for corr in CORRECTIONS:
                    boot[corr].append(auroc_cols(S[corr][idx][k2], r[k2] == g).mean())
            for a, b in PAIRS:
                dlt = np.array(boot[a]) - np.array(boot[b])
                lo, hi = np.percentile(dlt, [2.5, 97.5])
                cis.append({"cohort": name, "split": g, "pair": f"{a} - {b}", "diff": point[a].mean() - point[b].mean(),
                            "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)})
        print(f"{name} done", flush=True)
    dev.to_csv(RESULTS_DIR / "detector_dev.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "det_mean10.csv", index=False)
    cis = pd.DataFrame(cis)
    cis.to_csv(RESULTS_DIR / "det_mean10_ci.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(pd.DataFrame(rows).pivot_table(index=["split", "correction"], columns="cohort", values="mean_auroc").to_string())
    print(cis.to_string(index=False))


if __name__ == "__main__":
    main()
