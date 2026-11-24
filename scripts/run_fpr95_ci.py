"""FPR95 of the ten reported detectors with patient-level bootstrap CIs (paper, main table).

Same detectors, corrections and conditional bootstrap as scripts/run_mean_of_ten.py (corrections fitted once, patients
resampled N_BOOT times), but for FPR95 (chord.metrics: OOD accepted at the threshold that accepts 95% of the cohort's
ID slides; lower is better). Paired differences CHORD - X per detector and for the mean over the ten.
Writes results/det_mean10_fpr95.csv and det_mean10_fpr95_ci.csv.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from chord.data import RESULTS_DIR
from chord.detectors import DETECTORS, Detectors
from chord.method import OURS
from chord.metrics import ood_metrics
from chord.pipeline import COHORTS, SEED_INDEX, load_cohort, load_source_model

from run_detectors import CORRECTIONS, corrections
from run_mean_of_ten import PAIRS

N_BOOT, SEED = 1000, 20261102
MEAN = "Mean of ten"


def fpr95(score: np.ndarray, is_ood: np.ndarray) -> float:
    """As chord.metrics.ood_metrics: ID positive, ranked by -score, first ROC point with TPR >= 0.95."""
    fpr, tpr, _ = roc_curve(~is_ood, -score)
    return float(fpr[np.searchsorted(tpr, 0.95, side="left")])


def fpr95_cols(S: np.ndarray, is_ood: np.ndarray) -> np.ndarray:
    return np.array([fpr95(S[:, j], is_ood) for j in range(S.shape[1])])


def main():
    top = list(DETECTORS)
    model = load_source_model()
    det = Detectors(model)
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
            point = {corr: fpr95_cols(S[corr][keep], c.role[keep] == g) for corr in CORRECTIONS}
            for j, k in enumerate(top):  # must reproduce det_transductive.csv
                ref = ood_metrics(S["V0 none"][c.role == "id", j], S["V0 none"][c.role == g, j])["fpr95"]
                assert abs(ref - point["V0 none"][j]) < 1e-12, (name, g, k)
            for corr in CORRECTIONS:
                rows.append({"cohort": name, "split": g, "correction": corr, "mean_fpr95": point[corr].mean(),
                             **{f"fpr95_{k}": v for k, v in zip(top, point[corr])}})
            rng = np.random.default_rng([SEED, ci, g == "near"])
            boot = {corr: [] for corr in CORRECTIONS}
            for _ in range(N_BOOT):
                idx = np.concatenate([slides_of[p] for p in rng.integers(len(patients), size=len(patients))])
                r = c.role[idx]
                k2 = (r == "id") | (r == g)
                for corr in CORRECTIONS:
                    boot[corr].append(fpr95_cols(S[corr][idx][k2], r[k2] == g))
            for a, b in PAIRS:
                dlt = np.array(boot[a]) - np.array(boot[b])  # (N_BOOT, n_detectors)
                dlt = np.c_[dlt, dlt.mean(1)]
                diff = np.r_[point[a] - point[b], point[a].mean() - point[b].mean()]
                lo, hi = np.percentile(dlt, [2.5, 97.5], axis=0)
                for j, k in enumerate([*top, MEAN]):
                    cis.append({"cohort": name, "split": g, "pair": f"{a} - {b}", "detector": k, "diff": diff[j],
                                "lo": lo[j], "hi": hi[j], "significant": bool(lo[j] > 0 or hi[j] < 0)})
        print(f"{name} done", flush=True)
    rows, cis = pd.DataFrame(rows), pd.DataFrame(cis)
    rows.to_csv(RESULTS_DIR / "det_mean10_fpr95.csv", index=False)
    cis.to_csv(RESULTS_DIR / "det_mean10_fpr95_ci.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(rows.pivot_table(index=["split", "correction"], columns="cohort", values="mean_fpr95").to_string())
    print(cis[(cis.pair == f"{OURS} - V0 none") & cis.detector.isin([MEAN, "kNN", "CARef"])].to_string(index=False))


if __name__ == "__main__":
    main()
