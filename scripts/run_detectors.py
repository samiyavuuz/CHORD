"""The ten post-hoc detectors (chord.detectors) with and without each correction (paper, Table 2).

Corrections (transductive): no correction, per-cohort centering, CORAL (a = 0.1), MNN shift, CHORD
(chord.method.SETTINGS) and the label-informed reference. Detector statistics are fitted once on CPTAC-train; the
target cohort only changes the features. Patient-level conditional
bootstrap (N_BOOT resamples) of the paired AUROC differences CHORD - each baseline, all detectors at once (Mann-Whitney
AUROC on ranks).
Writes RESULTS_DIR/det_transductive.csv (all metrics) and RESULTS_DIR/det_ci_diff.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from chord import baselines as B
from chord.data import RESULTS_DIR
from chord.detectors import DETECTORS, Detectors
from chord.method import GROUP, GROUPINGS, OURS, REFERENCE, estimate_shift, frozen
from chord.metrics import ood_metrics
from chord.pipeline import COHORTS, SEED_INDEX, load_cohort, load_source_model

RES = RESULTS_DIR
N_BOOT, SEED = 1000, 20261030
CORRECTIONS = ("V0 none", "V1a centering", "V1b CORAL a=0.1", "V1c MNN", OURS, REFERENCE)
PAIRS = ((OURS, "V0 none"), (OURS, "V1b CORAL a=0.1"), (OURS, "V1a centering"), (OURS, "V1c MNN"))


def corrections(model, c):
    true = np.where(c.role == "id", c.id_class, -1)
    est = estimate_shift(model, c.x, **frozen())
    return {"V0 none": B.none(model, c.x), "V1a centering": B.centering(model, c.x),
            "V1b CORAL a=0.1": B.coral(model, c.x, model.cov, 0.1), "V1c MNN": B.mnn_shift(model, c.x),
            OURS: model.features(c.x, est.delta),
            REFERENCE: model.features(c.x, B.oracle_grouped_delta(model, c.x, true, GROUPINGS[GROUP]))}


def auroc_cols(S: np.ndarray, is_ood: np.ndarray) -> np.ndarray:
    """AUROC (OOD positive, higher score = OOD) for every column of S."""
    r = rankdata(S, axis=0)
    n1, n0 = is_ood.sum(), (~is_ood).sum()
    return (r[is_ood].sum(0) - n1 * (n1 + 1) / 2) / (n0 * n1)


def main():
    model = load_source_model()
    det = Detectors(model)
    names = list(DETECTORS)
    rows, diffs = [], []
    for name in COHORTS:
        ci = SEED_INDEX[name]
        c = load_cohort(name)
        S = {}
        for corr, u in corrections(model, c).items():
            sc = det.scores(u)
            S[corr] = np.stack([sc[k] for k in names], 1)
            for j, k in enumerate(names):
                for g in ("near", "far"):
                    met = ood_metrics(S[corr][c.role == "id", j], S[corr][c.role == g, j])
                    rows.append({"cohort": name, "correction": corr, "detector": k, "family": DETECTORS[k],
                                 "split": g, **met})
        patients, inv = np.unique(c.patient_id, return_inverse=True)
        slides_of = [np.flatnonzero(inv == p) for p in range(len(patients))]
        rng = np.random.default_rng([SEED, ci])
        boot = {g: {corr: [] for corr in CORRECTIONS} for g in ("near", "far")}
        for _ in range(N_BOOT):
            idx = np.concatenate([slides_of[p] for p in rng.integers(len(patients), size=len(patients))])
            role = c.role[idx]
            for g in ("near", "far"):
                keep = (role == "id") | (role == g)
                if (role == g).sum() == 0:
                    continue
                for corr in CORRECTIONS:
                    boot[g][corr].append(auroc_cols(S[corr][idx][keep], role[keep] == g))
        for g in ("near", "far"):
            if not boot[g]["V0 none"]:
                continue
            is_ood = c.role[(c.role == "id") | (c.role == g)] == g
            point = {corr: auroc_cols(S[corr][(c.role == "id") | (c.role == g)], is_ood) for corr in CORRECTIONS}
            for a, b in PAIRS:
                d = np.array(boot[g][a]) - np.array(boot[g][b])
                lo, hi = np.nanpercentile(d, [2.5, 97.5], axis=0)
                for j, k in enumerate(names):
                    diffs.append({"cohort": name, "split": g, "pair": f"{a} - {b}", "detector": k,
                                  "family": DETECTORS[k], "diff": point[a][j] - point[b][j], "lo": lo[j], "hi": hi[j],
                                  "significant": bool(lo[j] > 0 or hi[j] < 0)})
        print(f"{name} done", flush=True)
    RES.mkdir(exist_ok=True)
    rows, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    rows.to_csv(RES / "det_transductive.csv", index=False)
    diffs.to_csv(RES / "det_ci_diff.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.float_format", "{:.3f}".format)
    for g in ("near", "far"):
        t = rows[rows.split == g].pivot_table(index="detector", columns=["cohort", "correction"], values="auroc", sort=False)
        print(f"\n######## {g}-OOD AUROC")
        for name in COHORTS:
            if name in t.columns.get_level_values(0):
                print(f"\n== {name}\n" + t[name][list(CORRECTIONS)].to_string())
        dd = diffs[(diffs.split == g) & (diffs.pair == f"{OURS} - V0 none")]
        txt = dd.assign(txt=dd.apply(lambda r: f"{r['diff']:+.3f}{'*' if r.significant else ' '}", axis=1))
        print(f"\n== {g}: CHORD - no correction (* = 95% patient-bootstrap CI excludes 0)")
        print(txt.pivot(index="detector", columns="cohort", values="txt").loc[names].to_string())


if __name__ == "__main__":
    main()
