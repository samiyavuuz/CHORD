"""Tables and numbers of the paper, from the result CSVs in results/.

Writes results/paper/tables/main.tex (Table 2), results/paper/tables/cohorts.tex and ablation.tex (Tables 1 and 3)
and results/paper/numbers.tex (the numbers quoted in the paper text). "Mean of ten" is
the mean AUROC (FPR95) over the ten detectors (chord.detectors); the ablation table recomputes it from the
per-detector columns of the sensitivity sweep (sensitivity.csv). The claims made in the text are asserted here.
"""

import re

import pandas as pd

from chord import baselines as B
from chord.data import REPO_ROOT, SOURCE
from chord.method import GROUP, OURS, REFERENCE, SETTINGS
from chord.pipeline import load_cohort, load_source_model

R = REPO_ROOT / "results"
OUT = REPO_ROOT / "results" / "paper"
TAB = OUT / "tables"
MANUSCRIPT = REPO_ROOT / "paper" / "main.tex"  # LaTeX source of the paper, if available locally (not distributed)

TARGETS = ["TCGA", "PLCO", "EBRAINS"]
TAGS = {"TCGA": "Tcga", "PLCO": "Plco", "EBRAINS": "Ebrains", "CPTAC-test": "Ctest"}
FREE = ["V0 none", "V1a centering", "V1b CORAL a=0.1", "V1c MNN", OURS]
REF = REFERENCE
ROWNAME = {"V0 none": "No correction", "V1a centering": "Centering", "V1b CORAL a=0.1": "CORAL", "V1c MNN": "MNN shift",
           OURS: "\\textbf{CHORD (ours)}", REF: "\\textit{Label-informed$^\\dagger$}"}
macros = {}


def mac(name, value, fmt="{:.3f}"):
    assert name.isalpha(), name
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def tabular(lines, colspec):
    return "\n".join([f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def ten_detectors():
    """The ten detectors in CPTAC-dev order (results/detector_dev.csv); CARef, the strongest there, comes first."""
    dev = pd.read_csv(R / "detector_dev.csv")
    assert dev.detector.iloc[0] == "CARef", "the paper reports CARef as the strongest detector on CPTAC-dev"
    return dev.sort_values("dev_near_auroc", ascending=False).detector.tolist()


def main_table():
    t = pd.read_csv(R / "det_transductive.csv")
    top = {"auroc": pd.read_csv(R / "det_mean10.csv").rename(columns={"mean_auroc": "mean"}),
           "fpr95": pd.read_csv(R / "det_mean10_fpr95.csv").rename(columns={"mean_fpr95": "mean"})}
    c = {"auroc": pd.read_csv(R / "det_ci_diff.csv"), "fpr95": pd.read_csv(R / "det_mean10_fpr95_ci.csv")}
    tci = pd.read_csv(R / "det_mean10_ci.csv")
    vs_none = f"{OURS} - V0 none"
    better = {"auroc": max, "fpr95": min}

    def value(block, coh, g, m, metric="auroc"):
        if block == "Mean of ten":
            x = top[metric]
            return x[(x.cohort == coh) & (x.split == g) & (x.correction == m)]["mean"].iloc[0]
        return t[(t.detector == block) & (t.cohort == coh) & (t.split == g) & (t.correction == m)][metric].iloc[0]

    def diff(block, coh, g, pair=vs_none, metric="auroc"):
        if block == "Mean of ten" and metric == "auroc":
            return tci[(tci.cohort == coh) & (tci.split == g) & (tci.pair == pair)].iloc[0]
        x = c[metric]
        return x[(x.detector == block) & (x.cohort == coh) & (x.split == g) & (x.pair == pair)].iloc[0]

    cols = [(coh, g, metric) for coh in TARGETS for g in ("near", "far") for metric in ("auroc", "fpr95")]
    lines = ["Detector & Correction & \\multicolumn{4}{c}{TCGA} & \\multicolumn{4}{c}{PLCO} & \\multicolumn{4}{c}{EBRAINS} \\\\",
             "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}",
             " & & " + " & ".join(["\\multicolumn{2}{c}{Near} & \\multicolumn{2}{c}{Far}"] * 3) + " \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}",
             " & & " + " & ".join(["AUROC$\\uparrow$ & FPR95$\\downarrow$"] * 6) + " \\\\", "\\midrule"]
    for bi, block in enumerate(["kNN", "CARef", "Mean of ten"]):
        for m in FREE + [REF]:
            r = [block if m == FREE[0] else "", ROWNAME[m]]
            for coh, g, metric in cols:
                v = value(block, coh, g, m, metric)
                s = f"{v:.3f}"
                if m == OURS and diff(block, coh, g, metric=metric).significant:
                    s += "$^{*}$"
                if m != REF and round(v, 3) == better[metric](round(value(block, coh, g, k, metric), 3) for k in FREE):
                    s = f"\\textbf{{{s}}}"
                if m == REF:
                    s = f"\\textit{{{s}}}"
                r.append(s)
            lines.append(" & ".join(r) + " \\\\")
        if bi < 2:
            lines.append("\\midrule")
    (TAB / "main.tex").write_text(tabular(lines, "ll" + "c" * len(cols)))

    mac("knnNearNonePlco", value("kNN", "PLCO", "near", "V0 none"))
    mac("knnNearHivePlco", value("kNN", "PLCO", "near", OURS))
    mac("tenNearNoneCtest", value("Mean of ten", "CPTAC-test", "near", "V0 none"))
    mac("tenNearHiveCtest", value("Mean of ten", "CPTAC-test", "near", OURS))

    # claims in the text (Abstract, Section 4.1)
    ten_gain = {(coh, g): value("Mean of ten", coh, g, OURS) - value("Mean of ten", coh, g, "V0 none")
                for coh in TARGETS for g in ("near", "far")}
    assert all(d > 0 for d in ten_gain.values()), "CHORD improves mean near- and far-OOD AUROC on every target"
    assert all(value("Mean of ten", coh, "far", OURS, "fpr95") < value("Mean of ten", coh, "far", "V0 none", "fpr95")
               for coh in TARGETS), "CHORD reduces mean far-OOD FPR95 on every target"
    assert value("kNN", "PLCO", "near", OURS) < value("kNN", "PLCO", "near", "V0 none")
    for coh, g in (("EBRAINS", "near"), ("EBRAINS", "far"), ("PLCO", "far")):
        y = diff("Mean of ten", coh, g, f"{OURS} - V1c MNN")
        assert y.significant and y["diff"] > 0, ("CHORD significantly outperforms MNN", coh, g)
    print("mean near-OOD AUROC gain (relative, %):",
          {coh: round(float(100 * ten_gain[(coh, "near")] / value("Mean of ten", coh, "near", "V0 none")), 2) for coh in TARGETS})


def ablation_table(ten):
    d = pd.read_csv(R / "sensitivity.csv")
    for g in ("near", "far"):
        d[g] = d[[f"{x}|{g}" for x in ten]].mean(axis=1)
    p = d.groupby(["factor", "value", "cohort"])[["near", "far"]].mean()  # random ranking: mean over its seeds
    alt_group = "organ" if GROUP == "shared" else "shared"
    alt_t = "2" if SETTINGS["T"] == 1 else "1"
    rows = [("V0 none", "-", "No correction"), ("frozen", "-", "\\textbf{CHORD}"),
            ("r", "1.0", "w/o strict selection ($r=1$)"), ("gate", "False", "w/o similarity gate"),
            ("presence", "False", "w/o count threshold")]
    # rows of the paper's ablation table (Table 3), in order; a rule follows the rows in rule_after
    table = [rows[0], rows[1], rows[2], ("rank", "random", "Random ranking"), rows[3], rows[4],
             ("estimator", "class_mean", "Source class means"),
             ("grouping", alt_group, "Organ-wise shifts" if alt_group == "organ" else "Shared shift"),
             ("T", alt_t, "Second pass" if alt_t == "2" else "One pass")]
    rule_after = {1, 5}
    lines = ["Variant & \\multicolumn{2}{c}{TCGA} & \\multicolumn{2}{c}{PLCO} & \\multicolumn{2}{c}{EBRAINS} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}", " & Near & Far & Near & Far & Near & Far \\\\", "\\midrule"]
    for i, (f, v, name) in enumerate(table):
        lines.append(" & ".join([name, *[f"{p.loc[(f, v, coh), g]:.3f}" for coh in TARGETS for g in ("near", "far")]]) + " \\\\")
        if i in rule_after:
            lines.append("\\midrule")
    (TAB / "ablation.tex").write_text(tabular(lines, "lcccccc"))
    named = [*zip(rows, ["None", "Full", "AllCand", "NoGate", "NoPres"]), (("rank", "random", ""), "Random"), (("grouping", alt_group, ""), "AltGroup"), (("T", alt_t, ""), "AltT"),
             (("estimator", "class_mean", ""), "ClassMean")]
    for f, v, nm in [(*r[:2], n) for r, n in named]:
        for coh in TARGETS:
            for g, G in (("near", "Near"), ("far", "Far")):
                mac(f"abl{nm}{G}{TAGS[coh]}", p.loc[(f, v, coh), g])
    # largest within-cohort range of mean near-OOD AUROC when varying one parameter (Section 4: n_min in [10, 100],
    # r in [0.05, 0.2], gate percentile in [0, 25]; the paper's setting is included)
    swept = p.reset_index()
    for f, keep, nm in (("r", lambda v: float(v) <= 0.2, "R"), ("n_min", lambda v: True, "Nmin"), ("gate_q", lambda v: True, "Gate")):
        vals = sorted({v for v in swept[swept.factor == f].value if keep(v)}, key=float)
        span = []
        for coh in TARGETS:
            x = [p.loc[(f, v, coh), "near"] for v in vals] + [p.loc[("frozen", "-", coh), "near"]]
            span.append(max(x) - min(x))
        mac(f"span{nm}", max(span))


def cohort_table():
    m = pd.read_csv(REPO_ROOT / "data" / "manifest.csv")
    m = m[m.exclude_reason.isna()]
    # as Table 1 of the paper (CPTAC's LSCC is lung squamous cell carcinoma, written LUSC in the paper)
    rows = [("CPTAC", "CPTAC", "LUAD, LUSC, GBM"), ("TCGA", "TCGA", "LUAD, LUSC, GBM"),
            ("PLCO", "PLCO", "LUAD, LUSC"), ("EBRAINS", "EBRAINS", "GBM")]
    lines = ["Cohort & ID classes & ID & Near & Far \\\\", "\\midrule"]
    targets = 0
    for name, coh, cls in rows:
        x = m[(m.cohort == coh) & (m.diagnosis != "IHC")]
        n = x.role.value_counts()
        targets += len(x) if coh != SOURCE else 0
        lines.append(f"{name} & {cls} & {n['id']} & {n['near']} & {n['far']} \\\\")
    (TAB / "cohorts.tex").write_text(tabular(lines, "lcrrr"))
    mac("nTargetSlides", f"{targets:,}".replace(",", "{,}"))


def diagnostics():
    d = pd.read_csv(R / "final_diagnostics.csv").set_index(["cohort", "class"])
    # "neither lung class reaches the candidate-count threshold" on EBRAINS; "GBM is correctly excluded" on PLCO
    assert all(d.loc[("EBRAINS", k), "candidates"] < SETTINGS["n_min"] for k in (0, 1))
    assert not d.loc[("PLCO", 2), "present"]
    mac("ebGbmCand", str(int(d.loc[("EBRAINS", 2), "candidates"])))
    mac("ebGbmSel", str(int(d.loc[("EBRAINS", 2), "selected"])))
    mac("ebGbmPurity", f"{100 * d.loc[('EBRAINS', 2), 'purity']:.1f}")
    present = d[d.present & d.index.get_level_values(0).isin(TARGETS)]
    mac("selFar", str(int(d.loc[d.index.get_level_values(0).isin(TARGETS), "sel_far"].sum())))
    mac("selNear", str(int(d.loc[d.index.get_level_values(0).isin(TARGETS), "sel_near"].sum())))
    mac("selTotal", str(int(present.selected.sum())))


def mnn_anchors():
    """Share of OOD slides among the target slides in MNN pairs (counted per pair)."""
    model = load_source_model()
    for coh in ("PLCO", "EBRAINS"):
        c = load_cohort(coh)
        j, _ = B.mnn_pairs(model, c.x)
        mac(f"mnnOodShare{TAGS[coh]}", f"{100 * (c.role[j] != 'id').mean():.0f}")


def heldout():
    h = pd.read_csv(R / "final_heldout.csv")
    a = h.groupby(["cohort", "method"]).knn_near_auroc.mean()
    for coh in TARGETS:
        mac(f"hoNone{TAGS[coh]}", a[(coh, "V0 none")])
        mac(f"hoHive{TAGS[coh]}", a[(coh, OURS)])
    mac("hoSplits", str(h.rep.nunique()))


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    ten = ten_detectors()
    main_table()
    ablation_table(ten)
    cohort_table()
    diagnostics()
    heldout()
    mnn_anchors()
    if MANUSCRIPT.exists():
        used = set(re.findall(r"\\([A-Za-z]+)", MANUSCRIPT.read_text())) & set(macros)
        assert used == set(macros), f"macros not used by the paper: {sorted(set(macros) - used)}"
    (OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    print(f"{len(macros)} macros; tables in {TAB.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
