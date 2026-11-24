"""Build data/manifest.csv: one row per slide with its role (README, "Data").

Reads the TITAN embedding folders (read-only) and the metadata in data/meta. Writes data/manifest.csv and
data/manifest_summary.txt.
"""

import hashlib
import io
import re
from contextlib import redirect_stdout

import pandas as pd

from chord.data import MANIFEST_PATH, REPO_ROOT, ROLES, TITAN_ROOT, load_subset

META = REPO_ROOT / "data" / "meta"
SUMMARY_PATH = MANIFEST_PATH.with_name("manifest_summary.txt")
COLUMNS = ["cohort", "slide_id", "patient_id", "site", "diagnosis", "role", "id_class", "specimen_type",
           "medium", "tumor_nuclei_pct", "exclude_reason", "source_subset", "source_row"]


def new_row(cohort, subset, i, slide_id, **fields):
    r = dict(cohort=cohort, slide_id=slide_id, patient_id="", site="", diagnosis="", role="", id_class=-1,
             specimen_type="unknown", medium="unknown", tumor_nuclei_pct="", exclude_reason="",
             source_subset=subset, source_row=i)
    r.update(fields)
    return r


def exclude(r, reason):
    r.update(role="excluded", id_class=-1, exclude_reason=reason)
    return r


# ---------------------------------------------------------------- TCGA
TCGA_BARCODE = re.compile(r"^(TCGA-[0-9A-Z]{2}-[0-9A-Z]{4})-(\d{2})[A-Z]-\d{2}-(DX|TS|BS|MS)[0-9A-Z]*\.")
TCGA_SOURCES = [("LUAD", "id", 0), ("LUSC", "id", 1), ("GBM", "id", 2), ("LGG", "near", -1), ("HNSC", "near", -1),
                *[(p, "far", -1) for p in ("BRCA", "COAD", "KIRC", "KIRP", "SKCM", "PAAD")]]


def tcga_rows():
    for project, role, cls in TCGA_SOURCES:
        subset = f"TCGA/{project}"
        for i, s in enumerate(load_subset(subset)[1]):
            r = new_row("TCGA", subset, i, s, diagnosis=project, role=role, id_class=cls)
            m = TCGA_BARCODE.match(s)
            if not m:
                yield exclude(r, "unparsed_barcode")
                continue
            patient, sample, slide_type = m.groups()
            r.update(patient_id=patient, site=patient[5:7], medium="FFPE" if slide_type == "DX" else "frozen",
                     specimen_type="tumor" if int(sample) < 10 else "normal")
            if slide_type != "DX":
                yield exclude(r, "frozen_section")
            elif int(sample) >= 10:
                yield exclude(r, "normal_tissue")
            elif role == "id" and sample != "01":
                yield exclude(r, f"sample_type_{sample}")
            else:
                yield r


# ---------------------------------------------------------------- CPTAC
CPTAC_SOURCES = [("LUAD", "id", 0), ("LSCC", "id", 1), ("GBM", "id", 2), ("HNSCC", "near", -1),
                 ("PDA", "far", -1), ("UCEC", "far", -1), ("BRCA", "far", -1), ("COAD", "far", -1)]
CPTAC_NO_METADATA = {"BRCA", "COAD"}  # older CPTAC slide names, absent from the TCIA metadata
SPECIMEN = {"tumor_tissue": "tumor", "normal_tissue": "normal"}
MEDIUM = {"FFPE": "FFPE", "OCT": "frozen"}


def cptac_rows():
    meta = pd.read_csv(META / "cptac_metadata_07-09-2024.csv", dtype=str, keep_default_na=False)
    if not meta["Slide_ID"].is_unique:
        raise ValueError("CPTAC metadata: duplicate Slide_ID")
    meta = meta.set_index("Slide_ID")
    for code, role, cls in CPTAC_SOURCES:
        subset = f"CPTAC/{code}"
        for i, s in enumerate(load_subset(subset)[1]):
            r = new_row("CPTAC", subset, i, s, diagnosis=code, role=role, id_class=cls)
            if code in CPTAC_NO_METADATA:
                r.update(patient_id=s.split("-")[0])
                yield r
                continue
            if s not in meta.index:
                yield exclude(r, "no_metadata")
                continue
            m = meta.loc[s]
            r.update(patient_id=m["Case_ID"], specimen_type=SPECIMEN.get(m["Specimen_Type"], "unknown"),
                     medium=MEDIUM.get(m["Embedding_Medium"], "unknown"), tumor_nuclei_pct=m["Percent_Tumor_Nuclei"])
            if m["Tumor"] != code:
                yield exclude(r, "metadata_tumor_mismatch")
            elif r["medium"] == "frozen":
                yield exclude(r, "frozen_section")
            elif r["medium"] != "FFPE":
                yield exclude(r, "unknown_medium")
            elif r["specimen_type"] == "unknown":
                yield exclude(r, "unknown_specimen_type")
            elif r["specimen_type"] == "normal":
                yield exclude(r, "normal_tissue")
            else:
                yield r


# ---------------------------------------------------------------- PLCO
PLCO_NAME = re.compile(r"^Slide-\d+_(TI\d+) \d+$")
# RAREEE: the 115 rare lung carcinoma slides (near-OOD).
# IHC (immunohistochemistry, not H&E) is an additional far-OOD test set, kept separable via diagnosis "IHC".
PLCO_SOURCES = [("LUAD", "id", 0, "LUAD"), ("LUSC", "id", 1, "LUSC"), ("RAREEE", "near", -1, None),
                ("BRCA", "far", -1, "BRCA"), ("COAD", "far", -1, "COAD"), ("IHC", "far", -1, "IHC")]
PLCO_IHC_CONTROL = re.compile(r"^Slide-\d+_(PosCtrl|NegCtrl)_")


def plco_rare_table():
    rare = pd.read_csv(META / "plco_rare_lung.csv", dtype=str, keep_default_na=False)
    rare["slide_id"] = rare["lung_heslide_img_name"].str.removesuffix(".svs")
    if not rare["slide_id"].is_unique:
        raise ValueError("PLCO rare-lung table: duplicate slide names")
    return rare.set_index("slide_id")


def plco_rows():
    rare = plco_rare_table()
    for sub, role, cls, dx in PLCO_SOURCES:
        subset = f"PLCO/{sub}"
        for i, s in enumerate(load_subset(subset)[1]):
            if dx is None and s not in rare.index:
                raise ValueError(f"PLCO {sub}: {s} missing from the rare-lung table")
            diagnosis = dx if dx else rare.loc[s, "lung_cancer_subtype"]
            r = new_row("PLCO", subset, i, s, diagnosis=diagnosis, role=role, id_class=cls)
            if PLCO_IHC_CONTROL.match(s):
                yield exclude(r, "ihc_control_slide")
                continue
            m = PLCO_NAME.match(s)
            if not m:
                yield exclude(r, "unparsed_name")
                continue
            r.update(patient_id=m.group(1))
            yield r


# ---------------------------------------------------------------- EBRAINS
EBRAINS_BASE = "EBRAINS-EXTENDED-NEW"
EBRAINS_GROUPS = [("GBM", "id", 2), ("LGG", "near", -1), ("ID", "far", -1), ("REST", "far", -1)]
EBRAINS_EXCLUDED_DX = {"Metastatic_tumours": "possible_lung_metastasis"}


def ebrains_dx(folder):
    return folder.replace(",_", "_")  # e.g. "Glioblastoma,_IDH-wildtype" == "Glioblastoma_IDH-wildtype"


def ebrains_dx_folders(group):
    return sorted(p.name for p in (TITAN_ROOT / EBRAINS_BASE / group).iterdir() if p.is_dir())


def ebrains_rows():
    for group, role, cls in EBRAINS_GROUPS:
        for folder in ebrains_dx_folders(group):
            subset = f"{EBRAINS_BASE}/{group}/{folder}"
            dx = ebrains_dx(folder)
            for i, s in enumerate(load_subset(subset)[1]):
                r = new_row("EBRAINS", subset, i, s, patient_id=s, diagnosis=dx, role=role, id_class=cls)
                yield exclude(r, EBRAINS_EXCLUDED_DX[dx]) if dx in EBRAINS_EXCLUDED_DX else r


# ---------------------------------------------------------------- checks
def embedding_hash(subset, i):
    return hashlib.sha1(load_subset(subset)[0][i].tobytes()).hexdigest()


def deduplicate(df):
    """Collapse repeated slide ids within a cohort. Identical labels and embeddings: keep one row.
    Different labels or different embeddings: keep one row, marked excluded."""
    df = df.copy()
    df["emb_hash"] = [embedding_hash(s, i) for s, i in zip(df["source_subset"], df["source_row"])]
    drop, report = [], []
    for (cohort, sid), g in df.groupby(["cohort", "slide_id"], sort=False):
        if len(g) == 1:
            continue
        keep = g.index[0]
        drop += list(g.index[1:])
        labels = g[["role", "id_class", "diagnosis"]].drop_duplicates()
        if len(labels) > 1:
            df.loc[keep, ["role", "id_class", "exclude_reason"]] = ["excluded", -1, "conflicting_labels"]
            status = "conflicting_labels: " + " | ".join(f"{a}/{b}" for a, b in zip(g["role"], g["diagnosis"]))
        elif g["emb_hash"].nunique() > 1:
            df.loc[keep, ["role", "id_class", "exclude_reason"]] = ["excluded", -1, "conflicting_embeddings"]
            status = "conflicting_embeddings"
        else:
            status = "identical"
        report.append((cohort, sid, len(g), status))
    df = df.drop(index=drop)
    # different slide ids with byte-identical embeddings: keep the first, exclude the copies
    active = df[df["role"] != "excluded"]
    copies = active.index[active.duplicated(["cohort", "emb_hash"], keep="first")]
    df.loc[copies, ["role", "id_class", "exclude_reason"]] = ["excluded", -1, "duplicate_embedding"]
    return df, pd.DataFrame(report, columns=["cohort", "slide_id", "copies", "status"])


def check_ebrains_folders():
    """Diagnosis subfolders must exactly partition the group-level folders, and ALL = ID + REST + GBM + LGG."""
    union = set()
    for group, _, _ in EBRAINS_GROUPS:
        parts = set()
        for folder in ebrains_dx_folders(group):
            parts |= set(load_subset(f"{EBRAINS_BASE}/{group}/{folder}")[1])
        whole = set(load_subset(f"{EBRAINS_BASE}/{group}")[1])
        print(f"  {group}: group-level {len(whole)} unique, subfolders {len(parts)} unique, "
              f"only-in-group {len(whole - parts)}, only-in-subfolders {len(parts - whole)}")
        union |= parts
    everything = set(load_subset(f"{EBRAINS_BASE}/ALL")[1])
    print(f"  ALL: {len(everything)} unique vs union of groups {len(union)}; "
          f"only-in-ALL {len(everything - union)}, only-in-groups {len(union - everything)}")


def check_plco_patients(df):
    """TI ids are used as PLCO patient ids. Where plco_id is known (rare-lung table), TI and plco_id must map 1:1."""
    rare = plco_rare_table()
    sub = df[(df["cohort"] == "PLCO") & df["slide_id"].isin(rare.index)]
    pairs = pd.DataFrame({"ti": sub["patient_id"].to_numpy(), "plco": rare.loc[sub["slide_id"], "plco_id"].to_numpy()})
    print(f"  slides with known plco_id: {len(pairs)}; TI ids mapping to >1 plco_id: "
          f"{(pairs.groupby('ti')['plco'].nunique() > 1).sum()}; plco_ids mapping to >1 TI: "
          f"{(pairs.groupby('plco')['ti'].nunique() > 1).sum()}")


def report_shared_embeddings(df):
    """Different slide ids with byte-identical embeddings: the kept slide and its excluded copies."""
    copies = df[df["exclude_reason"] == "duplicate_embedding"]
    print(f"  excluded as duplicate_embedding: {len(copies)}")
    for _, x in df[df["emb_hash"].isin(copies["emb_hash"])].groupby("emb_hash"):
        print("   ", list(zip(x["cohort"], x["slide_id"], x["role"], x["diagnosis"])))
    leftover = df[df["role"] != "excluded"].duplicated("emb_hash").sum()
    print(f"  identical embeddings remaining among non-excluded slides (any cohort): {leftover}")


def report_patient_overlap(df):
    """Patients whose slides carry more than one diagnosis (non-excluded rows)."""
    active = df[df["role"] != "excluded"]
    for cohort, g in active.groupby("cohort"):
        n_dx = g.groupby("patient_id")["diagnosis"].nunique()
        multi = n_dx[n_dx > 1].index
        combos = g[g["patient_id"].isin(multi)].groupby("patient_id")["diagnosis"].apply(
            lambda s: "+".join(sorted(set(s)))).value_counts()
        print(f"  {cohort}: {len(multi)} patients with >1 diagnosis" + (f"; {combos.to_dict()}" if len(multi) else ""))


def summarize(df, dups):
    print("== Role counts per cohort")
    print(pd.crosstab(df["cohort"], df["role"], margins=True).to_string())
    print("\n== ID class counts (slides / patients)")
    idr = df[df["role"] == "id"]
    print(pd.concat([pd.crosstab(idr["cohort"], idr["id_class"]).add_prefix("slides_c"),
                     idr.groupby(["cohort", "id_class"])["patient_id"].nunique().unstack(fill_value=0).add_prefix("patients_c")],
                    axis=1).to_string())
    print("\n== Patients per role")
    print(df.groupby(["cohort", "role"])["patient_id"].nunique().unstack(fill_value=0).to_string())
    print("\n== Exclusions by reason")
    ex = df[df["role"] == "excluded"]
    print(pd.crosstab([ex["cohort"], ex["source_subset"].str.split("/").str[:2].str.join("/")],
                      ex["exclude_reason"]).to_string())
    print("\n== Near-OOD diagnoses")
    print(df[df["role"] == "near"].groupby(["cohort", "diagnosis"]).size().to_string())
    print("\n== EBRAINS ID (GBM) subtypes")
    print(df[(df["cohort"] == "EBRAINS") & (df["role"] == "id")]["diagnosis"].value_counts().to_string())
    print("\n== Medium / specimen type of ID rows")
    print(pd.crosstab([idr["cohort"], idr["medium"]], idr["specimen_type"]).to_string())
    print("\n== Duplicated slide ids (within cohort)")
    print(dups.groupby(["cohort", "status"]).size().to_string() if len(dups) else "  none")
    for _, d in dups[dups["status"] != "identical"].iterrows():
        print(f"    {d['cohort']} {d['slide_id']}: {d['status']}")


def main():
    raw = pd.DataFrame([*tcga_rows(), *cptac_rows(), *plco_rows(), *ebrains_rows()], columns=COLUMNS)
    df, dups = deduplicate(raw)
    df = df.reset_index(drop=True)
    assert not df.duplicated(["cohort", "slide_id"]).any()
    assert set(df["role"]) <= set(ROLES)
    assert ((df["role"] == "id") == (df["id_class"] >= 0)).all()
    assert (df.loc[df["role"] == "excluded", "exclude_reason"] != "").all()
    assert (df.loc[df["role"] != "excluded", "exclude_reason"] == "").all()
    assert (df.loc[df["role"] != "excluded", "patient_id"] != "").all()

    buf = io.StringIO()
    with redirect_stdout(buf):
        summarize(df, dups)
        print("\n== EBRAINS folder consistency")
        check_ebrains_folders()
        print("\n== PLCO patient-id check")
        check_plco_patients(df)
        print("\n== Identical embeddings across slide ids")
        report_shared_embeddings(df)
        print("\n== Patients with more than one diagnosis")
        report_patient_overlap(df)
    df.drop(columns="emb_hash").to_csv(MANIFEST_PATH, index=False)
    SUMMARY_PATH.write_text(buf.getvalue())
    print(buf.getvalue())
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(df)} rows) and {SUMMARY_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
