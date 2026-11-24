# Metadata sources

| File | Source | Retrieved | SHA-256 |
|---|---|---|---|
| `cptac_metadata_07-09-2024.csv` | TCIA CPTAC Pathology Portal, https://pathdb.cancerimagingarchive.net/system/files/collectionmetadata/202407/cptac_metadata_07-09-2024.csv | 2026-10-26 | `a4f86effa438352acadbf7a1f61be16a2cb5538ae8d9dda8d35e0677ab064a27` |
| `plco_rare_lung.csv` | PLCO lung clinical data (NCI CDAS) for the 115 rare lung carcinoma slides, provided with the PLCO TITAN embeddings | 2026-10-26 | `8ae64d39724686e2013b48b44a6090942185137b6ec8b43b9f13bbc64f36cb3d` |

Fields used:
- CPTAC: `Slide_ID`, `Case_ID`, `Tumor`, `Specimen_Type` (tumor_tissue / normal_tissue), `Embedding_Medium` (FFPE / OCT = frozen), `Percent_Tumor_Nuclei`.
- PLCO: `lung_heslide_img_name`, `plco_id`, `lung_cancer_subtype`.
