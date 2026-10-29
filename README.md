# MIND-AD: Multimodal In-Context Learning for Alzheimer's Disease Classification

This is the code for the manuscript *"MIND-AD: Multimodal In-Context Learning for Alzheimer's Disease Classification"*
by P. Shanmugasundaram and S. Saroja, Department of Computer Applications, National Institute of Technology
Tiruchirappalli, India (submitted to IEEE Journal of Biomedical and Health Informatics).

Corresponding author: S. Saroja (saroja@nitt.edu).

MIND-AD names the proposed pipeline and its evaluation protocol. It is not a new foundation model: the pre-trained
TabPFN v2 weights are used unchanged.

---

## Overview

**Task.** Cross-sectional classification of cognitively normal participants (CDR = 0) versus clinically characterized
very mild to moderate Alzheimer's disease (CDR > 0), aged ≥ 60 years, from OASIS-1 (N = 166: 85 vs 81).
- Inputs: structural-MRI features plus 8 clinical variables (age, sex, education, SES, eTIV, nWBV, ASF, MMSE).
- CDR defines the target and is never used as a predictor.
- The endpoint is CDR-defined, not a biomarker-confirmed AD diagnosis.

**Framework.**
1. **Study-label-free MRI representation.** A frozen ImageNet ResNet18 is applied to 9 fixed slices of the
   atlas-registered T1 volume, giving 512 features per subject. The network was never fitted or fine-tuned with OASIS
   diagnosis labels.
2. **Leakage-safe preprocessing.** Imputation and standardization are fitted on training subjects only.
3. **Adaptive PCA.** The MRI block is compressed by PCA, with k ∈ {8, 16, 24, 32, 64} chosen by inner cross-validation.
4. **Early fusion** with the 8 clinical variables.
5. **TabPFN v2 in-context learning**, with no cohort-specific weight training.

**Evaluation.**
- Frozen subject-level 5 × 5 repeated stratified nested cross-validation (25 outer folds).
- Primary analysis: Wilcoxon tests with Holm correction within each family.
- Dependence-aware sensitivity analyses: Nadeau–Bengio corrected t-test, global Holm over 84 tests and a subject-level
  bootstrap.

## Main results

Mean over the 25 outer test folds:

| Model | Balanced acc. | Macro-F1 | MCC | ROC-AUC | Brier |
|---|---|---|---|---|---|
| MIND-AD (8 clinical + MRI) | 0.812 | 0.812 | 0.635 | 0.887 | 0.135 |
| Clinical-only TabPFN | 0.795 | 0.794 | 0.600 | 0.892 | 0.135 |
| Fusion CatBoost (best conventional fusion) | 0.801 | 0.800 | 0.609 | 0.877 | 0.152 |

- **Calibration:** slope 0.97.
- **Against fusion learners:** in multiplicity-adjusted comparisons, the Brier score was lower than all six
  conventional fusion learners. AUC was higher than several fusion learners for a multi-view variant, with a second MRI
  representation and in an unfiltered cohort. Fewer differences remained significant under the dependence-aware
  analyses.
- **Against clinical-only models:** discrimination was comparable.
- **Locked external test:** the transportable configuration (MRI, age, sex, education, MMSE) was frozen and applied once
  to 29 overlap-screened OASIS-2 participants, using a harmonized raw-MRI pipeline built only from OASIS-1. It reached
  AUC 0.862 [0.70, 0.97] and Brier 0.137. Without the one image-flagged participant, AUC was 0.851. Clinical-only models
  reached 0.900.
- **Feature-provenance audit:** with identical models and folds, third-party Kaggle MRI embeddings yielded MCC 0.868 and
  AUC 0.983, against 0.635 and 0.887 for the documented features. The third-party embeddings encode atrophy less well,
  which points to an embedding-leakage risk; the mechanism cannot be established.
- **Cross-cohort overlap audit:** 83 of 150 OASIS-2 participants probably overlap with OASIS-1 (record linkage checked
  on raw images).

---

## Repository structure

| Folder | Content |
|---|---|
| `pipeline/` | Every analysis script, shared modules (`config.py`, `common.py`) and runners |
| `proposed/` | Minimal, self-contained implementation of the final MIND-AD algorithm (see `proposed/README.md`) |
| `protocols/` | Implementation plan and pre-registered protocol with amendments 1–12 and their outcomes |
| `results/` | Image-overlap audit of the external set (`part53*`): thresholds, per-participant scores and the sensitivity analysis |
| `data/` | Empty placeholders. Put the data here (see below); none is redistributed |

## 1. Installation

Tested with Python 3.13 (Anaconda) on Windows, CPU only.

```bash
pip install -r proposed/requirements.txt
pip install SimpleITK   # harmonized raw-MRI pipeline for the locked external validation (part43)
```

On first use, TabPFN downloads the openly available **model version 2** weights.

## 2. Data (not redistributed)

The data are public but subject to the OASIS Data Use Agreement. Download them yourself:

| File | Source |
|---|---|
| `data/oasis_cross-sectional.csv` | OASIS-1 clinical table: https://sites.wustl.edu/oasisbrains/home/oasis-1/ (Kaggle copy: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers) |
| OASIS-1 imaging discs 1–12 | https://sites.wustl.edu/oasisbrains/home/oasis-1/. `pipeline/download_oasis1.ps1` downloads them resumably. Then run `part28_extract_t88.py` and `part29_mri_features.py` (or `proposed/extract_mri_features.py`). |
| `data/external_oasis2/oasis_longitudinal_demographics.xlsx` | OASIS-2 clinical table and raw scans: https://sites.wustl.edu/oasisbrains/home/oasis-2/ (`part49_fast_download.ps1` downloads the external subset) |
| `mri_embeddings.csv` (audit only) | Third-party 512-D ResNet18 embeddings obtained from Kaggle in 2026. The original page cannot be re-identified and the extraction is undocumented; they are reportedly derived from https://www.kaggle.com/datasets/ninadaithal/imagesoasis. Available from the authors for audit on request, subject to the original dataset terms. |
| `data/external_adni/` | Optional: ADNI files for a future external validation (see the README there; requires your own ADNI access) |

## 3. Running

Run every script from `pipeline/`.
- Each step writes `outputs/<step>/` with its results, `run_log.txt` and `run_manifest.json`.
- The manifest records SHA-256 hashes of the script, config, inputs and outputs, plus parameters and package versions.
- Long steps use a **resumable queue** with per-fold checkpoints:
  ```powershell
  powershell -ExecutionPolicy Bypass -File run_queue.ps1     # or double-click RESUME_QUEUE.bat
  ```

| Script | Content |
|---|---|
| `phase01`–`phase03` | Protocol freeze, cohort audit, frozen folds |
| `part04`–`part25` | Development analyses with the third-party embeddings (not used as evidence) |
| `part28`–`part29` | T88 volume extraction and study-label-free MRI features |
| `part30` | Primary benchmark and feature-provenance audit |
| `part31` | Modality ablation; CDR 0 vs 0.5 (early detection) |
| `part32`–`part34` | Figures, subgroups, subgroup-gap bootstrap |
| `part35`–`part36` | Tissue morphometry (second MRI representation) and its benchmark |
| `part37c`, `part38` | Paper figures (Fig. 1 MIND-AD; Figs. 2–4 and S1) |
| `part39` | MRI beyond non-imaging clinical data; without MMSE |
| `part40` | Unfiltered cohort (N = 198, new folds) |
| `part41` | Permutation importance, calibration slope, global Holm, subgroup calibration |
| `part42`, `part42b`, `part42c` | OASIS-2 record-linkage overlap rule, diagnostics, clinical external check |
| `part43` | Harmonized raw-MRI pipeline (template, mask and parameters from OASIS-1 only) |
| `part44`, `part44b` | Raw-image verification of the OASIS-1/OASIS-2 overlap; post-hoc diagnostic |
| `part45` | Subject-level bootstrap of the main comparisons |
| `part46` | Results-evidence report (reads saved outputs only) |
| `part47`, `part48` | Frozen, hashed (locked) models; locked external validation (OASIS-2; ADNI-ready) |
| `part49`, `part50` | OASIS-2 subset download; harmonized-pipeline fidelity |
| `part51`, `part52` | Pre-registered variants: multi-view MIND-AD; multi-view with inner-CV threshold |
| `part53`, `part53b` | Complete image-overlap audit of the external set; sensitivity without the flagged participant |
| `part54`, `part54b` | Reproducibility of the locked external predictions and metrics |
| `part55` | Verification of every number in the manuscript against the result files |
| `tabpfn_retry.py` | Logged randomized-SVD fallback for TabPFN's internal ARPACK step (8 fits, Part 40 only) |

## 4. Reproducibility notes

- **Seeds:** outer CV uses 42, 2026, 3407, 7319 and 9871; the inner seed is seed × 100 + fold.
- **Fold-wise fitting:** all preprocessing, PCA, tuning and thresholds are fitted inside training folds only.
  - Exception: the morphometry cube mask is an unsupervised, cohort-level selection that uses no labels. The multi-view
    variant inherits it.
- **External test:** the external models were fitted once on OASIS-1, hashed and applied once. Re-applying them
  reproduces every reported external metric.
- **Run-to-run variation:** TabPFN probabilities can differ slightly between runs (≤ 0.016); the reported metrics are
  unchanged.
- **Protocol:** the pre-registered protocol, with every analysis variant and its outcome, is in `protocols/`.

## 5. Citation

If you use this code, please cite the manuscript (citation details will be added after publication):

```
P. Shanmugasundaram and S. Saroja, "MIND-AD: Multimodal In-Context Learning for Alzheimer's Disease
Classification," submitted to IEEE Journal of Biomedical and Health Informatics, 2026.
```

## 6. Acknowledgment

Data were provided by OASIS-1: Cross-Sectional and OASIS-2: Longitudinal: Principal Investigators: D. Marcus,
R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

## 7. Licence

MIT (code only). The data remain subject to the OASIS Data Use Agreement and the Kaggle dataset terms.
