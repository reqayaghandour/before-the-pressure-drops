# Before the Pressure Drops

Code and results for the MSc Data Science & Society thesis
**"Before the Pressure Drops: Predicting Intraoperative Hypotension at Surgical Start Using
Machine Learning and Identifying Whom the Model Misses"** (Reqaya Ghandour, Tilburg University, 2026;
supervisor dr. Gonzalo Nápoles, second reader dr. Görkem Saygili).

The thesis predicts intraoperative hypotension (IH), defined as an invasive mean arterial
pressure (MAP) below 65 mmHg for at least 60 seconds between surgical start and end, at the
start of surgery, using 3,468 surgical cases from the VitalDB open dataset.

| Research question | What is done | Main scripts |
|---|---|---|
| **RQ1** Model comparison | LR, random forest, XGBoost and a neural network on two feature sets, repeated nested cross-validation (10 × 5 outer × 5 inner folds), corrected repeated CV t-test | `run_model.py`, `compare_models.py`, `agreement.py` |
| **RQ2** Feature importance | Feature permutation importance, exact linear SHAP, MORF/LERF removal analysis, SOFI | `rq2_feature_importance.py`, `rq2_sofi.py` |
| **RQ3** Missed IH per subgroup | False negative rate (FNR) per subgroup (sex, ASA class, department, operation type) | `rq3_subgroups.py` |

Feature sets:
- **primary**: 27 preoperative features (demographics, ASA class, surgical characteristics, comorbidities, ECG, pulmonary function test and 12 laboratory values);
- **extended**: primary + three summaries of the invasive MAP in the last 5 minutes before surgical start (mean, lowest 60-second rolling mean, % of time below 65 mmHg).

## Data

The data come from **VitalDB** (Lee et al., 2022, *Scientific Data* 9:279), a public perioperative
database from Seoul National University Hospital, available under the
[Creative Commons Attribution 4.0 International licence](https://physionet.org/content/vitaldb/1.0.0/).
Its collection and release were approved by the Institutional Review Board of Seoul National
University Hospital (H-1408-101-605), and the data are de-identified.

The data are **not included** in this repository. They are downloaded by the scripts:
the clinical table with `download_data.py` and the arterial pressure tracks, case by case,
with the `vitaldb` Python package. All files are written to `data/raw/` and `data/processed/`.

## Repository structure

```
src/        analysis scripts (run from this folder)
figures/    scripts that redraw the thesis figures from results/, and the figures (output/)
results/    all results used in the thesis (CSV files)
data/       created by the scripts, not included (see above)
```

## Installation

```
pip install -r requirements.txt
```

## How to reproduce the analysis

Run the scripts from the `src/` folder, in this order. Steps 1 and 3 download the arterial
pressure track of every case and take several hours; the model runs in step 5 take from
minutes (LR) to many hours (XGBoost, NN). All scripts can be restarted after a crash.

| Step | Script | What it does | Output | Thesis |
|---|---|---|---|---|
| 0 | `download_data.py` | Download the clinical table of all 6,388 cases | `data/raw/clinical_data.csv` | 4.1 |
| 1 | `build_ih_target.py` | Build the outcome (IH) from the invasive MAP | `data/processed/ih_target.csv` | 4.2 |
| 2 | `merge_target_clinical.py` | Merge outcome and clinical data | `data/processed/analysis_dataset.csv` | 4.3 |
| 3 | `check_preopstart_and_target.py` | Pre-start MAP features and coverage, artefact check and recomputation of the outcome | `data/processed/preopstart_target_check.csv` | 4.2, 4.5 |
| – | `preprocessing.py` | Module: cohort selection (adults, no ASA 6), feature sets and preprocessing pipeline; run it directly for a quick check | – | 4.3, 4.5 |
| 4 | `eda_table.py`, `eda_correlation.py`, `eda_tsne_mi.py` | Exploratory data analysis | `results/eda/` | 4.4 |
| 5 | `run_model.py <model> <feature_set>` | Repeated nested CV for `lr`, `rf`, `xgb`, `nn` × `primary`, `extended` (8 runs) | `results/<model>_<feature_set>/` | 4.6–4.8, 5.1 |
| 6 | `compare_models.py` | Corrected repeated CV t-test with Bonferroni correction | `results/summary_table.csv`, `results/comparisons_roc_auc.csv` | 4.8, 5.1 |
| 7 | `agreement.py` | Agreement between the predictions of the four models | `results/agreement.csv` | 4.8, 5.1 |
| 8 | `count_params.py` | Selected hyperparameter values per model | printed (Appendix A) | 5.1 |
| 9 | `rq2_feature_importance.py <feature_set>` | Feature permutation importance, SHAP, MORF/LERF | `results/rq2/` | 4.9, 5.2 |
| 10 | `rq2_sofi.py <feature_set>` | SOFI ranking and comparison with SHAP | `results/rq2/sofi_*` | 4.9, 5.2 |
| 11 | `export_feature_values.py` | Feature values for the SHAP beeswarm figure | `results/rq2/feature_values_*` | 5.2 |
| 12 | `rq3_subgroups.py <feature_set>` | FNR, FPR and within-subgroup ROC-AUC per subgroup | `results/rq3/` | 4.10, 5.3 |
| 13 | `../figures/make_all.py` | Redraw Figures 1–9 at thesis width | `figures/output/` | all |

Additional checks used to support decisions in Section 4 (not needed to reproduce the results):
`check_art.py` (availability of invasive MAP), `check_other_invasive_map.py` (other invasive
MAP tracks) and `check_target_exclusion.py` (cases without a valid outcome).

### Notes on the design
- **No leakage**: all preprocessing (imputation, scaling, encoding, merging of rare categories) is fitted on the training folds only, inside a scikit-learn pipeline. Folds are grouped by patient, so no patient appears in both training and test data.
- **Identical folds**: all models and feature sets use the same outer folds (seed 42 + repetition), which allows paired comparisons.
- **Hyperparameter grids** (`run_model.py`, Table 2 of the thesis) contain the software default together with lower and higher values. When a boundary value of a grid was selected in most folds of an initial run, the grid was extended beyond that value and the model was rerun; the results in this repository are from the final grids.
- **Selected model**: as no model performed significantly better than the others, LR (`C = 0.1`, the value selected in most folds) was used for RQ2 and RQ3.

## Results

`results/` contains all CSV files used for the tables and figures in the thesis:
`summary_table.csv` and `comparisons_roc_auc.csv` (RQ1), per model and feature set the
`fold_scores.csv`, `predictions.csv` and `importances.csv`, and the folders `eda/`, `rq2/`
and `rq3/`. The predictions contain VitalDB case IDs only, no other patient information.

## Use of AI tools

Generative language models (ChatGPT and Claude) were used to obtain feedback, explain
methodological concepts and assist with writing and debugging code. All analyses and
decisions were made and checked by the author.
