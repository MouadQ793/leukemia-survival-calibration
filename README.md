# Calibrating survival probabilities for myeloid leukaemia patients

M2 data-science project: a reproduction of [Niculescu-Mizil and Caruana, *Predicting Good Probabilities with Supervised Learning* (ICML 2005)](https://doi.org/10.1145/1102351.1102430) on a survival problem. The question is whether the paper's findings on post-hoc calibration (Platt scaling, isotonic regression) still hold when the probabilities to calibrate are risks of death estimated from clinical and molecular data.

To use standard classifiers despite censoring, the problem is recast as **pairwise classification**: for two comparable patients *i* and *j*, a classifier predicts the probability that *i* dies before *j*. Four classifiers (logistic regression, random forest, XGBoost, gradient boosting) are trained, calibrated and compared with proper scoring rules. As an extension, the calibrated pairwise probabilities are combined into one global ranking of the patients with a **Minimum Weight Feedback Arc Set (MWFAS)** algorithm, and this ranking is compared with standard survival models (Cox and elastic-net Cox).

## Data

| Table | Content | Size |
|---|---|---|
| `data/clinical_train.csv` | Blood counts (WBC, ANC, monocytes, HB, PLT), bone-marrow blasts, karyotype (ISCN string), treatment centre | 3,323 patients |
| `data/molecular_train.csv` | One row per somatic mutation: gene, chromosome, effect, VAF, read depth | 10,935 mutations |
| `data/target_train.csv` | Overall survival in years (`OS_YEARS`) and death indicator (`OS_STATUS`) | 3,323 patients, 150 without outcome |
| `test/X_test/` | Same clinical and molecular tables, without outcomes | 1,193 patients (one centre, absent from training) |

The data come from a public data challenge on overall-survival prediction for patients with myeloid leukaemia and are not included in this repository (see [Getting the data](#getting-the-data)).

## Method

1. **Feature engineering** (`src/dataprocessor.py`), identical for every patient and blind to the outcome
   - Karyotypes are mapped to a risk category (normal / intermediate / adverse / missing) with rules inspired by ELN 2022.
   - Mutations are aggregated per patient: binary indicators for genes, chromosomes and mutation effects, plus a `DANGER` score (sum of the variant allele fractions). Patients without any mutation are kept, with every indicator at 0.
2. **Patients without an outcome are removed** (150), leaving 3,173 patients (50.4% deceased).
3. **Patient-level split** into train / calibration / test (60 / 20 / 20, stratified on the death indicator).
4. **Feature selection on the training patients only** (`src/func_utils.py`): Pearson correlation and chi² tests against survival time, then mutual-information filtering of redundant variables. 41 features are kept.
5. **Imputation and pairs** (`src/impute_and_pair.py`): median imputation fitted on the training patients, then pairs built *within* each set. Only pairs whose order is known despite censoring are kept (two deaths at different times, or a death before the other patient's censoring). Each pair is the concatenation of the two patients' features, in a random orientation: 437,370 training, 54,435 calibration and 52,176 test pairs.
6. **Models** (`src/modeltrainer.py`): logistic regression (on standardised features), random forest, XGBoost and gradient boosting, each tuned on ROC-AUC by **patient-disjoint cross-validation** (`patient_disjoint_cv`).
7. **Calibration** (`src/calibration_utils.py`): Platt scaling and isotonic regression, fitted on the calibration pairs and evaluated on the test pairs.
8. **Ranking, extension** (`src/mwfas.py`): the calibrated pairwise probabilities (averaged over both orientations of each pair) define a weighted tournament. The ranking is the vertex order whose backward edges (the feedback arc set) have minimum total weight, found by a net-score start and local search.
9. **Baselines** (`src/coxnet.py`): a Cox proportional-hazards model, without penalty and with an elastic-net penalty (CoxNet, the objective of glmnet and scikit-survival's `CoxnetSurvivalAnalysis`), fitted directly on the same training patients and features; the CoxNet penalty is chosen by cross-validation over patients.

## Leakage controls

The evaluation is designed so that no information about the calibration or test patients reaches the models:

- **No patient is shared between sets.** Pairs are built inside each set; a patient appears in hundreds of pairs, so a split at the pair level would leak patients.
- **Cross-validation over patients, not pairs.** Each fold validates on pairs of held-out patients only. A plain KFold over pairs gave CV ROC-AUCs of 0.98–0.996 for the tree models (against about 0.76 on the test set) and selected the most over-fitted settings; with patient-disjoint folds, CV and test scores agree within 0.006.
- **Everything fitted on data is fitted on training patients only**: feature selection, imputer, models, hyperparameters and the Cox baselines. The calibrators use the calibration set; the test set is used for evaluation only. The model used for the ranking is chosen by CV score, not test score.
- **The test set goes through the same code as the training set** (`build_patient_features`), and its columns are aligned by name to the training features.
- **No centre-specific technical variables.** The treatment centre is excluded, and the mutation score does not use sequencing depth, which depends on each centre's panel (median 621 to 1,496 reads across training centres, 493 in the test centre).

## Results

### Pairwise classifiers

Test-set performance on patient-disjoint pairs (52,176 comparable pairs from 635 patients). ROC-AUC on comparable pairs is a pairwise concordance measure.

| Model | Selected hyperparameters | CV ROC-AUC | Test ROC-AUC | Accuracy | Brier score | Log loss |
|---|---|---|---|---|---|---|
| Logistic regression | C = 10⁻⁴, balanced | **0.817** | 0.811 | **0.736** | 0.178 | 0.531 |
| Random forest | 200 trees, depth 10 | 0.815 | 0.810 | 0.735 | 0.182 | 0.545 |
| XGBoost | 100 trees, depth 3 | 0.809 | **0.812** | 0.736 | **0.177** | **0.529** |
| Gradient boosting | 100 trees, depth 3 | 0.811 | 0.811 | 0.735 | 0.177 | 0.530 |

### Calibration

Effect of post-hoc calibration on the log loss (test pairs):

| Model | Before | Platt scaling | Isotonic regression |
|---|---|---|---|
| Logistic regression | 0.531 | 0.533 | 0.532 |
| Random forest | 0.545 | **0.532** | **0.532** |
| XGBoost | **0.529** | 0.530 | 0.529 |
| Gradient boosting | 0.530 | 0.533 | 0.532 |

Comparison with the paper:

- **Reproduced:** the random forest is under-confident (its probabilities are pulled towards 0.5) and is the model that calibration helps; both methods bring it level with the other models. Calibration slightly degrades models that are already well calibrated. Isotonic regression is marginally better than Platt scaling for every model. ROC-AUC is unchanged by calibration, as expected from monotonic transformations.
- **Not reproduced:** in the paper, boosted trees are the most miscalibrated models. Here XGBoost and gradient boosting are well calibrated before any correction. Two differences likely explain this. The paper's boosting is AdaBoost, which pushes predictions away from 0 and 1, while both libraries used here minimise the log loss directly. And patient-disjoint cross-validation selects small models (100 trees of depth 3). An earlier version of this pipeline, tuned with a cross-validation that leaked patients between folds, selected deep boosted models (depth 7–8, 200 trees) that were strongly over-confident (test log loss 0.63–0.65) and that calibration improved, as in the paper. Differences between the four models are small (0.529–0.545 in log loss).

### Ranking and survival baselines

Internal test set (635 patients), Harrell's C-index. Intervals come from a bootstrap over patients (1,000 resamples); the differences use a paired bootstrap (same resampled patients for both scores).

| Risk score | C-index (95% CI) | Difference vs CoxNet (95% CI) |
|---|---|---|
| Cox (no penalty) | 0.730 [0.703, 0.757] | −0.004 [−0.009, +0.001] |
| CoxNet (elastic-net Cox) | 0.734 [0.707, 0.761] | reference |
| Pairwise logistic regression + MWFAS | 0.737 [0.710, 0.763] | +0.003 [−0.003, +0.009] |
| Pairwise XGBoost + MWFAS | 0.738 [0.711, 0.766] | +0.004 [−0.009, +0.018] |

- **Pairwise vs survival models:** the pairwise rankings are on par with Cox and CoxNet; every difference is far smaller than the uncertainty from the 635 test patients. The penalty helps the Cox model slightly: without it, the model fits the training patients better (C-index 0.746) than new ones. The three linear models give almost the same ranking (Spearman correlation 0.98–0.99); XGBoost ranks patients differently (0.91–0.93) for a similar C-index. The value of the pairwise approach lies in its calibrated pairwise probabilities, not in better ranking.
- **MWFAS:** the tournaments are almost transitive. With logistic regression the tournament has no cycle at all (a linear model on concatenated features ranks patients by a single linear score), and with the tree models the feedback arc set holds about 0.02% of the edge weight. MWFAS therefore does not improve on ranking by net score (identical C-index of 0.737 for logistic regression) with these models.

## Limitations

- Feature selection uses correlation with `OS_YEARS`, which ignores censoring (a censored time is a lower bound on survival). A censoring-aware criterion, such as univariate Cox models, would be more appropriate.
- The calibration and test sets each contain about 54,000 pairs, but only 635 patients; pairs that share a patient are not independent, so differences in the third decimal should not be over-interpreted. The ranking comparison includes bootstrap intervals over patients; the pairwise metrics come from one split, and repeated splits would give intervals for them too.
- Harrell's C-index is used; the challenge scores an IPCW C-index truncated at 7 years.
- The challenge test set comes from a single centre absent from training, with different karyotype conventions and case mix, so performance there may differ from the internal test set.

## Repository structure

```
├── 00_data_processing.ipynb        feature engineering, patient split, feature selection (training patients)
├── 01_impute_and_pair.sh           imputation fitted on training patients + pair building (runs src/impute_and_pair.py)
├── 02_logistic_regression.ipynb
├── 03_xgboost.ipynb
├── 04_random_forest.ipynb
├── 05_gradient_boosting.ipynb
├── 06_platt_calibration.ipynb
├── 07_isotonic_calibration.ipynb
├── 08_mwfas_ranking_test_set.ipynb ranking validation, Cox / CoxNet comparison, final MWFAS ranking of the challenge test set
├── src/
│   ├── dataprocessor.py            DataProcessor and build_patient_features: cytogenetics, gene aggregation
│   ├── func_utils.py               feature selection, patient split, pairs, patient-disjoint CV, C-index
│   ├── impute_and_pair.py          imputation and pair-building script
│   ├── modeltrainer.py             ModelML: training, tuning, saving and loading models
│   ├── eval_utils.py               metrics and evaluation plots
│   ├── calibration_utils.py        Platt and isotonic calibration
│   ├── coxnet.py                   Cox and elastic-net Cox (CoxNet) baselines
│   └── mwfas.py                    MWFAS ranking, upset losses
└── requirements.txt
```

The challenge data (`data/`, `test/`) and the generated folders (`models/`, `imputer/`, `model_calibrated*/`, `results/`, `submissions/`) are not tracked. The generated folders are created when the notebooks run.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run, from the repository root and in order:

1. `00_data_processing.ipynb`
2. `./01_impute_and_pair.sh` (or `python src/impute_and_pair.py` on Windows)
3. `02` to `05`: tune, train and evaluate the four models (gradient boosting is the slowest, about 1 hour on 2 cores)
4. `06` and `07`: calibration
5. `08_mwfas_ranking_test_set.ipynb`: ranking validation, comparison with Cox and CoxNet (`results/ranking_comparison.csv` and `.png`), and final ranking of the challenge test patients (`submissions/patient_rankings.csv`, higher `risk_score` = higher risk)

Each model notebook reuses a model saved in `models/` only if it was trained on exactly the current features; otherwise it retrains.

## Getting the data

The data comes from the challenge **"Overall survival prediction for patients diagnosed with myeloid leukemia"**, organised by QRT and Gustave Roussy on the ENS Challenge Data platform: https://challengedata.ens.fr

It is not included in this repository. To run the pipeline, create an account on the platform, download the challenge files and place them as follows:

    data/clinical_train.csv
    data/molecular_train.csv
    data/target_train.csv
    test/X_test/clinical_test.csv
    test/X_test/molecular_test.csv

## Author

Mouad Sehli