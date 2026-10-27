# Calibrating survival probabilities for myeloid leukaemia patients

M2 data-science project: a reproduction of [Niculescu-Mizil and Caruana, *Predicting Good Probabilities with Supervised Learning* (ICML 2005)](https://doi.org/10.1145/1102351.1102430) on a survival problem. The question is whether the paper's findings on post-hoc calibration (Platt scaling, isotonic regression) still hold when the probabilities to calibrate are risks of death estimated from clinical and molecular data.

To use standard classifiers despite censoring, the problem is recast as **pairwise classification**: for two comparable patients *i* and *j*, a classifier predicts the probability that *i* dies before *j*. Four classifiers (logistic regression, random forest, XGBoost, gradient boosting) are trained, calibrated and compared with proper scoring rules. As an extension, the calibrated pairwise probabilities are combined into one global ranking of the patients with a **Minimum Weight Feedback Arc Set (MWFAS)** algorithm.

## Data

| Table | Content | Size |
|---|---|---|
| `data/clinical_train.csv` | Blood counts (WBC, ANC, monocytes, HB, PLT), bone-marrow blasts, karyotype (ISCN string), treatment centre | 3,323 patients |
| `data/molecular_train.csv` | One row per somatic mutation: gene, chromosome, effect, VAF, read depth | 10,935 mutations |
| `data/target_train.csv` | Overall survival in years (`OS_YEARS`) and death indicator (`OS_STATUS`) | 3,323 patients |
| `test/X_test/` | Same clinical and molecular tables, without outcomes | 1,193 patients |

The data come from a public data challenge on overall-survival prediction for patients with myeloid leukaemia and are not included in this repository (see [Getting the data](#getting-the-data)).

## Method

1. **Feature engineering** (`src/dataprocessor.py`)
   - Karyotypes are mapped to a risk category (normal / intermediate / adverse) with rules inspired by ELN 2022.
   - Mutations are aggregated per patient: binary indicators for genes, chromosomes and mutation effects, plus a `DANGER` score (sum of VAF × depth).
2. **Feature selection** (`src/func_utils.py`): Pearson correlation and chi² tests against survival time, then mutual-information filtering to remove redundant variables.
3. **Patient-level split** into train / calibration / test (60 / 20 / 20). Pairs are built *within* each set, so no patient appears in two sets.
4. **Pair construction**: only pairs whose order is known despite censoring are kept (both deceased, or the earlier death is observed). Each pair is the concatenation of the two patients' features.
5. **Models** (`src/modeltrainer.py`): logistic regression, random forest, XGBoost and gradient boosting, each tuned by cross-validated search on ROC-AUC.
6. **Calibration** (`src/calibration_utils.py`): Platt scaling and isotonic regression, fitted on the calibration set.
7. **Ranking, extension** (`src/mwfas.py`): the calibrated pairwise probabilities define a weighted tournament graph. Removing a minimum-weight set of edges to break every cycle gives a DAG whose topological order is the final ranking of the test patients.

## Results

Test-set performance on patient-disjoint pairs (56,938 comparable pairs; ROC-AUC on comparable pairs is a pairwise concordance measure):

| Model | ROC-AUC | Accuracy | Brier score | Log loss |
|---|---|---|---|---|
| Logistic regression | **0.776** | **0.706** | **0.193** | **0.568** |
| Random forest | 0.770 | 0.703 | 0.195 | 0.573 |
| XGBoost | 0.761 | 0.693 | 0.209 | 0.628 |
| Gradient boosting | 0.760 | 0.695 | 0.211 | 0.646 |

Effect of post-hoc calibration on the log loss (test pairs):

| Model | Before | Platt scaling | Isotonic regression |
|---|---|---|---|
| Logistic regression | 0.568 | 0.569 | 0.571 |
| Random forest | 0.573 | 0.574 | 0.575 |
| XGBoost | 0.628 | 0.586 | 0.581 |
| Gradient boosting | 0.646 | 0.585 | 0.581 |

Comparison with the paper:

- **Reproduced:** the boosted trees are the most miscalibrated models and benefit most from calibration. Logistic regression and random forest are already well calibrated, and calibrating them slightly degrades their log loss, as the paper reports for well-calibrated methods. With a large calibration set (about 57,000 pairs), isotonic regression is slightly better than Platt scaling. ROC-AUC is unchanged, as expected from monotonic transformations.
- **Not reproduced:** in the paper, boosted trees and random forests give the best probabilities after calibration. Here logistic regression remains the best model, before and after calibration.

## Repository structure

```
├── 00_data_processing.ipynb        feature engineering, selection, patient split, pair generation
├── 01_impute_missing_values.sh     median imputation fitted on the training pairs (runs src/clean_nans.py)
├── 02_logistic_regression.ipynb
├── 03_xgboost.ipynb
├── 04_random_forest.ipynb
├── 05_gradient_boosting.ipynb
├── 06_platt_calibration.ipynb
├── 07_isotonic_calibration.ipynb
├── 08_mwfas_ranking_test_set.ipynb test-set pipeline and final MWFAS ranking
├── src/
│   ├── dataprocessor.py            DataProcessor: loading, merging, cytogenetics, gene aggregation
│   ├── func_utils.py               feature selection, patient split, pair generation
│   ├── clean_nans.py               imputation script
│   ├── modeltrainer.py             ModelML: training, tuning, saving and loading models
│   ├── eval_utils.py               metrics and evaluation plots
│   ├── calibration_utils.py        Platt and isotonic calibration
│   └── mwfas.py                    MWFAS ranking and upset losses
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
2. `./01_impute_missing_values.sh` (or `python src/clean_nans.py` on Windows)
3. `02` to `05`: train and evaluate the four models
4. `06` and `07`: calibration
5. `08_mwfas_ranking_test_set.ipynb`: final ranking of the test patients

Each model notebook reuses a model saved in `models/` if one exists, so re-running is fast after the first training.

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
