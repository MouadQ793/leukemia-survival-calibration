# Data Science Project - Machine Learning Model Training & Calibration

A machine learning pipeline for training, evaluating, and calibrating classification models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting) with data preprocessing and model calibration techniques.

---

## 📁 Project Structure

### 📓 Notebooks (Execute in Order)

| #   | Notebook                                  | Description                                                      |
| --- | ----------------------------------------- | ---------------------------------------------------------------- |
| 0   | `0 - Data treatement and processing.ipynb` | Load raw data, merge datasets, feature engineering, split data   |
| 1   | `1 - run_clean_nans.sh`                   | Bash script to run NaN imputation (see note below for non-Linux) |
| 2   | `2 - Logistic regression.ipynb`           | Train and tune Logistic Regression model                         |
| 3   | `3 - XGBoost.ipynb`                       | Train and tune XGBoost model                                     |
| 4   | `4 - Random Forest.ipynb`                 | Train and tune Random Forest model                               |
| 5   | `5 - Gradient Boosting DT.ipynb`          | Train and tune Gradient Boosting model                           |
| 6   | `6 - Plat Calibration.ipynb`              | Apply Platt Scaling calibration to all models                    |
| 7   | `7 - Isotonic Calibration.ipynb`          | Apply Isotonic Regression calibration to all models              |

> ⚠️ **Important**: Execute notebooks **in numerical order** (0 → 1 → 2 → ... → 7). Each step depends on outputs from previous steps.

---

### Note for Non-Linux Users

If you are **not on Linux** and cannot run the bash script `1 - run_clean_nans.sh`, you need to manually execute the Python script:

1. Navigate to the `src/` folder
2. Run `clean_nans.py` directly with Python:
   ```bash
   python src/clean_nans.py
   ```
   Or open the file and execute it in your IDE.

---

### 📂 src/ - Source Modules

The `src/` folder contains all reusable Python modules imported by the notebooks:

| File                  | Description                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `dataprocessor.py`    | `DataProcessor` class for loading, merging, and preprocessing clinical/molecular data            |
| `clean_nans.py`       | Handles missing values (NaN) using median/mean imputation with `SimpleImputer`                   |
| `modeltrainer.py`     | `ModelML` class for training models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting) with hyperparameter tuning |
| `eval_utils.py`       | Evaluation functions: metrics computation (accuracy, F1, ROC-AUC, Brier score, etc.) and plots   |
| `calibration_utils.py`| Calibration utilities: Platt Scaling and Isotonic Regression for probability calibration         |
| `func_utils.py`       | Utility functions for feature selection, correlation analysis, and data cleaning                 |

---

### 📂 Other Folders

| Folder                    | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `data/`                   | Raw and processed datasets (CSV, Excel)                  |
| `models/`                 | Saved tuned models (`.pkl`) and hyperparameters (`.json`)|
| `model_calibrated/`       | Platt-calibrated models and calibration plots            |
| `model_calibrated_isotonic/` | Isotonic-calibrated models and calibration plots      |
| `imputer/`                | Saved imputer object for missing value handling          |
| `results/`                | Evaluation metrics and results for each model            |
| `report/`                 | LaTeX report files                                       |

The folders report,model_calibrated/calibrated_isotonic aren't on the online repo because of their size, aswell as the the pickle files of the models.
