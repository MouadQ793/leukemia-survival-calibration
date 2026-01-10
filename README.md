# Data Science Project - Machine Learning Model Training & Calibration

A comprehensive machine learning pipeline for training, evaluating, and calibrating classification models (Logistic Regression, Random Forest, XGBoost) with data preprocessing, feature engineering, and model calibration techniques.

## 📁 Project Structure

```
DS-final/
├── README.md                              # This file
├── .gitignore                             # Git ignore configuration
│
├── 📓 NOTEBOOKS (Main Analysis & Experiments)
│   ├── 0 - Data treatement and processing.ipynb      # Data loading, cleaning, imputation
│   ├── 2 - Logistic regression.ipynb                 # Logistic Regression model training
│   ├── 3 - XGBoost.ipynb                             # XGBoost model training
│   ├── 4 - Random Forest.ipynb                       # Random Forest model training
│   ├── 5 - Plat Calibration.ipynb                    # Platt Scaling calibration
│   ├── 6 - Isotonic Calibration.ipynb                # Isotonic Regression calibration
│   ├── 7 - Final Results.ipynb                       # Summary & comparison of results
│   ├── compte rendu.ipynb                            # Analysis report (French: "compte rendu")
│   └── modeltrainer.ipynb                            # Legacy model training notebook
│
├── 📂 src/ (Python Modules)
│   ├── calibration_utils.py                          # Calibration methods (Platt, Isotonic)
│   ├── modeltrainer.py                               # Model training & hyperparameter tuning
│   ├── dataprocessor.py                              # Data loading & preprocessing
│   ├── eval_utils.py                                 # Model evaluation metrics
│   ├── func_utils.py                                 # Utility functions (correlation, features)
│   ├── clean_nans.py                                 # Missing value handling
│   └── __pycache__/                                  # Python cache
│
├── 📂 data/ (Input Data)
│   ├── clinical_train.csv                            # Clinical features training data
│   ├── molecular_train.csv                           # Molecular features training data
│   ├── target_train.csv                              # Target variable training data
│   ├── treated_data.csv                              # Preprocessed/treated data
│   └── processed_data_imputed.xlsx                   # Final processed data with imputation
│
├── 📂 models/ (Trained Models - Pre-tuned)
│   ├── logistic_regression_tuned.pkl                 # Tuned logistic regression model
│   ├── random_forest_tuned.pkl                       # Tuned random forest model
│   ├── xgboost_tuned.pkl                             # Tuned XGBoost model
│   ├── logistic_regression_tuned_params.json         # Tuning parameters for LR
│   ├── random_forest_tuned_params.json               # Tuning parameters for RF
│   └── xgboost_tuned_params.json                     # Tuning parameters for XGB
│
├── 📂 model_calibrated/ (Platt-Scaled Calibrated Models)
│   ├── logistic_regression_tuned_calibrated.pkl      # Platt-calibrated LR
│   ├── random_forest_tuned_calibrated.pkl            # Platt-calibrated RF
│   ├── xgboost_tuned_calibrated.pkl                  # Platt-calibrated XGB
│   ├── comparison_calibration.png                    # Calibration curves comparison
│   ├── calibration_results.csv                       # Platt calibration metrics
│   └── plots/                                        # Individual calibration plots
│
├── 📂 model_calibrated_isotonic/ (Isotonic-Calibrated Models)
│   ├── logistic_regression_tuned_calibrated.pkl      # Isotonic-calibrated LR
│   ├── random_forest_tuned_calibrated.pkl            # Isotonic-calibrated RF
│   ├── xgboost_tuned_calibrated.pkl                  # Isotonic-calibrated XGB
│   ├── comparison_calibration_isotonic.png           # Isotonic calibration comparison
│   ├── calibration_results.csv                       # Isotonic calibration metrics
│   └── plots/                                        # Individual isotonic calibration plots
│
├── 📂 imputer/ (Data Imputation)
│   └── imputer.pkl                                   # Fitted imputer for missing values
│
├── 📂 results/ (Model Evaluation Results)
│   ├── logistic_regression_tuned_train/
│   │   ├── logistic_regression_tuned_train_metrics.csv
│   │   └── logistic_regression_tuned_train_results.txt
│   ├── logistic_regression_tuned_test/
│   │   ├── logistic_regression_tuned_test_metrics.csv
│   │   └── logistic_regression_tuned_test_results.txt
│   ├── random_forest_tuned_train/
│   │   ├── random_forest_tuned_train_metrics.csv
│   │   └── random_forest_tuned_train_results.txt
│   ├── random_forest_tuned_test/
│   │   ├── random_forest_tuned_test_metrics.csv
│   │   └── random_forest_tuned_test_results.txt
│   ├── xgboost_tuned_train/
│   │   ├── xgboost_tuned_train_metrics.csv
│   │   └── xgboost_tuned_train_results.txt
│   └── xgboost_tuned_test/
│       ├── xgboost_tuned_test_metrics.csv
│       └── xgboost_tuned_test_results.txt
│
├── __pycache__/                           # Python cache files
└── run_clean_nans.sh                      # Bash script for data cleaning

```

---

## 🔄 Workflow Overview

### 1. **Data Preparation** (`0 - Data treatement and processing.ipynb`)
   - Load clinical, molecular, and target data
   - Handle missing values with imputation
   - Feature engineering and correlation analysis
   - Split data into calibration and test sets
   - Export processed data to Excel with multiple sheets

### 2. **Model Training** (`2-4 - Model notebooks`)
   - Train three classification models:
     - **Logistic Regression** (2 - Logistic regression.ipynb)
     - **Random Forest** (4 - Random Forest.ipynb)
     - **XGBoost** (3 - XGBoost.ipynb)
   - Hyperparameter tuning using Grid/Random Search
   - Cross-validation for robustness
   - Save tuned models and parameters

### 3. **Model Calibration** (`5-6 - Calibration notebooks`)
   - **Platt Scaling** (5 - Plat Calibration.ipynb)
     - Uses sigmoid calibration
     - Suitable for probabilistic calibration
   - **Isotonic Regression** (6 - Isotonic Calibration.ipynb)
     - More flexible, non-parametric calibration
     - Better for complex probability distributions
   - Generate calibration curves and metrics

### 4. **Results Summary** (`7 - Final Results.ipynb`)
   - Compare all models (trained and calibrated)
   - Visualize performance metrics
   - Generate final recommendations

---

## 📚 Core Modules

### **calibration_utils.py**
Handles model calibration using both Platt and Isotonic methods.

**Key Functions:**
- `calibrate_model(base_model, X_cal, y_cal, X_test, y_test, method)` - Calibrate a single model
- `calibrate_all_models(model_dir, output_dir, X_cal, y_cal, X_test, y_test, method)` - Calibrate all models
- `plot_all_calibration_comparison(results_df, save_path)` - Compare calibration curves

**Key Class:**
- `CalibratedModel` - Wrapper for calibrated model predictions

---

### **modeltrainer.py**
Manages model training and hyperparameter tuning.

**Key Functions:**
- `train_model(X_train, y_train, model_type)` - Train a model with default parameters
- `tune_hyperparameters(X_train, y_train, model_type)` - Hyperparameter tuning
- `save_model(model, path)` - Save trained model
- `load_model(path)` - Load saved model

---

### **dataprocessor.py**
Handles data loading and preprocessing.

**Key Functions:**
- `load_data(paths)` - Load CSV files and combine features with targets
- `handle_missing_values(df, strategy)` - Impute missing values
- `split_data(X, y, test_size, random_state)` - Train/test split
- `save_processed_data(data, path)` - Save to Excel with sheets

---

### **eval_utils.py**
Evaluation metrics and model performance assessment.

**Key Functions:**
- `compute_metrics(y_true, y_pred, y_proba)` - Calculate classification metrics
- `plot_roc_curve(y_true, y_proba)` - ROC curve visualization
- `plot_confusion_matrix(y_true, y_pred)` - Confusion matrix

---

### **func_utils.py**
General utility functions for feature analysis.

**Key Functions:**
- `compute_correlation_matrix(df)` - Feature correlation analysis
- `select_top_features(df, target, n_features)` - Feature selection
- `generate_feature_pairs(features)` - Pairwise feature generation

---

### **clean_nans.py**
Specialized missing value handling.

**Key Functions:**
- `identify_missing_patterns(df)` - Analyze missing value patterns
- `impute_missing_values(df, strategy, fill_value)` - Multiple imputation strategies

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost, joblib

### Installation
```bash
# Clone repository
git clone https://github.com/MouadQ793/DataScience.git
cd DataScience

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

1. **Data Preparation**
   ```bash
   jupyter notebook "0 - Data treatement and processing.ipynb"
   ```

2. **Train Models**
   ```bash
   jupyter notebook "2 - Logistic regression.ipynb"
   jupyter notebook "3 - XGBoost.ipynb"
   jupyter notebook "4 - Random Forest.ipynb"
   ```

3. **Calibrate Models**
   ```bash
   jupyter notebook "5 - Plat Calibration.ipynb"
   jupyter notebook "6 - Isotonic Calibration.ipynb"
   ```

4. **View Results**
   ```bash
   jupyter notebook "7 - Final Results.ipynb"
   ```

---

## 📊 Data Format

### Input Files (from `data/` folder)
- **clinical_train.csv** - Clinical features (rows: samples, cols: clinical features)
- **molecular_train.csv** - Molecular features (rows: samples, cols: molecular features)
- **target_train.csv** - Binary target variable (0/1)

### Processed Output
- **processed_data_imputed.xlsx** - Multi-sheet Excel file with:
  - `train` - Full training set
  - `calibration` - Data for calibration
  - `test` - Test set

---

## 🎯 Model Performance Metrics

Models are evaluated using:
- **Accuracy** - Overall correctness
- **Precision** - True positive rate among predicted positives
- **Recall** - True positive rate among actual positives
- **F1-Score** - Harmonic mean of precision & recall
- **ROC-AUC** - Area under ROC curve
- **Brier Score** - Calibration error for probabilities
- **Log Loss** - Cross-entropy loss

---

## 📝 Files Generated During Execution

After running the notebooks, you'll generate:

1. **Trained Models** (`models/`)
   - `.pkl` files (serialized models)
   - `_params.json` (hyperparameters)

2. **Calibrated Models** (`model_calibrated/`, `model_calibrated_isotonic/`)
   - Calibrated model `.pkl` files
   - Calibration curves (`.png`)
   - Metrics (`.csv`)

3. **Evaluation Results** (`results/`)
   - Train/test metrics for each model
   - Predictions and probabilities

4. **Data Imputer** (`imputer/`)
   - Fitted imputer for reproducibility

---

## 🔧 Configuration

Key parameters can be modified in notebooks:

- **Test/Calibration Split** - Adjust ratio in data processor
- **Hyperparameter Ranges** - Modify in model trainer
- **Imputation Strategy** - Change in data processor
- **Calibration Method** - Switch between `sigmoid` (Platt) and `isotonic`

---

## 📖 Usage Examples

### Training a New Model
```python
from src.modeltrainer import train_model, save_model

model = train_model(X_train, y_train, model_type='logistic_regression')
save_model(model, 'models/my_model.pkl')
```

### Calibrating a Model
```python
from src.calibration_utils import calibrate_model

calibrated = calibrate_model(
    base_model=model,
    X_cal=X_calibration,
    y_cal=y_calibration,
    X_test=X_test,
    y_test=y_test,
    method='sigmoid'  # or 'isotonic'
)
```

### Evaluating Models
```python
from src.eval_utils import compute_metrics

metrics = compute_metrics(y_true, y_pred, y_proba)
print(metrics)
```

---

## 📌 Notes

- **Data Privacy**: Raw data files are excluded from git (see `.gitignore`)
- **Model Size**: Trained models are large (~477MB) and excluded from git
- **Reproducibility**: Set random seeds in notebooks for consistent results
- **Imputation**: Fitted imputer is saved to ensure consistent preprocessing

---

## 🤝 Contributing

For modifications or improvements:
1. Create a new branch
2. Make your changes
3. Test thoroughly
4. Push and create a pull request

---

## 📧 Contact

For questions or issues, refer to the analysis notebooks or review the inline code comments.

---

## ⚖️ License

[Specify your license here]

---

**Last Updated**: January 2026
