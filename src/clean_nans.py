"""Impute missing values in the paired datasets (median imputer fitted on the training pairs only)."""
import pandas as pd
import joblib
import os
from sklearn.impute import SimpleImputer

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed_data.xlsx')
STRATEGY = 'median'  # 'median' or 'mean'
COLS_TO_DROP = ['patient_ids', 'feat_40', 'feat_82']
IMPUTER_PATH = os.path.join(PROJECT_ROOT, 'imputer/imputer.pkl')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/processed_data_imputed.xlsx')

# =============================================================================
# DATA LOADING
# =============================================================================
print(f"Loading data from: {DATA_PATH}")

train_data = pd.read_excel(DATA_PATH, sheet_name='train')
cal_data = pd.read_excel(DATA_PATH, sheet_name='calibration')
test_data = pd.read_excel(DATA_PATH, sheet_name='test')

print(f"   - Train: {train_data.shape[0]} rows")
print(f"   - Calibration: {cal_data.shape[0]} rows")
print(f"   - Test: {test_data.shape[0]} rows")

# =============================================================================
# FEATURES / TARGET SPLIT
# =============================================================================
X_train = train_data.iloc[:, :-1].drop(columns=COLS_TO_DROP, errors='ignore')
y_train = train_data.iloc[:, -1]

X_cal = cal_data.iloc[:, :-1].drop(columns=COLS_TO_DROP, errors='ignore')
y_cal = cal_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1].drop(columns=COLS_TO_DROP, errors='ignore')
y_test = test_data.iloc[:, -1]

# =============================================================================
# MISSING VALUES CHECK
# =============================================================================
print(f"\nMissing values before imputation:")
print(f"   - Train: {X_train.isnull().sum().sum()}")
print(f"   - Calibration: {X_cal.isnull().sum().sum()}")
print(f"   - Test: {X_test.isnull().sum().sum()}")

# =============================================================================
# IMPUTATION
# =============================================================================
print(f"\nImputing with strategy: '{STRATEGY}'")

imputer = SimpleImputer(strategy=STRATEGY)

# Fit on train ONLY, transform all sets
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_cal_imputed = pd.DataFrame(
    imputer.transform(X_cal),
    columns=X_cal.columns,
    index=X_cal.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print(f"Imputation done")
print(f"   - Train: {X_train_imputed.isnull().sum().sum()} missing values")
print(f"   - Calibration: {X_cal_imputed.isnull().sum().sum()} missing values")
print(f"   - Test: {X_test_imputed.isnull().sum().sum()} missing values")

# =============================================================================
# REBUILD DATAFRAMES
# =============================================================================
train_imputed = X_train_imputed.copy()
train_imputed['target'] = y_train.values

cal_imputed = X_cal_imputed.copy()
cal_imputed['target'] = y_cal.values

test_imputed = X_test_imputed.copy()
test_imputed['target'] = y_test.values

# =============================================================================
# SAVE
# =============================================================================
# Save the imputer
os.makedirs(os.path.dirname(IMPUTER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
joblib.dump(imputer, IMPUTER_PATH)
print(f"\nImputer saved: {IMPUTER_PATH}")

# Save the imputed data
with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    train_imputed.to_excel(writer, sheet_name='train', index=False)
    cal_imputed.to_excel(writer, sheet_name='calibration', index=False)
    test_imputed.to_excel(writer, sheet_name='test', index=False)

print(f"Imputed data saved: {OUTPUT_PATH}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Strategy: {STRATEGY}")
print(f"Train: {len(train_imputed)} rows, {X_train_imputed.shape[1]} features")
print(f"Calibration: {len(cal_imputed)} rows, {X_cal_imputed.shape[1]} features")
print(f"Test: {len(test_imputed)} rows, {X_test_imputed.shape[1]} features")
print(f"\nFiles created:")
print(f"   - {IMPUTER_PATH}")
print(f"   - {OUTPUT_PATH}")
print(f"{'='*50}")
print("Done.")