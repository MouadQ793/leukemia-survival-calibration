"""
Step 1 of the pipeline: impute missing values at the PATIENT level, then build the pairs.

- The median imputer is fitted on the training patients only, then applied to the
  calibration and test patients (and later to the challenge test set in notebook 08).
  Imputing patients rather than pairs gives every patient the same weight; fitting on pairs
  would over-weight patients that appear in many comparable pairs.
- Pairs are then built inside each split, so no patient appears in two splits.

Inputs  (written by 00_data_processing.ipynb):
    data/patients_{train,calibration,test}.csv, data/selected_features.json
Outputs:
    imputer/imputer.pkl, data/pairs_{train,calibration,test}.pkl
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.func_utils import generate_pairs_concat, save_pairs  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMPUTER_PATH = os.path.join(PROJECT_ROOT, 'imputer', 'imputer.pkl')
STRATEGY = 'median'  # 'median' or 'mean'
SPLITS = ['train', 'calibration', 'test']
# Maximum number of candidate pairs drawn per split before the comparability filter
MAX_CANDIDATE_PAIRS = {'train': 800_000, 'calibration': 100_000, 'test': 100_000}
RANDOM_STATE = 42


def main():
    with open(os.path.join(DATA_DIR, 'selected_features.json')) as f:
        features = json.load(f)['features']

    patients = {s: pd.read_csv(os.path.join(DATA_DIR, f'patients_{s}.csv')) for s in SPLITS}
    for s, df in patients.items():
        n_missing = int(df[features].isna().sum().sum())
        print(f"{s:<12}: {len(df)} patients, {n_missing} missing feature values")

    # Fit on the training patients ONLY
    imputer = SimpleImputer(strategy=STRATEGY)
    imputer.fit(patients['train'][features].astype(float))
    os.makedirs(os.path.dirname(IMPUTER_PATH), exist_ok=True)
    joblib.dump({'imputer': imputer, 'features': features}, IMPUTER_PATH)
    print(f"\nImputer ({STRATEGY}) fitted on {len(patients['train'])} training patients, "
          f"saved to {IMPUTER_PATH}")

    for s in SPLITS:
        df = patients[s]
        X = df[['ID']].copy()
        X[features] = imputer.transform(df[features].astype(float))
        Y = df[['OS_STATUS', 'OS_YEARS']]
        print(f"\n[{s}]")
        pairs, labels, _ = generate_pairs_concat(
            X, Y, max_pairs=MAX_CANDIDATE_PAIRS[s], random_state=RANDOM_STATE, id_column='ID'
        )
        save_pairs(pairs, labels, os.path.join(DATA_DIR, f'pairs_{s}.pkl'))

    print("\nDone.")


if __name__ == '__main__':
    main()
