#!/usr/bin/env bash
# Step 1 of the pipeline: impute missing values in the paired datasets.
# Run from anywhere; uses the Python interpreter on your PATH unless PYTHON is set,
# e.g.  PYTHON=.venv/bin/python ./01_impute_missing_values.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "Running src/clean_nans.py with ${PYTHON}..."
"${PYTHON}" "${SCRIPT_DIR}/src/clean_nans.py"
echo "Imputation finished."
