#!/usr/bin/env bash
# Step 1 of the pipeline: patient-level imputation (fitted on the training patients) and pair building.
# Run from anywhere; uses the Python interpreter on your PATH unless PYTHON is set,
# e.g.  PYTHON=.venv/bin/python ./01_impute_and_pair.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "Running src/impute_and_pair.py with ${PYTHON}..."
"${PYTHON}" "${SCRIPT_DIR}/src/impute_and_pair.py"
echo "Imputation and pair building finished."
