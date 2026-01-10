#!/bin/bash

# Shell script to run clean_nans.py and display progress

echo "============================================"
echo "Running clean_nans.py..."
echo "============================================"
echo ""

# Change to the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use the datascience Python environment
PYTHON_BIN="/home/mouad-sehli/datascience/bin/python"

# Run the Python script
$PYTHON_BIN "$SCRIPT_DIR/src/clean_nans.py"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Script completed successfully!"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "Script failed!"
    echo "============================================"
fi

# Keep terminal open for a few seconds
read -p "Press Enter to close..."
