#!/bin/bash
# Test RAT benchmark with evalscope framework
# This script uses the evalscope conda environment and runs the RAT benchmark test

set -e  # Exit on error

echo "========================================================================"
echo "RAT Benchmark Test Script"
echo "========================================================================"
echo ""

# Use evalscope conda environment Python directly
PYTHON_BIN="/root/data/conda/envs/evalscope/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python not found at $PYTHON_BIN"
    echo "Please ensure the evalscope conda environment is installed"
    exit 1
fi

echo "✓ Using Python from evalscope environment: $PYTHON_BIN"
$PYTHON_BIN --version
echo ""

# Check if local API is running
echo "Checking if local API is running at localhost:8007..."
if ! curl -s http://localhost:8007/v1/models > /dev/null; then
    echo "Error: Local API not running at localhost:8007"
    echo "Please start the API server before running this test"
    exit 1
fi

echo "✓ Local API is running"
echo ""

# Run the test script
echo "Running RAT benchmark test..."
echo "------------------------------------------------------------------------"
$PYTHON_BIN /root/data/code/evalscope/temp/test_rat_api.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Test completed successfully!"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "Test failed with exit code: $exit_code"
    echo "========================================================================"
fi

exit $exit_code
