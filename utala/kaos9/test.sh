#!/bin/bash
# Convenience script for running tests with reports and coverage

source venv/bin/activate

echo "Running utala: kaos 9 test suite with coverage..."
echo ""

# Run tests with both text and HTML reports plus coverage
python run_tests.py --both --coverage "$@"
