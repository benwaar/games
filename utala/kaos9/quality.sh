#!/bin/bash
# Convenience script for running code quality checks

source venv/bin/activate

echo "Running code quality checks..."
echo ""

# Run quality checks with report
python check_quality.py -r "$@"
