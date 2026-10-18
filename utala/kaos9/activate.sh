#!/bin/bash
# Activate the virtual environment
# Usage: source activate.sh

source venv/bin/activate
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "To deactivate, run: deactivate"
