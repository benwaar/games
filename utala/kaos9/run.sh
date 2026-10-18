#!/bin/bash
# Activate virtual environment and run Python script

# Activate the virtual environment
source venv/bin/activate

# Run the script (default to demo_human.py, or pass script name as argument)
SCRIPT="${1:-demo_human.py}"

echo "Running $SCRIPT in virtual environment..."
python "$SCRIPT"

# Note: Terminal stays in activated venv after script runs
# Use deactivate.sh or type 'deactivate' to exit venv
