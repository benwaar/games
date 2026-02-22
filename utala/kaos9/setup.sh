#!/bin/bash
# Setup Python 3.11 virtual environment for utala: kaos 9

set -e

echo "Setting up virtual environment with Python 3.11..."
echo ""

# Check if pyenv has Python 3.11
if ! pyenv versions | grep -q "3.11"; then
    echo "Error: Python 3.11 not found in pyenv"
    echo "Install with: pyenv install 3.11.10"
    exit 1
fi

# Create venv with Python 3.11
echo "Creating venv with Python 3.11..."
~/.pyenv/versions/3.11.10/bin/python3 -m venv venv

# Activate and install requirements
echo "Installing dependencies..."
source venv/bin/activate

# Install all requirements
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate:"
echo "  source activate.sh"
echo ""
echo "To run demos:"
echo "  ./run.sh           # Run demo.py"
