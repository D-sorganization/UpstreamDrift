#!/bin/bash
set -e

echo "Setting up Robotics Environment..."

# Ensure we are currently in the virtual environment or have permission
echo "Python location: $(which python3)"
echo "Pip location: $(which pip)"

# Update Pip
echo "Updating pip..."
pip install --upgrade pip setuptools wheel

# Install Requirements
echo "Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found in $(pwd)"
    exit 1
fi

echo "Environment setup complete."
