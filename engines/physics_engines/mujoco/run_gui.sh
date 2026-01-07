#!/bin/bash
# Launcher script for MuJoCo Golf Swing Model GUI (macOS/Linux)
# This script activates the conda environment and runs the GUI application

set -e  # Exit on error

echo "========================================"
echo "MuJoCo Golf Swing Model - GUI Launcher"
echo "========================================"
echo ""

# Function to find conda installation
find_conda() {
    # Common conda installation paths
    local conda_paths=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/conda/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/conda/etc/profile.d/conda.sh"
    )
    
    for path in "${conda_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    # Check if conda is in PATH
    if command -v conda &> /dev/null; then
        # Try to get conda base path
        local conda_base=$(conda info --base 2>/dev/null)
        if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            echo "$conda_base/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    
    return 1
}

# Initialize conda
CONDA_SCRIPT=$(find_conda)
if [ -n "$CONDA_SCRIPT" ]; then
    echo "Found conda at: $CONDA_SCRIPT"
    source "$CONDA_SCRIPT"
else
    echo "WARNING: Conda not found in standard locations"
    echo "Attempting to use conda from PATH..."
    if ! command -v conda &> /dev/null; then
        echo "ERROR: Conda not found"
        echo "Please install Miniconda/Anaconda or activate your environment manually"
        echo ""
        echo "Alternatively, activate your environment and run:"
        echo "  python -m python.mujoco_humanoid_golf"
        exit 1
    fi
fi

# Activate conda environment
echo "Activating conda environment 'sim-env'..."
if conda activate sim-env 2>/dev/null; then
    echo "Environment activated successfully"
else
    echo "WARNING: Could not activate conda environment 'sim-env'"
    echo "Attempting to run with current Python environment..."
    echo ""
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    echo "Please ensure Python is installed and in your PATH"
    exit 1
fi

echo ""
echo "Starting MuJoCo Golf Swing Model GUI..."
echo ""

# Run the application
python -m python.mujoco_humanoid_golf

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Application exited with error code: $EXIT_CODE"
    echo ""
    exit $EXIT_CODE
fi

