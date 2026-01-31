#!/bin/bash
# Run UpstreamDrift from WSL2 with full Pinocchio/Drake/MuJoCo support
#
# Usage from Windows:
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/diete/Repositories/UpstreamDrift/run_wsl.sh
#
# Or from WSL terminal:
#   bash /mnt/c/Users/diete/Repositories/UpstreamDrift/run_wsl.sh

set -e

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate golf_suite

# Fix library path for pinocchio/crocoddyl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Set up project path
PROJECT_DIR="/mnt/c/Users/diete/Repositories/UpstreamDrift"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Set display for GUI apps (WSLg handles this automatically)
export DISPLAY="${DISPLAY:-:0}"

echo "=========================================="
echo "  UpstreamDrift (WSL2 Environment)"
echo "=========================================="
echo ""
echo "Python:     $(python --version 2>&1 | cut -d' ' -f2)"
echo "Pinocchio:  $(python -c 'import pinocchio; print(pinocchio.__version__)' 2>/dev/null || echo 'N/A')"
echo "MuJoCo:     $(python -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null || echo 'N/A')"
echo "Drake:      $(python -c 'import pydrake; print("OK")' 2>/dev/null || echo 'N/A')"
echo "Pink:       $(python -c 'import pink; print(pink.__version__)' 2>/dev/null || echo 'N/A')"
echo "Crocoddyl:  $(python -c 'import crocoddyl; print(crocoddyl.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# Check what to run
if [ "$1" == "--pinocchio" ]; then
    echo "Launching Pinocchio GUI..."
    python src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py
elif [ "$1" == "--mujoco" ]; then
    echo "Launching MuJoCo Humanoid Launcher..."
    python src/engines/physics_engines/mujoco/python/humanoid_launcher.py
elif [ "$1" == "--drake" ]; then
    echo "Launching Drake GUI..."
    python src/engines/physics_engines/drake/python/src/drake_gui_app.py
else
    echo "Launching Golf Suite Launcher..."
    python src/launchers/golf_launcher.py "$@"
fi
