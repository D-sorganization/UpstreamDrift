#!/usr/bin/env bash
set -euo pipefail

echo "=== MuJoCo Golf Swing Model Setup ==="

# ------------------------------------------------------------------------------
# 1. Basic sanity checks
# ------------------------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3 (Ubuntu: sudo apt install python3) and rerun."
  exit 1
fi

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------------------------
# 2. Install system dependencies (Linux only)
# ------------------------------------------------------------------------------

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo ">>> Updating apt package lists..."
  sudo apt update

  echo ">>> Installing system dependencies..."
  sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    patchelf
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo ">>> macOS detected. Install dependencies via Homebrew if needed:"
  echo "    brew install python3"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
  echo ">>> Windows detected. Using Python venv (no system packages needed)."
fi

# ------------------------------------------------------------------------------
# 3. Create (or reuse) project-specific venv
# ------------------------------------------------------------------------------

VENV_DIR="${PROJECT_ROOT}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo ">>> Creating virtual environment at ${VENV_DIR} ..."
  python3 -m venv "${VENV_DIR}"
else
  echo ">>> Virtual environment already exists at ${VENV_DIR} (reusing)."
fi

echo ">>> Activating virtual environment..."
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# ------------------------------------------------------------------------------
# 4. Upgrade pip and install Python packages
# ------------------------------------------------------------------------------

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing dependencies from requirements.txt..."
if [ -f "${PROJECT_ROOT}/python/requirements.txt" ]; then
  pip install -r "${PROJECT_ROOT}/python/requirements.txt"
else
  echo "WARNING: requirements.txt not found, installing core packages..."
  pip install mujoco>=3.0.0 numpy pandas matplotlib scipy PyQt6
fi

# ------------------------------------------------------------------------------
# 5. Install development tools
# ------------------------------------------------------------------------------

echo ">>> Installing development tools..."
pip install pre-commit black isort ruff mypy nbstripout

# ------------------------------------------------------------------------------
# 6. Setup pre-commit hooks
# ------------------------------------------------------------------------------

if [ -f "${PROJECT_ROOT}/.pre-commit-config.yaml" ]; then
  echo ">>> Installing pre-commit hooks..."
  pre-commit install
else
  echo ">>> No .pre-commit-config.yaml found, skipping pre-commit setup."
fi

# ------------------------------------------------------------------------------
# 7. Sanity check
# ------------------------------------------------------------------------------

echo ">>> Running sanity check..."

python3 - << 'EOF'
import mujoco
import numpy as np
import sys

print("MuJoCo:", mujoco.__version__)
print("NumPy:", np.__version__)
print("Python:", sys.version)

# Test basic MuJoCo functionality
xml_string = """
<mujoco>
  <worldbody>
    <body>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
print("MuJoCo model loaded successfully!")
print("Sanity check passed.")
EOF

echo "=== Setup complete. To use this environment later, run: ==="
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "Or from the project root:"
echo "    source .venv/bin/activate"
echo ""
echo "To run the GUI:"
echo "    python python/mujoco_golf_pendulum/gui.py"

