#!/usr/bin/env bash
set -euo pipefail

echo "=== Modeling Stack Setup (Pinocchio, Pink, Drake, QP solvers) ==="

# ------------------------------------------------------------------------------
# 1. Basic sanity checks
# ------------------------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3 (Ubuntu: sudo apt install python3) and rerun."
  exit 1
fi

# ------------------------------------------------------------------------------
# 2. Update apt and install system dependencies
# ------------------------------------------------------------------------------

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
    libeigen3-dev \
    libboost-all-dev \
    liburdfdom-dev \
    liboctomap-dev \
    libassimp-dev \
    libspdlog-dev \
    libfmt-dev \
    libccd-dev \
    libfcl-dev

# ------------------------------------------------------------------------------
# 3. Create (or reuse) project-specific venv
# ------------------------------------------------------------------------------

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

echo ">>> Installing core dependencies from requirements.txt..."
if [ -f "${PROJECT_ROOT}/python/requirements.txt" ]; then
  pip install -r "${PROJECT_ROOT}/python/requirements.txt"
else
  echo "WARNING: requirements.txt not found, installing core packages..."
fi

echo ">>> Installing Pinocchio (pin)..."
pip install pin

echo ">>> Installing Pink (pin-pink) - stacked on top of Pinocchio..."
pip install pin-pink

echo ">>> Installing Crocoddyl - optimal control library for robotics..."
pip install crocoddyl

echo ">>> Installing QP solvers (qpsolvers, osqp, scs, quadprog)..."
pip install qpsolvers osqp scs quadprog

echo ">>> Installing Drake..."
pip install drake

# ------------------------------------------------------------------------------
# 5. Sanity check
# ------------------------------------------------------------------------------

echo ">>> Running sanity check..."

python3 - << 'EOF'
import pinocchio as pin
import pink
import crocoddyl
import pydrake
import qpsolvers
import numpy as np

print("Pinocchio:", pin.__version__)
print("Pink:", pink.__file__)
print("Crocoddyl:", crocoddyl.__version__)
print("Drake:", pydrake.__file__)
print("NumPy:", np.__version__)
print("QP solvers:", qpsolvers.available_solvers)

if not qpsolvers.available_solvers:
    raise SystemExit("ERROR: qpsolvers has no available solvers. Check osqp/scs/quadprog installs.")

print("Sanity check passed.")
EOF

echo "=== Setup complete. To use this environment later, run: ==="
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "Or from the project root:"
echo "    source .venv/bin/activate"
echo ""
echo "Crocoddyl and Pink are installed and ready to use with Pinocchio!"
