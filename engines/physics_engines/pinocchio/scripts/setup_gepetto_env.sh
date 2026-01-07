#!/usr/bin/env bash
set -euo pipefail

echo "=== Gepetto Viewer Setup (conda 'viz' environment) ==="

# ------------------------------------------------------------------------------
# 1. Ensure conda (Miniforge) is installed
# ------------------------------------------------------------------------------

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Use project-specific conda location if possible, fallback to user home
if [ -d "${PROJECT_ROOT}/.conda" ]; then
  CONDA_DIR="${PROJECT_ROOT}/.conda"
  CONDA_BIN="${CONDA_DIR}/bin/conda"
elif [ -d "${HOME}/miniforge3" ]; then
  CONDA_DIR="${HOME}/miniforge3"
  CONDA_BIN="${CONDA_DIR}/bin/conda"
else
  echo ">>> Miniforge not found. Installing to project directory..."
  CONDA_DIR="${PROJECT_ROOT}/.conda"
  cd "${PROJECT_ROOT}"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "${CONDA_DIR}"
  rm Miniforge3-Linux-x86_64.sh
  CONDA_BIN="${CONDA_DIR}/bin/conda"
fi

# shellcheck disable=SC1090
source "${CONDA_DIR}/etc/profile.d/conda.sh" 2>/dev/null || {
  echo ">>> Conda initialization script not found. Using conda directly."
  export PATH="${CONDA_DIR}/bin:${PATH}"
}

# ------------------------------------------------------------------------------
# 2. Create (or reuse) project-specific 'viz' conda environment
# ------------------------------------------------------------------------------

ENV_NAME="pinocchio-viz"

if "${CONDA_BIN}" env list | grep -qE "^${ENV_NAME} "; then
  echo ">>> Conda env '${ENV_NAME}' already exists (reusing)."
else
  echo ">>> Creating conda env '${ENV_NAME}' with gepetto-viewer and pinocchio ..."
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" -c conda-forge python=3.10 gepetto-viewer gepetto-viewer-corba pinocchio
fi

echo ">>> Activating '${ENV_NAME}' environment..."
conda activate "${ENV_NAME}"

# ------------------------------------------------------------------------------
# 3. Sanity check: launch Gepetto backend and basic Pinocchio call
# ------------------------------------------------------------------------------

echo ">>> Running Pinocchio sanity check in 'viz' env..."

python - << 'EOF'
import pinocchio as pin

print("Pinocchio (viz env):", pin.__version__)
print("Model frames example:", pin.buildSampleModelManipulator().nframes)
print("Gepetto Viewer is installed; run 'gepetto-gui' in this env to start the GUI.")
EOF

echo "=== Gepetto setup complete. Usage: ==="
echo "  1) source ${CONDA_DIR}/etc/profile.d/conda.sh"
echo "  2) conda activate ${ENV_NAME}"
echo "  3) gepetto-gui"
echo ""
echo "Or from the project root:"
echo "  source .conda/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
