#!/usr/bin/env bash
# Set up isolated MJX virtual environment pinned from scripts/config/mjx_env_pins.json
# Usage: bash scripts/setup_mjx_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/mjx_env_pins.json"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Configuration file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

VENV_DIR="${HOME}/.venv-mjx"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

echo "Setting up MJX environment in ${VENV_DIR}..."

if [ ! -f "${PYTHON_BIN}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

# Read pinned packages from JSON
PACKAGES="$(python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
pkgs = data.get("packages", data)
print(" ".join(f"{k}=={v}" if v else k for k, v in pkgs.items()))
' "${CONFIG_FILE}")"

echo "Installing pinned packages: ${PACKAGES}..."
"${PIP_BIN}" install ${PACKAGES}

echo ""
echo "Installed package verification:"
"${PYTHON_BIN}" -c "
import defusedxml, jax, mujoco, numpy, pytest, scipy
from mujoco import mjx
print(f'  jax:        {jax.__version__}')
print(f'  mujoco:     {mujoco.__version__}')
print(f'  mujoco-mjx: {mjx.__file__ is not None}')
print(f'  defusedxml: {defusedxml.__version__}')
print(f'  numpy:      {numpy.__version__}')
print(f'  scipy:      {scipy.__version__}')
print(f'  pytest:     {pytest.__version__}')
"

echo ""
echo "MJX environment setup complete at ${VENV_DIR}"
