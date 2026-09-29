#!/bin/bash
# Install script for UpstreamDrift physics engines and dependencies on WSL2 Ubuntu.
set -e

echo "=== UpstreamDrift WSL2 Ubuntu Dependency Installer ==="

# Define required system packages
SYSTEM_DEPS=(
    "build-essential"
    "python3-dev"
    "python3-pip"
    "python3-venv"
    "git"
    "curl"
    "patchelf"
    "ffmpeg"
    "libgl1"
    "libosmesa6"
    "libglew2.2"
    "libegl1"
    "libglib2.0-0"
)

echo "Updating apt package sources..."
sudo apt-get update

echo "Installing system dependencies..."
sudo apt-get install -y --no-install-recommends "${SYSTEM_DEPS[@]}"

# Setup Python virtual environment in the repository
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-wsl"

echo "Setting up virtual environment in $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing UpstreamDrift in editable mode with all engines and gui-tools..."
# [all-engines] installs drake and pinocchio
# [biomechanics] installs myosuite and opensim
# [gui-tools] installs PyQt6 and layout helpers
pip install -e "$REPO_ROOT[all-engines,biomechanics,gui-tools]"

echo "=== WSL2 Dependency Installation Complete ==="
echo "To run the launcher in WSL mode, make sure to activate the WSL venv:"
echo "  source $VENV_DIR/bin/activate"
