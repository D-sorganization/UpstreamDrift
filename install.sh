#!/bin/bash
# Golf Modeling Suite - Quick Install Script
# Usage: curl -fsSL https://golf-suite.io/install.sh | bash
#
# Security / threat model
# -----------------------
# This script installs the package with `pipx install .` (preferred) or
# `pip3 install .`.  Neither step verifies package hashes by default.
# Trust model: you are running a script fetched directly from the project's
# official URL over HTTPS; the integrity of that download depends on TLS and
# the hosting provider.
#
# For a higher-assurance install (e.g. in a regulated environment):
#   1. Download the wheel from the GitHub release and verify its SHA-256
#      against the value published in the release notes.
#   2. Install with:  pip3 install --require-hashes upstream_drift-*.whl
#
# Dependency hygiene: transitive pip dependencies are resolved at install time
# and are NOT hash-pinned by this script.  If reproducible installs are
# required, generate a requirements file with `pip-compile --generate-hashes`
# and install from that instead.

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Golf Modeling Suite - Installation Script             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected: $OS $ARCH"

# Check for Python 3.11+
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
        echo "✓ Python $PY_VERSION found"
    else
        echo "✗ Python 3.11+ required (found $PY_VERSION)"
        echo "  Please install Python 3.11 or newer"
        exit 1
    fi
else
    echo "✗ Python 3 not found"
    echo "  Please install Python 3.11 or newer"
    exit 1
fi

# Check for pipx (recommended) or pip
if command -v pipx &> /dev/null; then
    echo "✓ pipx found - using isolated installation"
    INSTALL_CMD="pipx install ."
elif command -v pip3 &> /dev/null; then
    echo "⚠ pipx not found - using pip (consider installing pipx)"
    INSTALL_CMD="pip3 install ."
else
    echo "✗ Neither pipx nor pip found"
    exit 1
fi

echo
echo "Installing Golf Modeling Suite..."
echo "  Command: $INSTALL_CMD"
echo

$INSTALL_CMD

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                     ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                               ║"
echo "║   To start:   upstream-drift                                      ║"
echo "║   Help:       upstream-drift --help                               ║"
echo "║                                                               ║"
echo "║   The app will open in your browser at localhost:8000         ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
