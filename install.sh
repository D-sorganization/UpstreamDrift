#!/bin/bash

# UpstreamDrift - Quick Install Script
#
# Supported execution models:
#   curl -fsSL https://raw.githubusercontent.com/D-sorganization/UpstreamDrift/main/install.sh | bash
#   bash install.sh
#   UPSTREAM_DRIFT_INSTALL_SOURCE=/path/to/checkout bash install.sh
#
# When the script is run from a checkout that contains pyproject.toml, it
# installs from the local tree. Otherwise it installs from the Git repository.

set -euo pipefail

REPO_URL="https://github.com/D-sorganization/UpstreamDrift.git"
INSTALL_SOURCE="${UPSTREAM_DRIFT_INSTALL_SOURCE:-}"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         UpstreamDrift - Installation Script                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected: $OS $ARCH"

# Check for Python 3.11+
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

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

if [ -z "$INSTALL_SOURCE" ]; then
    if [ -f "./pyproject.toml" ]; then
        INSTALL_SOURCE="."
    else
        INSTALL_SOURCE="git+${REPO_URL}"
    fi
fi

if [ "$INSTALL_SOURCE" = "." ]; then
    echo "Using local checkout at $(pwd)"
else
    echo "Using remote install source: $INSTALL_SOURCE"
fi

if command -v pipx &> /dev/null; then
    echo "✓ pipx found - using isolated installation"
    INSTALL_CMD=(pipx install "$INSTALL_SOURCE")
else
    echo "⚠ pipx not found - using python3 -m pip (consider installing pipx)"
    INSTALL_CMD=(python3 -m pip install --user "$INSTALL_SOURCE")
fi

echo
echo "Installing UpstreamDrift..."
echo "  Command: ${INSTALL_CMD[*]}"
echo

"${INSTALL_CMD[@]}"

# ---------------------------------------------------------------------------
# Optional Rust extension wheels (issue #7600).
#
# The package install above is pure-Python; the PyO3 crates under rust_core/
# accelerate the physics / mocap hot paths. When installing from a local
# checkout that contains rust_core/, build each crate's wheel with maturin and
# install it into the same environment the package went to. No external wheel
# index is used — everything compiles from source. The Python facades fall
# back to pure Python if these are absent, so a missing Rust toolchain is a
# warning, not a hard failure. Keep this crate list in lock-step with the
# maturin build loop in .github/workflows/ci-standard.yml.
# ---------------------------------------------------------------------------
RUST_CRATES=(
    "rust_core/upstream-physics"
    "rust_core/upstream-mocap-preproc"
    "rust_core/upstream-mocap-io"
    "rust_core/upstream-muscle"
    "rust_core/upstream-motion-matching"
    "rust_core/ai_backend"
)

if [ "$INSTALL_SOURCE" = "." ] && [ -d "rust_core" ]; then
    if command -v cargo &> /dev/null; then
        echo
        echo "Building Rust extension wheels (maturin)..."
        if ! command -v maturin &> /dev/null; then
            echo "  Installing maturin..."
            python3 -m pip install --user maturin || python3 -m pip install maturin
        fi

        WHEEL_DIR="$(mktemp -d)"
        build_failed=0
        for crate in "${RUST_CRATES[@]}"; do
            echo "  - $crate"
            if ! maturin build --release --features python \
                -m "$crate/Cargo.toml" --out "$WHEEL_DIR"; then
                echo "  ⚠ Failed to build $crate; skipping Rust acceleration."
                build_failed=1
                break
            fi
        done

        if [ "$build_failed" -eq 0 ]; then
            if command -v pipx &> /dev/null; then
                # pipx installed the package in its own venv; inject wheels there.
                for whl in "$WHEEL_DIR"/*.whl; do
                    pipx inject upstream-drift "$whl" || true
                done
            else
                python3 -m pip install --user --no-deps "$WHEEL_DIR"/*.whl
            fi
            echo "✓ Rust extension wheels installed"
        fi
        rm -rf "$WHEEL_DIR"
    else
        echo
        echo "⚠ Rust toolchain (cargo) not found — skipping Rust acceleration."
        echo "  Install Rust from https://rustup.rs to enable the native kernels."
    fi
fi

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                     ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                               ║"
echo "║   To start:   upstream-drift                                  ║"
echo "║   Help:       upstream-drift --help                           ║"
echo "║                                                               ║"
echo "║   The app will open in your browser at localhost:8000         ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
