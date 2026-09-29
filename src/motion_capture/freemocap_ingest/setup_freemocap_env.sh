#!/usr/bin/env bash
#
# setup_freemocap_env.sh - Setup script for FreeMoCap isolated Python environment
#
# This script creates an isolated Python environment for running FreeMoCap
# as a sidecar process, keeping the AGPL-licensed code separate from the
# main UpstreamDrift process.
#
# Usage:
#   ./setup_freemocap_env.sh [ENV_NAME]
#
# Arguments:
#   ENV_NAME - Optional name for the environment (default: freemocap-env)
#
# Requirements:
#   - Python 3.10+ (3.12 recommended)
#   - pip or conda
#
# The script will:
#   1. Create a new virtual environment
#   2. Install FreeMoCap and its dependencies
#   3. Verify the installation
#

set -euo pipefail

# Configuration
ENV_NAME="${1:-freemocap-env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_DIR="${HOME}/${ENV_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for Python
check_python() {
    if command -v python"${PYTHON_VERSION}" &> /dev/null; then
        PYTHON_EXE=python"${PYTHON_VERSION}"
    elif command -v python3 &> /dev/null; then
        PYTHON_EXE=python3
        log_warn "Python ${PYTHON_VERSION} not found, using python3 instead"
    else
        log_error "Python not found. Please install Python ${PYTHON_VERSION}+"
        exit 1
    fi

    log_info "Using Python: $($PYTHON_EXE --version)"
}

# Check for pip
check_pip() {
    if ! $PYTHON_EXE -m pip --version &> /dev/null; then
        log_error "pip not found. Please install pip for Python"
        exit 1
    fi
    log_info "pip version: $($PYTHON_EXE -m pip --version)"
}

# Create virtual environment
create_venv() {
    if [ -d "$INSTALL_DIR" ]; then
        log_warn "Directory $INSTALL_DIR already exists"
        read -p "Remove and recreate? (y/N): " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            log_info "Using existing environment at $INSTALL_DIR"
            return 0
        fi
    fi

    log_info "Creating virtual environment at $INSTALL_DIR"
    $PYTHON_EXE -m venv "$INSTALL_DIR"

    # Activate the environment
    source "$INSTALL_DIR/bin/activate"
    log_info "Virtual environment created and activated"
}

# Upgrade pip and install base packages
upgrade_pip() {
    log_info "Upgrading pip and base packages..."
    pip install --upgrade pip setuptools wheel
}

# Install FreeMoCap
install_freemocap() {
    log_info "Installing FreeMoCap and dependencies..."

    # FreeMoCap installation
    # Note: FreeMoCap has specific dependency requirements
    # Pinning versions to avoid conflicts

    pip install \
        "opencv-contrib-python>=4.8.0,<4.9.0" \
        "pydantic>=2.0.0,<3.0.0" \
        "numpy>=1.24.0,<2.0.0" \
        "pandas>=2.0.0" \
        "scipy>=1.10.0" \
        "matplotlib>=3.7.0"

    # Install FreeMoCap
    # Using latest stable release
    pip install freemocap>=1.8.0

    log_info "FreeMoCap installation complete"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    if python -c "import freemocap" 2>/dev/null; then
        log_info "FreeMoCap import successful"
    else
        log_error "FreeMoCap import failed"
        return 1
    fi

    # Check version
    VERSION=$(python -c "import freemocap; print(freemocap.__version__)" 2>/dev/null || echo "unknown")
    log_info "FreeMoCap version: $VERSION"

    # Verify CLI is available
    if python -m freemocap --help &>/dev/null; then
        log_info "FreeMoCap CLI available"
    else
        log_warn "FreeMoCap CLI not responding to --help"
    fi

    return 0
}

# Print usage instructions
print_usage() {
    echo ""
    echo "========================================"
    echo "FreeMoCap environment setup complete!"
    echo "========================================"
    echo ""
    echo "To activate the environment:"
    echo "  source $INSTALL_DIR/bin/activate"
    echo ""
    echo "To run FreeMoCap:"
    echo "  python -m freemocap"
    echo ""
    echo "Or use the UpstreamDrift launcher:"
    echo "  python -m src.motion_capture.freemocap_ingest /path/to/session"
    echo ""
    echo "To deactivate:"
    echo "  deactivate"
    echo ""
}

# Main execution
main() {
    echo "========================================"
    echo "FreeMoCap Environment Setup"
    echo "========================================"
    echo ""

    check_python
    check_pip
    create_venv
    upgrade_pip
    install_freemocap

    if verify_installation; then
        print_usage
        log_info "Setup completed successfully!"
        exit 0
    else
        log_error "Setup completed with errors"
        exit 1
    fi
}

main "$@"