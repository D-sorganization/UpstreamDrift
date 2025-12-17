#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
pip install pre-commit black isort ruff mypy nbstripout
pre-commit install
echo "Pre-commit hooks installed."

