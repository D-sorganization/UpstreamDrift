#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install --upgrade pip
python3 -m pip install pre-commit ruff mypy nbstripout
pre-commit install
pre-commit install --hook-type pre-push
echo "Pre-commit hooks installed (commit + pre-push)."
