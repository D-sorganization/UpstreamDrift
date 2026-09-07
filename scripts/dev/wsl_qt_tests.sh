#!/usr/bin/env bash
# Run the PyQt6 GUI tests on Linux from a Windows checkout (#9684).
#
# Why: on some Windows hosts any PyQt6 item-view / rich-text widget created
# under pytest dies with a fail-fast (0xC0000409) and no output, while CI on
# Linux is fine. Instead of bisecting Windows, run the same tests the way CI
# does: a Python 3.12 virtual environment managed by uv inside WSL, and
# pytest under xvfb.
#
# Usage (from PowerShell or Git Bash on the Windows side):
#   wsl -d Ubuntu -- bash scripts/dev/wsl_qt_tests.sh [pytest args...]
# Defaults to `-m ui` (every test marked as a GUI test). The venv lives in
# .wsl-venv at the repository root (ignored by git) and is reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VENV="$ROOT/.wsl-venv"
PY_VERSION="${WSL_QT_PYTHON:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 2
  fi
fi
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run is required: sudo apt-get install -y xvfb libgl1 libegl1 libxkbcommon0" >&2
  exit 2
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating $VENV (python $PY_VERSION)"
  uv venv --python "$PY_VERSION" "$VENV"
fi
# Install the project with its Qt/test extras; idempotent, uv resolves fast.
uv pip install --python "$VENV/bin/python" -q -e ".[dev]" 2>/dev/null \
  || uv pip install --python "$VENV/bin/python" -q -e . pytest pytest-qt PyQt6 opencv-python-headless numpy scipy pydantic

export QT_QPA_PLATFORM=offscreen
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(-m ui)
fi
exec xvfb-run -a "$VENV/bin/python" -m pytest -q -p no:cacheprovider -o addopts="" "${ARGS[@]}"
