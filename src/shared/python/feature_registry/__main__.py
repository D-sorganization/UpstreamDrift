"""``python -m src.shared.python.feature_registry`` entry point."""

from __future__ import annotations

import sys

from src.shared.python.feature_registry.registry import _main

if __name__ == "__main__":  # pragma: no cover - thin shim
    sys.exit(_main(sys.argv[1:]))
