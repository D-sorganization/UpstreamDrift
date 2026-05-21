"""Compatibility shim for signal_toolkit.

This module delegates to the vendored ud-tools signal_toolkit package
so that ``src.shared.python.signal_toolkit.*`` imports continue to work
after the toolkit was moved to ``vendor/ud-tools``.

By extending ``__path__`` with the vendor directory, Python's normal
submodule resolution will find ``calculus.py``, ``core.py``, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Locate the vendor signal_toolkit directory
# __file__ is: <repo>/src/shared/python/signal_toolkit/__init__.py
# parents[4] is: <repo> root
_REPO_ROOT = Path(__file__).parents[4]
_VENDOR_ST = (
    _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python" / "signal_toolkit"
)

if not _VENDOR_ST.is_dir():
    raise ImportError(f"vendored signal_toolkit package not found at {_VENDOR_ST}")

__path__ = [str(_VENDOR_ST)]
sys.modules.setdefault("signal_toolkit", sys.modules[__name__])
