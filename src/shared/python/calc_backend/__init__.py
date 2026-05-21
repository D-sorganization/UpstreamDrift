"""Compatibility entrypoint for the vendored Tools calculation backend.

UpstreamDrift imports shared code through ``src.shared.python`` while the
provider package in ``vendor/ud-tools`` is laid out as ``calc_backend``.  This
shim keeps the local namespace stable and delegates submodule loading to the
vendored implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SUITE_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_BACKEND = (
    _SUITE_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python" / "calc_backend"
)

if not _VENDORED_BACKEND.is_dir():  # pragma: no cover - repository layout guard
    msg = f"vendored calc_backend package not found at {_VENDORED_BACKEND}"
    raise ImportError(msg)

__path__ = [str(_VENDORED_BACKEND)]

# Some vendored modules use absolute ``calc_backend.*`` imports. Point those at
# this package so both import spellings resolve through the same provider path.
sys.modules.setdefault("calc_backend", sys.modules[__name__])

from .protocols import CalculationEngine, ExpressionEvaluator, ValidationMixin

__version__ = "1.0.0"

__all__ = [
    "CalculationEngine",
    "ExpressionEvaluator",
    "ValidationMixin",
]
