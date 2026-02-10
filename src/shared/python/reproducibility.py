"""Backward compatibility shim - module moved to data_io.reproducibility."""

import sys as _sys

from .data_io import reproducibility as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
