"""Backward compatibility shim - module moved to data_io.swing_capture_import."""

import sys as _sys

from .data_io import swing_capture_import as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
