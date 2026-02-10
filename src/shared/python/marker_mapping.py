"""Backward compatibility shim - module moved to data_io.marker_mapping."""

import sys as _sys

from .data_io import marker_mapping as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
