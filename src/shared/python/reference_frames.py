"""Backward compatibility shim - module moved to spatial_algebra.reference_frames."""

import sys as _sys

from .spatial_algebra import reference_frames as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
