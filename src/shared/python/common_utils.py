"""Backward compatibility shim - module moved to data_io.common_utils."""

import sys as _sys

from .data_io import common_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
