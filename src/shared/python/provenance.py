"""Backward compatibility shim - module moved to data_io.provenance."""
import sys as _sys

from .data_io import provenance as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
