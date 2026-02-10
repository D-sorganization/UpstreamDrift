"""Backward compatibility shim - module moved to signal_toolkit.signal_processing."""

import sys as _sys

from .signal_toolkit import signal_processing as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
