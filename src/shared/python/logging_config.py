"""Backward compatibility shim - module moved to logging_pkg.logging_config."""
import sys as _sys

from .logging_pkg import logging_config as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
