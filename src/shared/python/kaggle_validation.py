"""Backward compatibility shim - module moved to validation_pkg.kaggle_validation."""
import sys as _sys

from .validation_pkg import kaggle_validation as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
