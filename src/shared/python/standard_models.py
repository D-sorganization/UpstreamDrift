"""Backward compatibility shim - module moved to config.standard_models."""
import sys as _sys

from .config import standard_models as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
