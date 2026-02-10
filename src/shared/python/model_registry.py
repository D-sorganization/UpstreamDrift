"""Backward compatibility shim - module moved to config.model_registry."""
import sys as _sys

from .config import model_registry as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
