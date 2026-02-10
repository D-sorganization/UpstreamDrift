"""Backward compatibility shim - module moved to config.config_utils."""

import sys as _sys

from .config import config_utils as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
