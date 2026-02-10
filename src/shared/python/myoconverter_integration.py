"""Backward compatibility shim - module moved to biomechanics.myoconverter_integration."""

import sys as _sys

from .biomechanics import myoconverter_integration as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
