"""Backward compatibility shim - module moved to physics.grip_contact_model."""

import sys as _sys

from .physics import grip_contact_model as _real_module  # noqa: E402

_sys.modules[__name__] = _real_module
