"""Backward compatibility shim - module moved to biomechanics.activation_dynamics."""

import sys as _sys

from .biomechanics import activation_dynamics as _real_module  # noqa: E402
from .biomechanics.activation_dynamics import (  # noqa: F401
    ActivationDynamics,
    logger,
)

_sys.modules[__name__] = _real_module
