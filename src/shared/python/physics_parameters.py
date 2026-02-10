"""Backward compatibility shim - module moved to physics.physics_parameters."""

import sys as _sys

from .physics import physics_parameters as _real_module  # noqa: E402
from .physics.physics_parameters import (  # noqa: F401
    ParameterCategory,
    PhysicsParameter,
    PhysicsParameterRegistry,
    get_parameter_registry,
    get_registry,
)

_sys.modules[__name__] = _real_module
