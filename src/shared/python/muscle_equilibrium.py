"""Backward compatibility shim - module moved to biomechanics.muscle_equilibrium."""

import sys as _sys

from .biomechanics import muscle_equilibrium as _real_module  # noqa: E402
from .biomechanics.muscle_equilibrium import (  # noqa: F401
    INITIAL_GUESS_RATIO,
    MAX_ITERATIONS,
    TOLERANCE,
    EquilibriumSolver,
    compute_equilibrium_state,
    logger,
)

_sys.modules[__name__] = _real_module
