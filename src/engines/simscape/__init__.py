"""SimscapeAdapter package — Python bridge to MATLAB Simscape Multibody.

This package contains the protocol-compliant skeleton implementing the
``PhysicsEngine`` protocol for the GolfSwing3D_Kinetic Simulink model.

The MATLAB Engine API for Python plumbing inside ``step()`` and
``simulate_with_coefficients()`` is intentionally deferred to issue #4006
(`#037`); this package provides only the lifecycle, contract and metadata
layer so dependent code (issue #038, system-identification, etc.) can be
wired without requiring a MATLAB install on dev/CI machines.

Public surface
--------------
- :class:`~src.engines.simscape.adapter.SimscapeAdapter`
- :class:`~src.engines.simscape._errors.SimscapeNotInstalledError`
- :class:`~src.engines.simscape._errors.SimscapeModelNotFoundError`
- :class:`~src.engines.simscape._errors.SimscapeStateError`
"""

from __future__ import annotations

from src.engines.simscape._errors import (
    SimscapeEngineStartupError,
    SimscapeModelNotFoundError,
    SimscapeNotInstalledError,
    SimscapeSimulationError,
    SimscapeStateError,
)
from src.engines.simscape._output import SimscapeOutput
from src.engines.simscape.adapter import SimscapeAdapter
from src.engines.simscape.pool import PoolConfig, SimscapeAdapterPool

__all__ = [
    "PoolConfig",
    "SimscapeAdapter",
    "SimscapeAdapterPool",
    "SimscapeEngineStartupError",
    "SimscapeModelNotFoundError",
    "SimscapeNotInstalledError",
    "SimscapeOutput",
    "SimscapeSimulationError",
    "SimscapeStateError",
]
