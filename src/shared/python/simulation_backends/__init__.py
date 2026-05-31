"""GPU-ready, backend-agnostic simulation layer for the golf model.

This package provides a clean abstraction over multiple physics backends for the
double-pendulum / golf-club model:

* ``ode`` — CPU reference backend wrapping the analytical RK4 dynamics.
* ``mujoco`` — CPU MuJoCo backend; also exposes dynamics primitives
  (``M(q)``, bias forces) for independent cross-validation.
* ``mjwarp`` — GPU MuJoCo Warp backend for massively parallel *batched* rollouts
  (optional ``[warp]`` extra; gracefully unavailable without CUDA).

Construct backends via :func:`make_backend`; the model itself is described once
by :class:`GolfModelParams` and rendered to every backend. See ADR-0023 and
``docs/simulation_backends/README.md``.

Importing this package has **no** GPU dependency: optional backends are loaded
lazily by the factory only when requested.
"""

from __future__ import annotations

from .capabilities import (
    has_mujoco,
    has_mjx,
    has_warp,
    require_mujoco,
    require_mjx,
    require_warp,
    warp_device_available,
)
from .exceptions import (
    BackendCapabilityError,
    BackendError,
    BackendNotAvailableError,
    UnknownBackendError,
)
from .factory import available_backends, make_backend
from .model_params import GolfModelParams, LowerSegmentParams, UpperSegmentParams
from .protocol import (
    SCHEMA_VERSION,
    BackendCapabilities,
    BatchedBackend,
    BatchTrace,
    DynamicsProvider,
    SimState,
    SimulationBackend,
    Trace,
)
from .provenance import (
    PROVENANCE_FLAT_PREFIX,
    PROVENANCE_META_KEY,
    ProvenanceStamp,
    attach_provenance_to_checkpoint,
    attach_provenance_to_trace,
    serialize_provenance,
)
from .wrench_extractor import (
    WrenchImpulses,
    compute_wrench_impulses,
    force_torque_from_wrench_array,
    static_support_wrench_trace,
    trace_with_wrench_trace,
    trace_wrench_impulses,
    wrench_array_from_force_torque,
    wrench_array_from_trace,
    wrench_trace_from_array,
    wrench_trace_from_force_torque,
)

__all__ = [
    "SCHEMA_VERSION",
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendError",
    "BackendNotAvailableError",
    "BatchTrace",
    "BatchedBackend",
    "DynamicsProvider",
    "GolfModelParams",
    "LowerSegmentParams",
    "PROVENANCE_FLAT_PREFIX",
    "PROVENANCE_META_KEY",
    "ProvenanceStamp",
    "SimState",
    "SimulationBackend",
    "Trace",
    "UnknownBackendError",
    "UpperSegmentParams",
    "WrenchImpulses",
    "attach_provenance_to_checkpoint",
    "attach_provenance_to_trace",
    "available_backends",
    "compute_wrench_impulses",
    "force_torque_from_wrench_array",
    "has_mujoco",
    "has_mjx",
    "has_warp",
    "make_backend",
    "require_mujoco",
    "require_mjx",
    "require_warp",
    "serialize_provenance",
    "static_support_wrench_trace",
    "trace_with_wrench_trace",
    "trace_wrench_impulses",
    "warp_device_available",
    "wrench_array_from_force_torque",
    "wrench_array_from_trace",
    "wrench_trace_from_array",
    "wrench_trace_from_force_torque",
]
