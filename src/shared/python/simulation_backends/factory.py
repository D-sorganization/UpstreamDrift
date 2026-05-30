"""Backend registry and factory.

``make_backend`` is the single entry point for constructing a simulation
backend by name. Backends are imported **lazily** inside the factory so that:

* importing this module never pulls in heavy or optional dependencies
  (``mujoco``, ``mujoco_warp``, CUDA);
* a missing optional GPU stack surfaces as a clear
  :class:`BackendNotAvailableError` *when requested*, not as an import error at
  package load time.

Each registered backend class accepts ``params: GolfModelParams`` as its first
positional argument; any extra keyword arguments are forwarded verbatim.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .exceptions import UnknownBackendError

if TYPE_CHECKING:
    from .model_params import GolfModelParams
    from .protocol import SimulationBackend

#: Registry mapping backend name -> (module suffix, class name). Module suffix
#: is relative to this package. Keep names lowercase (epic convention).
_REGISTRY: dict[str, tuple[str, str]] = {
    "ode": ("ode_backend", "ODEBackend"),
    "mujoco": ("mujoco_backend", "MuJoCoBackend"),
    "mjwarp": ("mjwarp_backend", "MJWarpBackend"),
}


def available_backends() -> tuple[str, ...]:
    """Return the registered backend names (sorted, stable order)."""
    return tuple(sorted(_REGISTRY))


def make_backend(
    name: str,
    params: GolfModelParams,
    **kwargs: object,
) -> SimulationBackend:
    """Construct a simulation backend by name.

    Args:
        name: Backend identifier; one of :func:`available_backends`.
        params: The single-source-of-truth model parameters.
        **kwargs: Backend-specific options forwarded to the constructor.

    Returns:
        A concrete object satisfying the
        :class:`~simulation_backends.protocol.SimulationBackend` Protocol.

    Raises:
        UnknownBackendError: If ``name`` is not registered.
        BackendNotAvailableError: If the backend's optional deps are missing.
    """
    if not isinstance(name, str) or not name:
        raise UnknownBackendError(
            f"backend name must be a non-empty string, got {name!r}"
        )
    key = name.lower()
    if key not in _REGISTRY:
        raise UnknownBackendError(
            f"unknown backend {name!r}; available: {', '.join(available_backends())}"
        )
    module_suffix, class_name = _REGISTRY[key]
    module = importlib.import_module(f"{__package__}.{module_suffix}")
    backend_cls = getattr(module, class_name)
    return backend_cls(params, **kwargs)
