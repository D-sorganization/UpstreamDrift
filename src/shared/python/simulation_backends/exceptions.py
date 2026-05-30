"""Typed exception hierarchy for the simulation-backends subsystem.

A small, specific hierarchy keeps error handling narrow (no blind
``except Exception``) and lets callers distinguish *"the backend name is wrong"*
from *"the optional GPU stack is not installed"* from *"this backend cannot do
that"*.
"""

from __future__ import annotations


class BackendError(Exception):
    """Base class for all simulation-backend errors."""


class UnknownBackendError(BackendError, KeyError):
    """Raised when a backend name is not registered in the factory.

    Subclasses :class:`KeyError` so existing ``except KeyError`` sites keep
    working, while new code can catch the precise type.
    """


class BackendNotAvailableError(BackendError):
    """Raised when a backend's optional dependencies are not installed.

    For example, requesting the ``mjwarp`` backend without the ``[warp]`` extra
    (CUDA + ``mujoco_warp`` + ``warp-lang``) installed.
    """


class BackendCapabilityError(BackendError, NotImplementedError):
    """Raised when a backend is asked for a service it does not provide.

    For example, requesting ``mass_matrix`` from a backend whose
    :attr:`~simulation_backends.protocol.BackendCapabilities.provides_dynamics`
    flag is ``False``.
    """
