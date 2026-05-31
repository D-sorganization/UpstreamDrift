"""Unit tests for the simulation-backend factory and registry.

These tests exercise :func:`make_backend` / :func:`available_backends` purely
through the package's public surface. They confirm that:

* the registry advertises exactly the four planned backends,
* the ODE backend constructs and structurally satisfies both the
  :class:`SimulationBackend` and :class:`DynamicsProvider` Protocols,
* name lookup is case-insensitive and forwards keyword arguments,
* unknown / empty names raise :class:`UnknownBackendError`,
* the GPU backend fails loudly with :class:`BackendNotAvailableError` when the
  optional Warp stack is absent, and
* the MuJoCo backend is a :class:`DynamicsProvider` when ``mujoco`` is present.

No source module is imported other than the package itself, so the lazy-import
contract (importing the package pulls in no GPU/MuJoCo deps) is respected.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.shared.python.simulation_backends import (
    BackendNotAvailableError,
    DynamicsProvider,
    GolfModelParams,
    SimulationBackend,
    UnknownBackendError,
    available_backends,
    has_mujoco,
    has_mjx,
    has_warp,
    make_backend,
)

pytestmark = pytest.mark.unit

#: Dotted path of the optional GPU backend module. It is implemented in a
#: sibling epic task; until it lands, the ``mjwarp`` unavailability contract
#: cannot be exercised, so the relevant test is skipped rather than coupled to
#: an absent module.
_MJWARP_MODULE = "src.shared.python.simulation_backends.mjwarp_backend"


def _mjwarp_backend_present() -> bool:
    """Return whether the lazy ``mjwarp`` backend module is importable yet."""
    return importlib.util.find_spec(_MJWARP_MODULE) is not None


def _params() -> GolfModelParams:
    """Return the canonical default model parameters (shared fixture helper)."""
    return GolfModelParams.default()


def test_available_backends_is_exact_sorted_quad() -> None:
    """The registry advertises exactly the four planned backends, sorted."""
    assert available_backends() == ("mjwarp", "mjx", "mujoco", "ode")


def test_make_ode_satisfies_both_protocols() -> None:
    """The ODE backend is a runtime-checkable SimulationBackend + DynamicsProvider."""
    backend = make_backend("ode", _params())
    assert isinstance(backend, SimulationBackend)
    assert isinstance(backend, DynamicsProvider)


def test_make_backend_name_is_case_insensitive() -> None:
    """An upper-cased name resolves to the same backend (case-insensitive lookup)."""
    backend = make_backend("ODE", _params())
    assert isinstance(backend, SimulationBackend)


def test_make_backend_forwards_kwargs() -> None:
    """Extra keyword arguments are forwarded to the backend constructor."""
    backend = make_backend("ode", _params(), dt=0.02)
    assert isinstance(backend, SimulationBackend)


@pytest.mark.parametrize("bad_name", ["nope", ""])
def test_make_backend_unknown_name_raises(bad_name: str) -> None:
    """Unknown and empty backend names raise UnknownBackendError."""
    with pytest.raises(UnknownBackendError):
        make_backend(bad_name, _params())


@pytest.mark.skipif(
    has_warp(), reason="warp stack installed; cannot assert unavailability"
)
@pytest.mark.skipif(
    not _mjwarp_backend_present(),
    reason="mjwarp backend module not yet implemented (sibling epic task)",
)
def test_make_mjwarp_without_warp_raises_not_available() -> None:
    """Requesting the GPU backend without the Warp stack fails loudly.

    The factory imports the backend module lazily; that module gates on
    :func:`require_warp`, so a missing CUDA/Warp stack surfaces as a
    :class:`BackendNotAvailableError` (not a bare ImportError).
    """
    with pytest.raises(BackendNotAvailableError):
        make_backend("mjwarp", _params())


@pytest.mark.skipif(has_mjx(), reason="MJX/JAX stack installed")
def test_make_mjx_without_mjx_raises_not_available() -> None:
    """Requesting the differentiable backend without MJX/JAX fails clearly."""
    with pytest.raises(BackendNotAvailableError):
        make_backend("mjx", _params())


@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_make_mujoco_is_dynamics_provider() -> None:
    """When mujoco is present, the CPU backend exposes dynamics primitives."""
    backend = make_backend("mujoco", _params())
    assert isinstance(backend, DynamicsProvider)
