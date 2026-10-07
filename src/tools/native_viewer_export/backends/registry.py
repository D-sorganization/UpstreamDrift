"""Backend registry: engine name to backend factory."""

from __future__ import annotations

from collections.abc import Callable

from src.tools.native_viewer_export.core import ENGINES, NativeBackend


def _drake() -> NativeBackend:
    from src.tools.native_viewer_export.backends.drake_meshcat import (
        DrakeMeshcatBackend,
    )

    return DrakeMeshcatBackend()


def _pinocchio() -> NativeBackend:
    from src.tools.native_viewer_export.backends.pinocchio_meshcat import (
        PinocchioMeshcatBackend,
    )

    return PinocchioMeshcatBackend()


def _opensim() -> NativeBackend:
    from src.tools.native_viewer_export.backends.opensim_simbody import (
        OpenSimSimbodyBackend,
    )

    return OpenSimSimbodyBackend()


def _myosuite() -> NativeBackend:
    from src.tools.native_viewer_export.backends.myosuite_arena import (
        MyoSuiteArenaBackend,
    )

    return MyoSuiteArenaBackend()


BACKEND_FACTORIES: dict[str, Callable[[], NativeBackend]] = {
    "drake": _drake,
    "pinocchio": _pinocchio,
    "opensim": _opensim,
    "myosuite": _myosuite,
}


def make_backend(engine: str) -> NativeBackend:
    """Instantiate the backend for ``engine`` (lazy imports keep SDKs optional)."""
    try:
        factory = BACKEND_FACTORIES[engine]
    except KeyError:
        raise ValueError(
            f"unknown engine {engine!r}; expected one of {list(ENGINES)}"
        ) from None
    return factory()
