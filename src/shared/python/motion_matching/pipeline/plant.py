"""Engine-independent plant interface and registry for the motion matching pipeline.

Defines the ``MatchingPlant`` protocol (extending ``FullBodyPlant``) and a
fail-closed engine registry mapping engine names ('mujoco', 'drake', 'pinocchio')
to plant instances. Concrete physics engines are imported lazily inside their
respective plant modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.full_body_step import FullBodyPlant

logger = logging.getLogger(__name__)


class EngineUnavailableError(RuntimeError):
    """Raised when an engine is not registered or its native SDK is not installed."""


@runtime_checkable
class MatchingPlant(FullBodyPlant, Protocol):
    """High-level physics plant protocol required by the motion-matching pipeline."""

    @property
    def engine_name(self) -> str:
        """Name of the underlying physics engine (e.g. 'mujoco', 'drake', 'pinocchio')."""
        ...

    @property
    def plant_sha(self) -> str:
        """Deterministic digest of the model/specification powering this plant."""
        ...

    @property
    def ground_plane(self) -> GroundPlane:
        """World contact ground plane."""
        ...

    def create_ik(
        self, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> BaseFullBodyIK:
        """Instantiate an inverse kinematics solver adapter for this plant."""
        ...

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World poses (3x3 rotation, 3D position) for each body named in mapping."""
        ...

    def marker_positions(
        self, q: np.ndarray, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> np.ndarray:
        """Evaluate forward marker positions of shape (markers, 3)."""
        ...

    def contact_forces(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any:
        """Evaluate foot-ground normal and tangential contact forces."""
        ...

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        """Evaluate kinematic loop closure residuals between grip sites."""
        ...

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance plant state [q, v] by one time step dt under efforts tau."""
        ...


PlantFactory = Callable[[bytes | Mapping[str, Any]], MatchingPlant]

_PLANT_REGISTRY: dict[str, PlantFactory] = {}
_BUILTIN_INITIALIZED = False


def _init_builtins() -> None:
    global _BUILTIN_INITIALIZED
    if _BUILTIN_INITIALIZED:
        return
    _BUILTIN_INITIALIZED = True

    def _load_mujoco(spec: bytes | Mapping[str, Any]) -> MatchingPlant:
        from src.shared.python.motion_matching.pipeline.plants.mujoco_plant import (
            MujocoMatchingPlant,
        )

        return MujocoMatchingPlant(spec)

    def _load_drake(spec: bytes | Mapping[str, Any]) -> MatchingPlant:
        from src.shared.python.motion_matching.pipeline.plants.drake_plant import (
            DrakeMatchingPlant,
        )

        return DrakeMatchingPlant(spec)

    def _load_pinocchio(spec: bytes | Mapping[str, Any]) -> MatchingPlant:
        from src.shared.python.motion_matching.pipeline.plants.pinocchio_plant import (
            PinocchioMatchingPlant,
        )

        return PinocchioMatchingPlant(spec)

    _PLANT_REGISTRY["mujoco"] = _load_mujoco
    _PLANT_REGISTRY["drake"] = _load_drake
    _PLANT_REGISTRY["pinocchio"] = _load_pinocchio


@precondition(lambda engine, factory: bool(engine), "Engine name must be non-empty")
def register_plant(engine: str, factory: PlantFactory) -> None:
    """Register a factory for a given physics engine name."""
    _init_builtins()
    _PLANT_REGISTRY[engine.lower()] = factory


@postcondition(
    lambda result: isinstance(result, list), "Must return a list of engine names"
)
def available_engines() -> list[str]:
    """Return all currently registered engine identifiers."""
    _init_builtins()
    return sorted(_PLANT_REGISTRY.keys())


@precondition(lambda engine, spec: bool(engine), "Engine name must be non-empty")
def get_plant(engine: str, spec: bytes | Mapping[str, Any]) -> MatchingPlant:
    """Instantiate a MatchingPlant for the named engine from a specification.

    Raises:
        EngineUnavailableError: If the engine is unknown or its dependencies fail.
    """
    _init_builtins()
    key = engine.lower()
    if key not in _PLANT_REGISTRY:
        known = ", ".join(sorted(_PLANT_REGISTRY.keys()))
        raise EngineUnavailableError(
            f"Unknown engine '{engine}'. Registered engines: [{known}]"
        )
    factory = _PLANT_REGISTRY[key]
    try:
        return factory(spec)
    except ImportError as exc:
        raise EngineUnavailableError(
            f"Engine '{engine}' SDK dependencies are not installed: {exc}"
        ) from exc
    except Exception as exc:
        raise EngineUnavailableError(
            f"Failed to instantiate MatchingPlant for engine '{engine}': {exc}"
        ) from exc


def compute_attachment_marker_positions(
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
) -> np.ndarray:
    """Compute 3D marker positions given body poses and body-local offsets."""
    positions = []
    for label in attachments:
        body, offset = attachments[label]
        rot, trans = poses[body]
        pos = rot @ np.asarray(offset, dtype=float) + trans
        positions.append(pos)
    return np.asarray(positions, dtype=float)


def integrate_euler_step(
    accelerations_fn: Callable[
        [Mapping[str, float], Mapping[str, float], Mapping[str, float]],
        Mapping[str, float],
    ],
    coordinate_order: Sequence[str],
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single semi-implicit Euler integration step for coordinate-mapped plants."""
    q_dict = {c: float(q[i]) for i, c in enumerate(coordinate_order)}
    v_dict = {c: float(v[i]) for i, c in enumerate(coordinate_order)}
    tau_dict = {c: float(tau[i]) for i, c in enumerate(coordinate_order)}
    acc = accelerations_fn(q_dict, v_dict, tau_dict)
    a = np.array([acc[c] for c in coordinate_order], dtype=float)
    next_v = v + a * dt
    next_q = q + next_v * dt
    return next_q, next_v
