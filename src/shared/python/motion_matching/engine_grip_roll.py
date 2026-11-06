"""Engine-agnostic address grip roll from adapter ``BodyStates`` (SHOW-A12, #14737).

Every engine grip adapter (Drake, Pinocchio, OpenSim, MyoSuite, MuJoCo) already
returns 4x4 world poses of the club and hand bodies through
``ModelGripAdapter.sample_states``. This module turns those poses, plus each
binding's hand dorsal axes in the hand body frame, into the
:class:`~.grip_roll.GripPoseState` the one shared metric reads, so no engine
carries its own roll formula or zero (DRY with ``grip_roll``, A7 #14731).

Binding manifest keys (optional, ``grip-binding/v1``):
``hand_dorsal_axis_local = {"lead": [x, y, z], "trail": [x, y, z]}`` is the
back-of-hand direction in each hand body frame; ``club_shaft_axis_local`` is the
butt-to-head direction in the club body frame (default ``[0, 1, 0]``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.grip_contact.binding import BodyStates
from src.shared.python.grip_contact.contact_topology import ContactTopology
from src.shared.python.grip_contact.topology_contact import palm_pad_direction
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
)

Array = NDArray[np.float64]
Vec3 = tuple[float, float, float]

_MIN_NORM = 1e-6
_DEFAULT_SHAFT: Vec3 = (0.0, 1.0, 0.0)


def _vec3(value: Any, name: str) -> Vec3:
    arr = np.asarray(value, dtype=float)
    require(
        arr.shape == (3,)
        and bool(np.isfinite(arr).all())
        and float(np.linalg.norm(arr)) > _MIN_NORM,
        f"{name} must be a finite, non-zero 3-vector",
    )
    unit = arr / np.linalg.norm(arr)
    return (float(unit[0]), float(unit[1]), float(unit[2]))


@dataclass(frozen=True)
class EngineGripAxes:
    """Unit axes, each in its own body frame: the club shaft and both dorsals."""

    club_shaft_local: Vec3
    lead_dorsal_local: Vec3
    trail_dorsal_local: Vec3

    def __post_init__(self) -> None:
        for name in ("club_shaft_local", "lead_dorsal_local", "trail_dorsal_local"):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))


def engine_grip_axes_from_manifest(manifest: Mapping[str, Any]) -> EngineGripAxes:
    """Axes recorded in a ``grip-binding/v1`` manifest.

    Raises ``ValueError`` naming the model when the dorsal axes are absent: they
    are never guessed (an engine whose hand frame does not define one is reported).
    """
    model = manifest.get("model_id", "<unknown>")
    dorsal = manifest.get("hand_dorsal_axis_local")
    if not isinstance(dorsal, Mapping) or not {"lead", "trail"} <= set(dorsal):
        raise ValueError(
            f"binding {model!r} has no hand_dorsal_axis_local with lead and trail"
        )
    return EngineGripAxes(
        club_shaft_local=manifest.get("club_shaft_axis_local", _DEFAULT_SHAFT),
        lead_dorsal_local=dorsal["lead"],
        trail_dorsal_local=dorsal["trail"],
    )


def engine_grip_axes_from_topology(
    topology: ContactTopology, club_shaft_local: Any = _DEFAULT_SHAFT
) -> EngineGripAxes:
    """Axes derived from the contact topology, never stored per engine.

    The dorsal of each hand is ``+u``: outward from the shaft through the palm
    pads (the palm faces the shaft, the back of the hand faces away), in the
    topology grip frame, which a grip-frame engine's hand body shares.
    """
    return EngineGripAxes(
        club_shaft_local=club_shaft_local,
        lead_dorsal_local=palm_pad_direction(topology, "lead"),
        trail_dorsal_local=palm_pad_direction(topology, "trail"),
    )


def engine_grip_axes_from_grip_frame(
    states: BodyStates, binding: Any, topology: ContactTopology
) -> EngineGripAxes:
    """Axes for engines whose hand bodies use the grip frame (``+x`` = shaft).

    The shaft in the club body frame is the lead hand's x column seen from the
    club frame. Precondition: the engine's hand frames follow the grip-frame
    convention of ``contact_topology`` (Pinocchio native: ``R_grip``).
    """
    club = _rotation_of(states, binding.club_body)
    hand_x = _rotation_of(states, binding.lead_hand_body)[:, 0]
    return engine_grip_axes_from_topology(topology, tuple(club.T @ hand_x))


def _shaft_frame(shaft_local: Vec3) -> Array:
    """Rotation ``B`` with column 1 = the shaft axis (club frame the metric expects)."""
    y = np.asarray(shaft_local)
    helper = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(y, helper)
    x /= np.linalg.norm(x)
    return np.column_stack([x, y, np.cross(x, y)])


def _rotation_of(states: BodyStates, body: str) -> Array:
    require(body in states, f"body states are missing {body!r}")
    return np.asarray(states[body], dtype=float)[:3, :3]


def grip_pose_state_from_body_states(
    states: BodyStates,
    binding: Any,
    axes: EngineGripAxes,
    target: TargetFrame | None = None,
) -> GripPoseState:
    """``GripPoseState`` of the club and hands in ``states``.

    Preconditions: ``states`` holds the binding's club, lead-hand and trail-hand
    bodies. The metric's club frame is the engine club frame turned so its y
    column is the engine's shaft axis; the dorsals are re-expressed in it.
    """
    basis = _rotation_of(states, binding.club_body) @ _shaft_frame(
        axes.club_shaft_local
    )

    def dorsal(body: str, local: Vec3) -> Array:
        return basis.T @ (_rotation_of(states, body) @ np.asarray(local))

    return GripPoseState(
        club_rotation=basis,
        lead_dorsal_club=dorsal(binding.lead_hand_body, axes.lead_dorsal_local),
        trail_dorsal_club=dorsal(binding.trail_hand_body, axes.trail_dorsal_local),
        target=target or TargetFrame(),
    )


def engine_address_grip_roll_deg(
    states: BodyStates,
    binding: Any,
    axes: EngineGripAxes,
    target: TargetFrame | None = None,
) -> tuple[float, float]:
    """``(lead, trail)`` roll in degrees, by the shared ``grip_roll`` metric."""
    return address_grip_roll_deg(
        grip_pose_state_from_body_states(states, binding, axes, target)
    )
