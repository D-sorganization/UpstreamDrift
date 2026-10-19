"""Pinocchio reference ModelGripAdapter (HRG-GS-M1, #13323).

Thin reference adapter wrapping the G6 prescribed-motion grip setup
(:class:`~src.engines.physics_engines.pinocchio.python.grip_contact_sim.ClubInHands`
and :class:`~src.shared.python.grip_contact.interface.GripInterface`):
- Resolves a :class:`HandClubBinding` from :class:`GripInterface` frames with
  every site bound as ``rigid_proxy`` on the lead or trail hand body.
- Samples 4x4 world poses and spatial velocities from the hand and club frames.
- Records externally applied wrenches for verification.

Does not import ``pinocchio`` at module load time, so pure state-sampling tests
run in any environment while full ``ClubInHands`` integration runs when
``pinocchio`` is available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src.shared.python.core.handedness import (
    DEFAULT_HANDEDNESS,
    Handedness,
    validate_handedness,
)
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    HandClubBinding,
    bind_rigid_hand_sites,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import ContactTopology
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.pinocchio.python.grip_contact_sim import (
        ClubInHands,
    )

__all__ = ["PinocchioGripAdapter"]

DEFAULT_CLUB_BODY: Final[str] = "club"
DEFAULT_LEAD_HAND_BODY: Final[str] = "lead_hand"
DEFAULT_TRAIL_HAND_BODY: Final[str] = "trail_hand"


def _state_to_pose_and_vel(
    state: BushingState | RigidBodyState,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Convert a BushingState or RigidBodyState into a 4x4 pose and (v, omega)."""
    pose = np.eye(4)
    pose[:3, :3] = state.rotation
    pose[:3, 3] = state.position_m
    vel = (state.velocity_m_s.copy(), state.omega_rad_s.copy())
    return pose, vel


def _default_rigid_state() -> RigidBodyState:
    """Stationary RigidBodyState at world origin with identity orientation."""
    return RigidBodyState(
        rotation=np.eye(3),
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )


class PinocchioGripAdapter:
    """Reference :class:`ModelGripAdapter` wrapping G6 ``ClubInHands`` / ``GripInterface``."""

    def __init__(
        self,
        source: GripInterface | ClubInHands,
        *,
        sim: ClubInHands | None = None,
        model_id: str = "pinocchio_club_in_hands",
        model_hash: str = "full-body-v1",
        handedness: Handedness = DEFAULT_HANDEDNESS,
        attachment_mode: str = "contact",
    ) -> None:
        if isinstance(source, GripInterface):
            self._interface = source
            self._sim: Any = sim
        elif hasattr(source, "interface") and isinstance(
            source.interface, GripInterface
        ):
            self._interface = source.interface
            self._sim = source
        else:
            raise TypeError(
                f"source must be GripInterface or ClubInHands, got {type(source).__name__}"
            )
        validate_handedness(handedness)
        self._model_id = model_id
        self._model_hash = model_hash
        self._handedness: Handedness = handedness
        self._attachment_mode = attachment_mode
        self._club_body = DEFAULT_CLUB_BODY
        self._lead_hand_body = DEFAULT_LEAD_HAND_BODY
        self._trail_hand_body = DEFAULT_TRAIL_HAND_BODY

        init_club = _default_rigid_state()
        init_hands = hand_frame_states(init_club, self._interface)
        self._hand_states: dict[str, BushingState] = dict(init_hands)
        self._club_state: RigidBodyState = init_club
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._wrench_history: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Most recently applied body wrenches ``{body_id: (force_n, moment_nm)}``."""
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    @property
    def wrench_history(self) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
        """History of all ``apply_wrenches`` calls on this adapter."""
        return list(self._wrench_history)

    def capabilities(self) -> GripCapabilities:
        """Return declared capabilities for the Pinocchio reference adapter."""
        return GripCapabilities(
            attachment_modes=("contact", "contact_v1"),
            supports_physical_pairs=True,
            supports_wrench_application=True,
            native_contact=False,
            available_bodies=(
                self._club_body,
                self._lead_hand_body,
                self._trail_hand_body,
            ),
        )

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        """Resolve a :class:`HandClubBinding` from the :class:`GripInterface` frames."""
        if not isinstance(topology, ContactTopology):
            raise TypeError(
                f"topology must be ContactTopology, got {type(topology).__name__}"
            )
        sites = bind_rigid_hand_sites(
            topology,
            lead_hand_body=self._lead_hand_body,
            trail_hand_body=self._trail_hand_body,
        )
        left_frame = self._interface.left
        right_frame = self._interface.right
        grip_frames = {
            "L": left_frame.matrix(),
            "R": right_frame.matrix(),
        }
        return HandClubBinding(
            model_id=self._model_id,
            model_hash=self._model_hash,
            handedness=self._handedness,
            attachment_mode=self._attachment_mode,
            club_body=self._club_body,
            lead_hand_body=self._lead_hand_body,
            trail_hand_body=self._trail_hand_body,
            sites=sites,
            unavailable_sites={},
            club_grip_frames=grip_frames,
        )

    def set_prescribed_states(
        self,
        *,
        weld_club: RigidBodyState | None = None,
        hand_states: Mapping[str, BushingState] | None = None,
        club_state: RigidBodyState | BushingState | None = None,
    ) -> None:
        """Set current hand and club kinematic states directly."""
        if hand_states is not None:
            for side in ("L", "R"):
                if side not in hand_states:
                    raise ValueError(f"hand_states missing side {side!r}")
                st = hand_states[side]
                if not isinstance(st, BushingState):
                    raise TypeError(
                        f"hand_states[{side!r}] must be BushingState, got {type(st).__name__}"
                    )
            self._hand_states = {"L": hand_states["L"], "R": hand_states["R"]}
        elif weld_club is not None:
            if not isinstance(weld_club, RigidBodyState):
                raise TypeError(
                    f"weld_club must be RigidBodyState, got {type(weld_club).__name__}"
                )
            self._hand_states = hand_frame_states(weld_club, self._interface)

        if club_state is not None:
            if isinstance(club_state, RigidBodyState):
                self._club_state = club_state
            elif isinstance(club_state, BushingState):
                self._club_state = RigidBodyState(
                    rotation=club_state.rotation,
                    position_m=club_state.position_m,
                    velocity_m_s=club_state.velocity_m_s,
                    omega_rad_s=club_state.omega_rad_s,
                )
            else:
                raise TypeError("club_state must be RigidBodyState or BushingState")

    def update_from_sim(
        self,
        t: float,
        y: np.ndarray,
        hand_source: Callable[[float], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Update hand and club states from the wrapped ``ClubInHands`` at ``(t, y)``."""
        if self._sim is None:
            raise RuntimeError("PinocchioGripAdapter has no wrapped ClubInHands sim")
        q, qdot = hand_source(t)
        kinematics = self._sim.kinematics
        weld_state = kinematics.state(q, qdot)
        self._hand_states = hand_frame_states(weld_state, self._interface)
        self._club_state = self._sim.club_state(y)

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and velocities for the bound club and hand bodies."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        club_pose, club_vel = _state_to_pose_and_vel(self._club_state)
        lead_pose, lead_vel = _state_to_pose_and_vel(self._hand_states["L"])
        trail_pose, trail_vel = _state_to_pose_and_vel(self._hand_states["R"])
        poses = {
            binding.club_body: club_pose,
            binding.lead_hand_body: lead_pose,
            binding.trail_hand_body: trail_pose,
        }
        velocities = {
            binding.club_body: club_vel,
            binding.lead_hand_body: lead_vel,
            binding.trail_hand_body: trail_vel,
        }
        return BodyStates(poses, velocities=velocities)

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Record external wrenches ``{body_id: (force_n, moment_about_body_origin_nm)}``."""
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_id, pair in wrenches.items():
            f_arr = np.asarray(pair[0], dtype=float)
            m_arr = np.asarray(pair[1], dtype=float)
            if f_arr.shape != (3,) or not bool(np.isfinite(f_arr).all()):
                raise ValueError(f"force for {body_id!r} must be a finite 3-vector")
            if m_arr.shape != (3,) or not bool(np.isfinite(m_arr).all()):
                raise ValueError(f"moment for {body_id!r} must be a finite 3-vector")
            cleaned[body_id] = (f_arr.copy(), m_arr.copy())
        self._applied_wrenches = cleaned
        self._wrench_history.append(cleaned)
