"""Native Drake golfer grip adapter and bilateral hold driver (HRG-GS-M10, #13332).

Thin :class:`~src.shared.python.grip_contact.binding.ModelGripAdapter` wrapping
the canonical Drake golf humanoid (:mod:`~src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf`):
- Resolves a :class:`~src.shared.python.grip_contact.binding.HandClubBinding` from
  ``src/config/grip_bindings/drake_native_golfer.json`` via
  :func:`~src.shared.python.grip_contact.binding_manifest.binding_from_manifest`.
- Samples 4x4 world poses and spatial velocities from the Drake ``MultibodyPlant``
  context.
- Applies shared-law wrenches as ``ExternallyAppliedSpatialForce`` on the plant's
  applied spatial force input port (no native collision geometry on grip pairs).
- Enforces physics honesty:
  - ``right_hand`` (trail) is welded to ``club_shaft`` via fixed joint
    ``grip_lead``; its hand-club pairs act within one rigid body and are
    reporting/allocated only, never applied. Calling :meth:`apply_wrenches` or
    :meth:`apply_pair_wrench` for a body or pair welded to ``club_shaft`` raises
    ``ValueError``.
  - ``left_hand`` (lead) under ``attachment="bilateral"`` is a distinct rigid
    body connected to ``club_shaft`` via ``LinearBushingRollPitchYaw``, so its
    pad pairs are physical in parallel with the bushing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.shared.python.core.handedness import (
    Handedness,
    validate_handedness,
)
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    apply_report,
    evaluate_contacts,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
)
from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.pair_contact import PairWrench
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    hand_x_origin,
)
from src.shared.python.grip_contact.topology_series import topology_mapping_hash
from src.shared.python.motion_matching.grip_contract import GripSpec

__all__ = [
    "CLUB_LEAD_GRIP_FRAME",
    "CLUB_TRAIL_GRIP_FRAME",
    "DEFAULT_MANIFEST_PATH",
    "LEAD_HAND_GRIP_FRAME",
    "TRAIL_HAND_GRIP_FRAME",
    "BilateralHoldResult",
    "NativeDrakeGolferGripAdapter",
    "club_grip_frames_for_topology",
    "project_pair_contact_normal",
    "simulate_bilateral_hold",
    "solve_bilateral_grip_pose",
]

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST_PATH: Final[Path] = (
    _REPO_ROOT / "src" / "config" / "grip_bindings" / "drake_native_golfer.json"
)
_DEFAULT_SPEC_PATH: Final[Path] = (
    _REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "full_body_spec_anthro_driver.json"
)

CLUB_LEAD_GRIP_FRAME: Final[str] = "club_lead_grip"
CLUB_TRAIL_GRIP_FRAME: Final[str] = "club_trail_grip"
LEAD_HAND_GRIP_FRAME: Final[str] = "left_hand_grip"
TRAIL_HAND_GRIP_FRAME: Final[str] = "right_hand_grip"

_WELDED_TRAIL_REASON: Final[str] = (
    "trail hand 'right_hand' is welded to 'club_shaft' via fixed joint "
    "'grip_lead'; hand-club pairs act within one rigid body and are "
    "reporting/allocated only"
)


def _drake_mod(name: str) -> Any:
    return import_module(name)


def club_grip_frames_for_topology(
    topology: ContactTopology,
) -> dict[str, np.ndarray]:
    """Return 4x4 club grip frame transforms ``{'L': T_L, 'R': T_R}`` on ``club_shaft``.

    In ``golfer.urdf``, ``club_shaft`` is welded to ``right_hand`` at
    ``(0, 0, -0.10)``, so the trail-hand anchor on ``club_shaft`` sits at
    ``(0, 0, +0.10)`` and the lead-hand anchor is offset along ``+x`` by
    ``hand_x_origin(topology, 'lead') - hand_x_origin(topology, 'trail')``.
    """
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )
    dx = hand_x_origin(topology, "lead") - hand_x_origin(topology, "trail")
    mat_l = np.eye(4)
    mat_l[:3, 3] = np.array([dx, 0.0, 0.10], dtype=float)
    mat_r = np.eye(4)
    mat_r[:3, 3] = np.array([0.0, 0.0, 0.10], dtype=float)
    return {"L": mat_l, "R": mat_r}


def project_pair_contact_normal(
    pw: PairWrench,
    *,
    nhat_ba_w: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Record contact normal force as the projection onto ``nhat_BA_W`` (G0 convention).

    In the G0 Drake convention, ``nhat_BA_W`` is the unit contact normal pointing
    from body B (``club_shaft``) toward body A (hand site) in world axes, so
    ``n_into_club = -nhat_BA_W`` points from the hand site into the club.

    Returns:
        ``(normal_force_n, tangential_force_n, nhat_ba_w)``.
    """
    if not isinstance(pw, PairWrench):
        raise TypeError(f"pw must be PairWrench, got {type(pw).__name__}")
    raw_ba = -pw.normal if nhat_ba_w is None else np.asarray(nhat_ba_w, dtype=float)
    if raw_ba.shape != (3,) or not bool(np.isfinite(raw_ba).all()):
        raise ValueError("nhat_ba_w must be a finite 3-vector")
    norm_ba = float(np.linalg.norm(raw_ba))
    if norm_ba <= 1e-12:
        raise ValueError("nhat_ba_w must have non-zero norm")
    unit_ba = raw_ba / norm_ba
    n_into_club = -unit_ba
    force_on_club = pw.force_b_n
    f_norm = max(0.0, float(np.dot(force_on_club, n_into_club)))
    f_tang = float(np.linalg.norm(force_on_club - f_norm * n_into_club))
    return f_norm, f_tang, unit_ba


def solve_bilateral_grip_pose(
    plant: Any,
    context: Any | None = None,
    *,
    topology: ContactTopology | None = None,
) -> np.ndarray:
    """Solve for joint positions placing ``left_hand`` onto ``club_lead_grip`` on ``club_shaft``.

    Postconditions:
        At the returned configuration ``q``, the world-space position and
        orientation of ``left_hand`` match the ``club_lead_grip`` frame on
        ``club_shaft`` to ``<= 1e-5``.
    """
    ik_mod = _drake_mod("pydrake.multibody.inverse_kinematics")
    math_mod = _drake_mod("pydrake.math")
    solvers_mod = _drake_mod("pydrake.solvers")

    topo = topology if topology is not None else contact_topology()
    dx = hand_x_origin(topo, "lead") - hand_x_origin(topo, "trail")
    club_body = plant.GetBodyByName("club_shaft")
    left_hand_body = plant.GetBodyByName("left_hand")
    club_frame = club_body.body_frame()
    left_frame = left_hand_body.body_frame()

    ik = ik_mod.InverseKinematics(plant)
    target_pt = np.array([dx, 0.0, 0.10], dtype=float)
    zero_pt = np.zeros(3, dtype=float)
    ident_rot = math_mod.RotationMatrix()
    ik.AddPositionConstraint(left_frame, zero_pt, club_frame, target_pt, target_pt)
    ik.AddOrientationConstraint(left_frame, ident_rot, club_frame, ident_rot, 0.0)

    q_ref = (
        plant.GetPositions(context).copy()
        if context is not None
        else plant.GetDefaultPositions().copy()
    )
    prog = ik.prog()
    q_var = ik.q()
    prog.AddBoundingBoxConstraint(q_ref[:7], q_ref[:7], q_var[:7])
    prog.AddQuadraticErrorCost(np.eye(len(q_ref)), q_ref, q_var)
    result = solvers_mod.Solve(prog, q_ref)
    if not result.is_success():  # pragma: no cover
        raise RuntimeError("InverseKinematics failed to find bilateral grip pose")
    q_sol = np.asarray(result.GetSolution(q_var), dtype=float)
    if context is not None:
        plant.SetPositions(context, q_sol)
    return q_sol


def _state_to_pose_and_vel(
    state: BushingState | RigidBodyState,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Convert a BushingState or RigidBodyState into a 4x4 pose and (v, omega)."""
    pose = np.eye(4)
    pose[:3, :3] = state.rotation
    pose[:3, 3] = state.position_m
    vel = (state.velocity_m_s.copy(), state.omega_rad_s.copy())
    return pose, vel


def _invert_frame_offset(
    frame_state: BushingState,
    offset_4x4: np.ndarray,
) -> RigidBodyState:
    """Recover parent body RigidBodyState from a child fixed-offset BushingState."""
    rot_off = offset_4x4[:3, :3]
    pos_off = offset_4x4[:3, 3]
    rot_body = frame_state.rotation @ rot_off.T
    arm = rot_body @ pos_off
    pos_body = frame_state.position_m - arm
    omega_body = frame_state.omega_rad_s.copy()
    vel_body = frame_state.velocity_m_s - cross3(omega_body, arm)
    return RigidBodyState(
        rotation=rot_body,
        position_m=pos_body,
        velocity_m_s=vel_body,
        omega_rad_s=omega_body,
    )


def _eval_plant_body_state(
    plant: Any,
    context: Any,
    body_id: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sample 4x4 world pose and ``(v_world, omega_world)`` for ``body_id`` from Drake."""
    body = plant.GetBodyByName(body_id)
    pose_drake = plant.EvalBodyPoseInWorld(context, body)
    rot_drake = pose_drake.rotation()
    vel_drake = plant.EvalBodySpatialVelocityInWorld(context, body)
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rot_drake.matrix(), dtype=float)
    pose[:3, 3] = np.asarray(pose_drake.translation(), dtype=float)
    v_lin = np.asarray(vel_drake.translational(), dtype=float)
    v_ang = np.asarray(vel_drake.rotational(), dtype=float)
    return pose, (v_lin, v_ang)


class NativeDrakeGolferGripAdapter:
    """Thin :class:`ModelGripAdapter` for the native Drake golf humanoid (#13332)."""

    def __init__(
        self,
        plant: Any,
        context: Any,
        *,
        topology: ContactTopology | None = None,
        interface: GripInterface | None = None,
        manifest_path: Path | str | None = None,
        handedness: Handedness | None = None,
        model_hash: str | None = None,
    ) -> None:
        m_path = (
            Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
        )
        raw_manifest = json.loads(m_path.read_text(encoding="utf-8"))
        if handedness is not None:
            validate_handedness(handedness)
            raw_manifest["handedness"] = handedness
        if model_hash is not None:
            raw_manifest["model_hash"] = model_hash
        validate_manifest(raw_manifest)

        self._plant = plant
        self._context = context
        self._manifest: dict[str, Any] = raw_manifest
        self._topology = topology if topology is not None else contact_topology()
        self._interface = interface
        self._club_body = str(raw_manifest["club_body"])
        self._lead_hand_body = str(raw_manifest["lead_hand_body"])
        self._trail_hand_body = str(raw_manifest["trail_hand_body"])
        self._prescribed_states: BodyStates | None = None
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_spatial_forces: tuple[Any, ...] = ()

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Most recently applied physical body wrenches ``{body_id: (force_n, moment_nm)}``."""
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    @property
    def last_applied_spatial_forces(self) -> tuple[Any, ...]:
        """Tuple of ``ExternallyAppliedSpatialForce`` entries fixed on the plant."""
        return self._last_spatial_forces

    def _welded_to_club_names(self) -> set[str]:
        club = self._plant.GetBodyByName(self._club_body)
        welded = self._plant.GetBodiesWeldedTo(club)
        return {b.name() for b in welded}

    def is_body_welded_to_club(self, body_id: str) -> bool:
        """Return ``True`` if ``body_id`` is ``club_shaft`` or welded to ``club_shaft``."""
        if not isinstance(body_id, str) or not body_id.strip():
            raise ValueError("body_id must be a non-empty string")
        return body_id in self._welded_to_club_names()

    def are_bodies_welded(self, body_a: str, body_b: str) -> bool:
        """Return ``True`` if ``body_a`` and ``body_b`` belong to the same welded sub-body."""
        b_a = self._plant.GetBodyByName(body_a)
        welded = self._plant.GetBodiesWeldedTo(b_a)
        welded_names = {b.name() for b in welded}
        return body_b in welded_names

    def registered_grip_collision_count(self) -> int:
        """Return the number of native grip collision geometries registered on hand-club pairs.

        The shared contact law is evaluated via ``evaluate_contacts`` and applied as
        ``ExternallyAppliedSpatialForce``; zero native grip collision geometries are
        registered for these pairs so contact is never applied twice.
        """
        return 0

    def hand_capabilities(self, hand: str) -> GripCapabilities:
        """Return declared :class:`GripCapabilities` for ``hand`` (``'lead'``/``'L'`` or ``'trail'``/``'R'``).

        Physics honesty rule: ``supports_physical_pairs`` is ``True`` ONLY when
        the hand is a distinct rigid body from ``club_shaft`` and is NOT welded
        to ``club_shaft``.
        """
        if not isinstance(hand, str):
            raise TypeError(f"hand must be a str, got {type(hand).__name__}")
        if hand in ("lead", "L", self._lead_hand_body):
            body_name = self._lead_hand_body
        elif hand in ("trail", "R", self._trail_hand_body):
            body_name = self._trail_hand_body
        else:
            raise ValueError(
                f"hand must be 'lead', 'trail', 'L', 'R', or a bound hand body, got {hand!r}"
            )
        avail = (self._club_body, self._lead_hand_body, self._trail_hand_body)
        is_distinct_unwelded = (
            body_name != self._club_body and not self.is_body_welded_to_club(body_name)
        )
        if is_distinct_unwelded:
            return GripCapabilities(
                attachment_modes=("contact", "contact_v1", "bushing"),
                supports_physical_pairs=True,
                supports_wrench_application=True,
                native_contact=False,
                available_bodies=avail,
            )
        return GripCapabilities(
            attachment_modes=("weld",),
            supports_physical_pairs=False,
            supports_wrench_application=False,
            native_contact=False,
            available_bodies=avail,
        )

    def capabilities(self, hand: str | None = None) -> GripCapabilities:
        """Return adapter capabilities (or per-hand capabilities when ``hand`` is given)."""
        if hand is not None:
            return self.hand_capabilities(hand)
        lead_caps = self.hand_capabilities("lead")
        trail_caps = self.hand_capabilities("trail")
        any_phys = (
            lead_caps.supports_physical_pairs or trail_caps.supports_physical_pairs
        )
        avail = (self._club_body, self._lead_hand_body, self._trail_hand_body)
        if any_phys:
            return GripCapabilities(
                attachment_modes=("contact", "contact_v1", "bushing", "weld"),
                supports_physical_pairs=True,
                supports_wrench_application=True,
                native_contact=False,
                available_bodies=avail,
            )
        return GripCapabilities(
            attachment_modes=("weld",),
            supports_physical_pairs=False,
            supports_wrench_application=False,
            native_contact=False,
            available_bodies=avail,
        )

    def _club_grip_frames(self, topology: ContactTopology) -> dict[str, np.ndarray]:
        if self._interface is not None:
            left_f = self._interface.left
            right_f = self._interface.right
            return {"L": left_f.matrix(), "R": right_f.matrix()}
        return club_grip_frames_for_topology(topology)

    def resolve_binding(
        self,
        topology: ContactTopology,
        *,
        physical_only: bool = False,
    ) -> HandClubBinding:
        """Resolve a :class:`HandClubBinding` from ``drake_native_golfer.json`` and ``topology``.

        When ``physical_only=True``, any site belonging to a hand body welded to
        ``club_shaft`` (the trail ``right_hand``) is marked unavailable with a
        reason so that only distinct unwelded hand-club pairs enter ``body_wrenches()``.
        """
        if not isinstance(topology, ContactTopology):
            raise TypeError(
                f"topology must be ContactTopology, got {type(topology).__name__}"
            )
        manifest = dict(self._manifest)
        unavail = dict(manifest.get("unavailable_sites", {}))
        if physical_only:
            if self.is_body_welded_to_club(self._trail_hand_body):
                for site in topology.sites_for_hand("trail"):
                    unavail[site.name] = _WELDED_TRAIL_REASON
            if self.is_body_welded_to_club(self._lead_hand_body):
                for site in topology.sites_for_hand("lead"):
                    unavail[site.name] = (
                        f"lead hand {self._lead_hand_body!r} is welded to {self._club_body!r}"
                    )
        manifest["unavailable_sites"] = unavail
        grip_frames = self._club_grip_frames(topology)
        return binding_from_manifest(manifest, topology, grip_frames)

    def set_prescribed_grip_states(
        self,
        *,
        hand_states: Mapping[str, BushingState],
        club_frames: Mapping[str, BushingState] | None = None,
        club_state: RigidBodyState | BushingState | None = None,
    ) -> None:
        """Set current hand and club grip states for prescribed-hold evaluation."""
        for side in ("L", "R"):
            if side not in hand_states:
                raise ValueError(f"hand_states missing side {side!r}")
        grip_frames = self._club_grip_frames(self._topology)
        if club_state is not None:
            resolved_club: BushingState | RigidBodyState = club_state
        elif club_frames is not None and "L" in club_frames:
            resolved_club = _invert_frame_offset(club_frames["L"], grip_frames["L"])
        else:
            raise ValueError("either club_state or club_frames must be provided")

        c_pose, c_vel = _state_to_pose_and_vel(resolved_club)
        l_pose, l_vel = _state_to_pose_and_vel(hand_states["L"])
        r_pose, r_vel = _state_to_pose_and_vel(hand_states["R"])
        poses = {
            self._club_body: c_pose,
            self._lead_hand_body: l_pose,
            self._trail_hand_body: r_pose,
        }
        vels = {
            self._club_body: c_vel,
            self._lead_hand_body: l_vel,
            self._trail_hand_body: r_vel,
        }
        self._prescribed_states = BodyStates(poses, velocities=vels)

    def clear_prescribed_grip_states(self) -> None:
        """Clear prescribed grip states and resume sampling directly from Drake ``context``."""
        self._prescribed_states = None

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and spatial velocities for bound bodies."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        if self._prescribed_states is not None:
            return self._prescribed_states

        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_id in binding.required_bodies():
            pose, vel = _eval_plant_body_state(self._plant, self._context, body_id)
            poses[body_id] = pose
            velocities[body_id] = vel
        return BodyStates(poses, velocities=velocities)

    def evaluate_contacts(
        self,
        topology_model: TopologyContactModel,
        binding: HandClubBinding | None = None,
        *,
        closure: Any | None = None,
    ) -> GripStepReport:
        """Evaluate grip contacts at the current sampled states."""
        resolved_binding = (
            binding
            if binding is not None
            else self.resolve_binding(topology_model.topology)
        )
        states = self.sample_states(resolved_binding)
        return evaluate_contacts(
            topology_model, resolved_binding, states, closure=closure
        )

    def apply_pair_wrench(
        self,
        pair_name: str,
        pw: PairWrench,
        binding: HandClubBinding,
    ) -> None:
        """Apply a single pair wrench after verifying its two bodies are not welded together."""
        if not isinstance(pw, PairWrench):
            raise TypeError(f"pw must be PairWrench, got {type(pw).__name__}")
        site_map = binding.site_binding_map()
        pair = self._topology.pair(pair_name)
        if pair.site_a not in site_map:
            raise ValueError(f"site {pair.site_a!r} is not bound in binding")
        body_a = site_map[pair.site_a].body_id
        body_b = (
            binding.club_body
            if pair.site_b == "grip"
            else site_map[pair.site_b].body_id
        )
        if self.are_bodies_welded(body_a, body_b):
            raise ValueError(
                f"cannot apply wrench for pair {pair_name!r}: bodies {body_a!r} "
                f"and {body_b!r} are welded together within one rigid body"
            )
        states = self.sample_states(binding)
        p_a = states[body_a][:3, 3]
        p_b = states[body_b][:3, 3]
        tau_a = pw.moment_a_nm + cross3(pw.point_a_m - p_a, pw.force_a_n)
        tau_b = pw.moment_b_nm + cross3(pw.point_b_m - p_b, pw.force_b_n)
        self.apply_wrenches(
            {body_a: (pw.force_a_n, tau_a), body_b: (pw.force_b_n, tau_b)}
        )

    def apply_wrenches(
        self,
        wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Apply external wrenches ``{body_id: (force_n, moment_about_body_origin_nm)}`` in world axes.

        Raises:
            ValueError: if any hand body welded to ``club_shaft`` (such as
                ``right_hand``) has a wrench to apply, rather than silently
                cancelling internal rigid-body forces.
        """
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        welded_to_club = self._welded_to_club_names() - {self._club_body}
        has_unwelded_hand = any(
            b not in welded_to_club and b != self._club_body for b in wrenches
        )

        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_id, pair in wrenches.items():
            f_arr = np.asarray(pair[0], dtype=float)
            m_arr = np.asarray(pair[1], dtype=float)
            if f_arr.shape != (3,) or not bool(np.isfinite(f_arr).all()):
                raise ValueError(f"force for {body_id!r} must be a finite 3-vector")
            if m_arr.shape != (3,) or not bool(np.isfinite(m_arr).all()):
                raise ValueError(f"moment for {body_id!r} must be a finite 3-vector")
            is_non_zero = bool(np.any(f_arr != 0.0) or np.any(m_arr != 0.0))
            if body_id in welded_to_club:
                if is_non_zero or not has_unwelded_hand:
                    raise ValueError(
                        f"cannot apply physical grip wrench to {body_id!r}: body is "
                        f"welded to {self._club_body!r} (acts within one rigid body)"
                    )
                continue
            cleaned[body_id] = (f_arr.copy(), m_arr.copy())

        plant_mod = _drake_mod("pydrake.multibody.plant")
        mpl_math = _drake_mod("pydrake.multibody.math")
        val_mod = _drake_mod("pydrake.common.value")

        spatial_forces: list[Any] = []
        for body_id, (f_vec, m_vec) in cleaned.items():
            body = self._plant.GetBodyByName(body_id)
            sf = plant_mod.ExternallyAppliedSpatialForce()
            sf.body_index = body.index()
            sf.p_BoBq_B = np.zeros(3, dtype=float)
            sf.F_Bq_W = mpl_math.SpatialForce(tau=m_vec, f=f_vec)
            spatial_forces.append(sf)

        self._applied_wrenches = cleaned
        self._last_spatial_forces = tuple(spatial_forces)
        port = self._plant.get_applied_spatial_force_input_port()
        port.FixValue(self._context, val_mod.AbstractValue.Make(spatial_forces))


@dataclass(frozen=True)
class BilateralHoldResult:
    """Result of simulating a bilateral hold on the native Drake golfer."""

    time_s: np.ndarray
    max_lead_hand_gap_m: float
    reports: tuple[GripStepReport, ...]
    physical_reports: tuple[GripStepReport, ...]
    hand_frames: Mapping[str, tuple[BushingState, ...]]
    club_frames: Mapping[str, tuple[BushingState, ...]]
    club_states: tuple[RigidBodyState, ...]
    binding: HandClubBinding
    physical_binding: HandClubBinding
    adapter: NativeDrakeGolferGripAdapter


def _sample_grip_bushing_states(
    plant: Any,
    context: Any,
    binding: HandClubBinding,
) -> tuple[dict[str, BushingState], dict[str, BushingState], RigidBodyState]:
    """Extract world BushingStates of hand grip frames and club grip frames from Drake."""
    c_pose, (c_v, c_w) = _eval_plant_body_state(plant, context, binding.club_body)
    l_pose, (l_v, l_w) = _eval_plant_body_state(plant, context, binding.lead_hand_body)
    r_pose, (r_v, r_w) = _eval_plant_body_state(plant, context, binding.trail_hand_body)

    club_state = RigidBodyState(
        rotation=c_pose[:3, :3],
        position_m=c_pose[:3, 3],
        velocity_m_s=c_v,
        omega_rad_s=c_w,
    )
    hand_l = BushingState(l_pose[:3, :3], l_pose[:3, 3], l_v, l_w)
    hand_r = BushingState(r_pose[:3, :3], r_pose[:3, 3], r_v, r_w)
    grip_frames = binding.club_grip_frames
    club_l = club_state.frame(grip_frames["L"])
    club_r = club_state.frame(grip_frames["R"])
    return {"L": hand_l, "R": hand_r}, {"L": club_l, "R": club_r}, club_state


def _lock_prescribed_hold_chain(plant: Any, context: Any) -> None:
    """Lock pelvis, spine, and right-arm joints so left-arm moves freely under the bushing."""
    pelvis = plant.GetBodyByName("pelvis")
    pelvis.Lock(context)
    for j_idx in plant.GetJointIndices():
        joint = plant.get_joint(j_idx)
        j_name = joint.name()
        if not j_name.startswith("left_"):
            joint.Lock(context)


def simulate_bilateral_hold(
    *,
    style: GripStyle = GripStyle.OVERLAP,
    trigger_finger: bool = False,
    duration_s: float = 0.04,
    dt_s: float = 0.002,
    squeeze_per_hand_n: float = 1100.0,
) -> BilateralHoldResult:
    """Simulate a prescribed bilateral hold on the native Drake golfer for ``duration_s``.

    The pelvis, spine, and right arm (which carries the welded ``club_shaft``)
    are held at the bilateral address pose while the 8-DOF left arm is free under
    gravity, connected to ``club_shaft`` by the bilateral ``LinearBushingRollPitchYaw``
    in parallel with the physical lead-hand pad wrenches applied via
    :class:`NativeDrakeGolferGripAdapter`.
    """
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        load_humanoid_into_plant,
    )

    plant_mod = _drake_mod("pydrake.multibody.plant")
    analysis_mod = _drake_mod("pydrake.systems.analysis")

    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    spec = json.loads(_DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
    interface = GripInterface.from_spec(spec)
    topo_model = build_topology_model(
        interface, topo, squeeze_per_hand_n=squeeze_per_hand_n
    )
    mapping_hash = topology_mapping_hash(topo_model)

    plant = plant_mod.MultibodyPlant(1.0e-3)
    load_humanoid_into_plant(plant, attachment="bilateral", topology=topo)
    plant.Finalize()

    simulator = analysis_mod.Simulator(plant)
    context = simulator.get_mutable_context()
    q_init = solve_bilateral_grip_pose(plant, context, topology=topo)
    plant.SetPositions(context, q_init)
    plant.SetVelocities(context, np.zeros(plant.num_velocities()))
    _lock_prescribed_hold_chain(plant, context)

    adapter = NativeDrakeGolferGripAdapter(
        plant, context, topology=topo, model_hash=mapping_hash
    )
    binding = adapter.resolve_binding(topo, physical_only=False)
    phys_binding = adapter.resolve_binding(topo, physical_only=True)

    n_steps = int(round(duration_s / dt_s)) + 1
    times = np.arange(n_steps, dtype=float) * dt_s
    reports: list[GripStepReport] = []
    phys_reports: list[GripStepReport] = []
    hand_l_seq: list[BushingState] = []
    hand_r_seq: list[BushingState] = []
    club_l_seq: list[BushingState] = []
    club_r_seq: list[BushingState] = []
    club_state_seq: list[RigidBodyState] = []
    max_gap_m = 0.0

    for k, t in enumerate(times):
        if k > 0:
            simulator.AdvanceTo(float(t))
        phys_rep = evaluate_contacts(
            topo_model, phys_binding, adapter.sample_states(phys_binding)
        )
        full_rep = evaluate_contacts(
            topo_model, binding, adapter.sample_states(binding)
        )
        hands, clubs, c_state = _sample_grip_bushing_states(plant, context, binding)
        gap = float(np.linalg.norm(hands["L"].position_m - clubs["L"].position_m))
        max_gap_m = max(max_gap_m, gap)

        reports.append(full_rep)
        phys_reports.append(phys_rep)
        hand_l_seq.append(hands["L"])
        hand_r_seq.append(hands["R"])
        club_l_seq.append(clubs["L"])
        club_r_seq.append(clubs["R"])
        club_state_seq.append(c_state)

    apply_report(adapter, phys_reports[-1])

    return BilateralHoldResult(
        time_s=times,
        max_lead_hand_gap_m=max_gap_m,
        reports=tuple(reports),
        physical_reports=tuple(phys_reports),
        hand_frames={"L": tuple(hand_l_seq), "R": tuple(hand_r_seq)},
        club_frames={"L": tuple(club_l_seq), "R": tuple(club_r_seq)},
        club_states=tuple(club_state_seq),
        binding=binding,
        physical_binding=phys_binding,
        adapter=adapter,
    )
