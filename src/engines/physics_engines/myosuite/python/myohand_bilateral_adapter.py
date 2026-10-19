"""MyoHub myoHand articulated bilateral grip adapter (HRG-GS-M8, #13330).

Wraps the composed bilateral ``myoHand`` + free ``golf_club`` MJCF scenes
(:mod:`myohand_bilateral_scene`) into the shared :class:`ModelGripAdapter`
contract:
- Segment-mapped ``grip-binding/v1`` manifests binding 100% of requested
  :class:`ContactTopology` sites onto ``distph/midph/proxph/metacarpal/thumb``
  bodies (``representation="phalange"``) across ``overlap``, ``interlock``,
  and ``ten_finger`` (with and without ``trigger_finger``).
- Style finger pose mapping from :class:`GripPose` (:mod:`model_appearance.grip_pose`
  and :mod:`model_appearance.grip_hand_mesh`) onto the 23 ``myoHand`` joints per
  hand, locked via MuJoCo ``<joint>`` equality constraints in frozen-finger
  acceptance mode while keeping all 78 muscle actuators strictly inactive.
- Shared-law pad-grip contact wrenches applied via ``data.xfrc_applied`` on the
  mapped phalanx/metacarpal/thumb bodies and ``golf_club``, plus physical
  hand-hand contact pairs (``trail_palm_lead_thumb``, ``overlap``, ``interlock``)
  using the M6 pair-contact path.
- Strict mutual exclusion between native MuJoCo contact and shared contact, and
  zero allocated force application.
- M3 qualification harness parity against ``PinocchioGripAdapter`` and
  ``law_level_series`` within 5% peak / 2% RMS, plus Grasp-Resultant and
  Newton-Euler inverse-dynamics verification with honest friction-cone ratios
  and ``INDETERMINATE`` labelling for the hyperstatic per-hand split and squeeze.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Literal

import numpy as np

from src.engines.physics_engines.myosuite.python.golfer_grip_adapter import (
    BimanualGraspSynthesis,
)
from src.engines.physics_engines.myosuite.python.myohand_bilateral_scene import (
    HAND_SCENE_REL_DIR,
    MYOHAND_CANONICAL_JOINTS,
    check_myohand_assets_available,
    generate_myohand_bilateral_scene,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    ModelGripAdapter,
    SchemaFeatureUnavailableError,
    apply_report,
    evaluate_contacts,
    validate_binding,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.hand_hand_allocation import AllocationPrior
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import compute_quantity_error
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
)
from src.shared.python.grip_contact.topology_series import (
    FIXTURE_DT_S,
    TopologyKineticsSeries,
    topology_mapping_hash,
)
from src.shared.python.loop_nullspace.evidence import (
    GripFrame as NullspaceGripFrame,
)
from src.shared.python.loop_nullspace.grip_split import grasp_map
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    HandMeshParams,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.motion_matching.contact_law import ContactParameters
from src.shared.python.motion_matching.grip_contract import GripSpec

__all__ = [
    "FINGER_POSE_JOINT_MAP",
    "GraspResultantNewtonReport",
    "MYOHAND_BILATERAL_MODEL_IDS",
    "MyoHandBilateralGripAdapter",
    "build_prescribed_grip_trajectory",
    "evaluate_grasp_resultant_newton_check",
    "pose_to_myohand_joint_targets",
    "run_myohand_frozen_qualification_series",
]

_THIS_FILE: Final[Path] = Path(__file__).resolve()
_REPO_ROOT: Final[Path] = _THIS_FILE.parents[5]
_MANIFESTS_DIR: Final[Path] = _REPO_ROOT / "src/config/grip_bindings"
_SPEC_DRIVER_PATH: Final[Path] = (
    _REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
_SPEC_IRON_PATH: Final[Path] = (
    _REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_iron7.json"
)

MYOHAND_BILATERAL_MODEL_IDS: Final[tuple[str, ...]] = (
    "myosuite_myohand_bilateral_driver",
    "myosuite_myohand_bilateral_iron",
)

_MODEL_SCENE_FILENAMES: Final[dict[str, str]] = {
    "myosuite_myohand_bilateral_driver": "myohand_bilateral_driver.xml",
    "myosuite_myohand_bilateral_iron": "myohand_bilateral_iron.xml",
}

_MODEL_SPEC_PATHS: Final[dict[str, Path]] = {
    "myosuite_myohand_bilateral_driver": _SPEC_DRIVER_PATH,
    "myosuite_myohand_bilateral_iron": _SPEC_IRON_PATH,
}
_MODEL_CACHE: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}

#: Canonical joint limits [q_min, q_max] (rad) from myohand_body.xml.
_JOINT_LIMITS_RAD: Final[dict[str, tuple[float, float]]] = {
    "pro_sup": (-1.5708, 1.5708),
    "deviation": (-0.174533, 0.436332),
    "flexion": (-0.785398, 0.785398),
    "cmc_abduction": (-0.62, 0.12),
    "cmc_flexion": (-0.75, 0.2618),
    "mp_flexion": (-0.698132, 0.610865),
    "ip_flexion": (-1.39626, 0.261799),
    "mcp2_flexion": (-0.9, 1.5708),
    "mcp2_abduction": (-0.261799, 0.261799),
    "pm2_flexion": (0.0, 1.74533),
    "md2_flexion": (0.0, 1.39626),
    "mcp3_flexion": (-0.9, 1.5708),
    "mcp3_abduction": (-0.261799, 0.261799),
    "pm3_flexion": (0.0, 1.74533),
    "md3_flexion": (0.0, 1.39626),
    "mcp4_flexion": (-0.9, 1.5708),
    "mcp4_abduction": (-0.261799, 0.261799),
    "pm4_flexion": (0.0, 1.74533),
    "md4_flexion": (0.0, 1.39626),
    "mcp5_flexion": (-0.9, 1.5708),
    "mcp5_abduction": (-0.261799, 0.261799),
    "pm5_flexion": (0.0, 1.74533),
    "md5_flexion": (0.0, 1.39626),
}

#: Documented mapping table from GripPose / GripHandMesh parameters onto the 23 myoHand joints.
_JOINT_MAP_ROWS: Final[tuple[tuple[str, str, str], ...]] = (
    ("pro_sup", "radius", "v_azimuth_deg"),
    ("deviation", "lunate", "palm_diagonal_deg"),
    ("flexion", "lunate", "roll_offset_deg"),
    ("cmc_abduction", "firstmc", "thumb_offset_deg"),
    ("cmc_flexion", "firstmc", "trail_lifeline_over_lead_thumb"),
    ("mp_flexion", "proximal_thumb", "thumb_stations_m[0:2]"),
    ("ip_flexion", "distal_thumb", "thumb_stations_m[1:3]"),
    ("mcp2_flexion", "proxph2", "phalanx_lengths_m['index'][0]"),
    ("mcp2_abduction", "proxph2", "trigger_finger / style splay"),
    ("pm2_flexion", "midph2", "phalanx_lengths_m['index'][1]"),
    ("md2_flexion", "distph2", "phalanx_lengths_m['index'][2]"),
    ("mcp3_flexion", "proxph3", "phalanx_lengths_m['middle'][0]"),
    ("mcp3_abduction", "proxph3", "style middle splay"),
    ("pm3_flexion", "midph3", "phalanx_lengths_m['middle'][1]"),
    ("md3_flexion", "distph3", "phalanx_lengths_m['middle'][2]"),
    ("mcp4_flexion", "proxph4", "phalanx_lengths_m['ring'][0]"),
    ("mcp4_abduction", "proxph4", "style ring splay"),
    ("pm4_flexion", "midph4", "phalanx_lengths_m['ring'][1]"),
    ("md4_flexion", "distph4", "phalanx_lengths_m['ring'][2]"),
    ("mcp5_flexion", "proxph5", "phalanx_lengths_m['little'][0] + linked_little"),
    ("mcp5_abduction", "proxph5", "linked_fingers() little splay"),
    ("pm5_flexion", "midph5", "phalanx_lengths_m['little'][1] + linked_little"),
    ("md5_flexion", "distph5", "phalanx_lengths_m['little'][2] + linked_little"),
)
FINGER_POSE_JOINT_MAP: Final[tuple[dict[str, str], ...]] = tuple(
    {"joint": j, "body": b, "source": s} for j, b, s in _JOINT_MAP_ROWS
)


@dataclass(frozen=True)
class GraspResultantNewtonReport:
    """Grasp-resultant inverse-dynamics and Newton-Euler balance report (HRG-GS-M8).

    ``per_hand_split_status`` and ``squeeze_status`` are explicitly labelled
    ``"INDETERMINATE"`` because a hyperstatic two-hand grip has a 6-D internal
    nullspace and cannot uniquely determine per-hand split or internal squeeze
    from club dynamics alone.
    """

    model_id: str
    style: str
    club_mass_kg: float
    force_peak_rel_error: float
    force_rms_rel_error: float
    moment_peak_rel_error: float
    moment_rms_rel_error: float
    newton_force_rel_error: float
    newton_moment_com_residual_nm: float
    net_hand_force_n: tuple[float, float, float]
    required_force_n: tuple[float, float, float]
    net_hand_moment_com_nm: tuple[float, float, float]
    required_moment_com_nm: tuple[float, float, float]
    grasp_synthesis: BimanualGraspSynthesis
    friction_cone_ratios: dict[str, float | None]
    per_hand_split_status: Literal["INDETERMINATE"] = "INDETERMINATE"
    squeeze_status: Literal["INDETERMINATE"] = "INDETERMINATE"


def _clamp_joint(name: str, val_rad: float) -> float:
    lo, hi = _JOINT_LIMITS_RAD[name]
    return float(np.clip(val_rad, lo, hi))


def _hand_finger_targets(
    pose: GripPose,
    hand: Hand,
    *,
    trigger_finger: bool,
    grip_radius_m: float,
    params: HandMeshParams,
) -> dict[str, float]:
    """Compute the 23 joint targets (rad) for one hand from GripPose and HandMeshParams."""
    placed = pose.placement(hand)
    k = params.scale
    clearance = params.pad_clearance_m
    out: dict[str, float] = {
        "pro_sup": _clamp_joint("pro_sup", math.radians(placed.v_azimuth_deg)),
        "deviation": _clamp_joint(
            "deviation", 0.25 * math.radians(pose.palm_diagonal_deg - 40.0)
        ),
        "flexion": _clamp_joint(
            "flexion", 0.25 * math.radians(pose.roll_offset_deg(hand))
        ),
        "cmc_abduction": _clamp_joint(
            "cmc_abduction", -0.35 - math.radians(placed.thumb_offset_deg)
        ),
        "cmc_flexion": _clamp_joint(
            "cmc_flexion",
            -0.25 + (0.05 if pose.trail_lifeline_over_lead_thumb else 0.0),
        ),
        "mp_flexion": _clamp_joint("mp_flexion", 0.28),
        "ip_flexion": _clamp_joint("ip_flexion", -0.32),
    }

    finger_indices = (("index", 2), ("middle", 3), ("ring", 4), ("little", 5))
    lead_r = k * max(params.finger_radius_m["index"], params.finger_radius_m["middle"])
    for fname, idx in finger_indices:
        r_f = k * params.finger_radius_m[fname]
        l0, l1, l2 = (k * v for v in params.phalanx_lengths_m[fname])
        d = grip_radius_m + clearance + r_f
        splay = 0.0
        flex_delta = 0.0
        if hand is Hand.TRAIL and fname == "little" and pose.linked_fingers():
            if pose.style is GripStyle.OVERLAP:
                d = grip_radius_m + clearance + 2.0 * lead_r + clearance + r_f
                splay = 0.18
            elif pose.style is GripStyle.INTERLOCK:
                d = grip_radius_m + clearance + lead_r + r_f
                splay = 0.24
        if hand is Hand.LEAD and fname == "index" and pose.style is GripStyle.INTERLOCK:
            splay = -0.20
        if hand is Hand.TRAIL and fname == "index" and trigger_finger:
            splay = -0.15
            flex_delta = -0.18

        before = 0.5 * l0
        mcp_flex = 2.0 * math.atan2(before, d) + flex_delta
        pip_flex = 2.0 * math.atan2(l0 - before, d) + flex_delta
        dip_flex = 2.0 * math.atan2(max(l1 - 0.5 * before, 0.004), d) + 0.5 * flex_delta
        out[f"mcp{idx}_flexion"] = _clamp_joint(f"mcp{idx}_flexion", mcp_flex)
        out[f"mcp{idx}_abduction"] = _clamp_joint(f"mcp{idx}_abduction", splay)
        out[f"pm{idx}_flexion"] = _clamp_joint(f"pm{idx}_flexion", pip_flex)
        out[f"md{idx}_flexion"] = _clamp_joint(f"md{idx}_flexion", dip_flex)
    return out


def pose_to_myohand_joint_targets(
    pose: GripPose = DEFAULT_GRIP_POSE,
    *,
    trigger_finger: bool = False,
    grip_radius_m: float = 0.014,
    params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS,
) -> dict[Hand, dict[str, float]]:
    """Map a :class:`GripPose` onto the 23 canonical ``myoHand`` joints per hand.

    Preconditions:
        ``pose`` is a :class:`GripPose`; ``grip_radius_m`` is in [0.008, 0.03] m.
    Postconditions:
        Returns ``{Hand.LEAD: {jnt: rad}, Hand.TRAIL: {jnt: rad}}`` with all 23
        canonical joints clamped within ``_JOINT_LIMITS_RAD``.
    """
    if not isinstance(pose, GripPose):
        raise TypeError(f"pose must be a GripPose, got {type(pose).__name__}")
    if not (math.isfinite(grip_radius_m) and 0.008 <= grip_radius_m <= 0.03):
        raise ValueError(
            f"grip_radius_m must be in [0.008, 0.03] m, got {grip_radius_m}"
        )
    return {
        Hand.LEAD: _hand_finger_targets(
            pose,
            Hand.LEAD,
            trigger_finger=trigger_finger,
            grip_radius_m=grip_radius_m,
            params=params,
        ),
        Hand.TRAIL: _hand_finger_targets(
            pose,
            Hand.TRAIL,
            trigger_finger=trigger_finger,
            grip_radius_m=grip_radius_m,
            params=params,
        ),
    }


def _body_transform(data: Any, body_id: int) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    mat[:3, 3] = np.asarray(data.xpos[body_id], dtype=float)
    return mat


def _validate_adapter_args(
    model_id: str,
    grip_spec: GripSpec | None,
    grip_pose: GripPose | None,
    contact_mode: str,
    squeeze_n: float,
) -> None:
    if not isinstance(model_id, str) or model_id not in MYOHAND_BILATERAL_MODEL_IDS:
        raise ValueError(
            f"Unsupported model_id {model_id!r}; expected one of {MYOHAND_BILATERAL_MODEL_IDS}"
        )
    if grip_spec is not None and not isinstance(grip_spec, GripSpec):
        raise TypeError(
            f"grip_spec must be GripSpec or None, got {type(grip_spec).__name__}"
        )
    if grip_pose is not None and not isinstance(grip_pose, GripPose):
        raise TypeError(
            f"grip_pose must be GripPose or None, got {type(grip_pose).__name__}"
        )
    if contact_mode not in ("shared", "native"):
        raise ValueError(
            f"contact_mode must be 'shared' or 'native', got {contact_mode!r}"
        )
    if not math.isfinite(squeeze_n) or squeeze_n <= 0.0:
        raise ValueError(f"squeeze_n must be finite and > 0, got {squeeze_n}")


def _state_to_pose_and_vel(
    state: BushingState | RigidBodyState,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(state.rotation, dtype=float)
    pose[:3, 3] = np.asarray(state.position_m, dtype=float)
    vel = (
        np.asarray(state.velocity_m_s, dtype=float).copy(),
        np.asarray(state.omega_rad_s, dtype=float).copy(),
    )
    return pose, vel


class MyoHandBilateralGripAdapter(ModelGripAdapter):
    """Articulated bilateral MyoHub myoHand grip adapter (HRG-GS-M8, #13330)."""

    def __init__(
        self,
        model_id: str = "myosuite_myohand_bilateral_driver",
        *,
        grip_spec: GripSpec | None = None,
        grip_pose: GripPose | None = None,
        contact_mode: Literal["shared", "native"] = "shared",
        frozen_fingers: bool = True,
        squeeze_n: float = 1100.0,
        law: ContactParameters | None = None,
        repo_root: Path | None = None,
    ) -> None:
        _validate_adapter_args(model_id, grip_spec, grip_pose, contact_mode, squeeze_n)
        import mujoco

        self._mujoco = mujoco
        self._repo_root = _REPO_ROOT if repo_root is None else Path(repo_root)
        check_myohand_assets_available(self._repo_root)

        self._model_id = model_id
        self._grip_spec = grip_spec or GripSpec(
            style=GripStyle.OVERLAP, trigger_finger=True
        )
        self._grip_pose = grip_pose or GripPose(
            style=self._grip_spec.style,
            handedness=self._grip_spec.handedness,
        )
        self._contact_mode: Literal["shared", "native"] = contact_mode
        self._frozen_fingers = bool(frozen_fingers)
        self._squeeze_n = float(squeeze_n)

        manifest_path = _MANIFESTS_DIR / f"{model_id}.json"
        self._manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        validate_manifest(self._manifest)

        spec_dict = json.loads(_MODEL_SPEC_PATHS[model_id].read_text(encoding="utf-8"))
        self._interface = GripInterface.from_spec(spec_dict)
        base_topo = contact_topology(self._grip_spec)
        topo_model = build_topology_model(
            self._interface,
            base_topo,
            squeeze_per_hand_n=self._squeeze_n,
        )
        if law is not None:
            topo_model = replace(topo_model, law=law)
        self._topology_model = topo_model
        self._topology = self._topology_model.topology
        self._binding = self.resolve_binding(self._topology)

        scene_path = (
            self._repo_root / HAND_SCENE_REL_DIR / _MODEL_SCENE_FILENAMES[model_id]
        )
        if not scene_path.is_file():
            generate_myohand_bilateral_scene(repo_root=self._repo_root)

        scene_key = str(scene_path.resolve())
        if scene_key not in _MODEL_CACHE:
            compiled = mujoco.MjModel.from_xml_path(scene_key)
            _MODEL_CACHE[scene_key] = (
                compiled,
                np.asarray(compiled.geom_contype, dtype=int).copy(),
                np.asarray(compiled.geom_conaffinity, dtype=int).copy(),
            )
        self._model, orig_contype, orig_conaff = _MODEL_CACHE[scene_key]
        self._model.geom_contype[:] = orig_contype
        self._model.geom_conaffinity[:] = orig_conaff
        self._data = mujoco.MjData(self._model)
        self._configure_physics_and_pose()
        self._calib_rel_transforms = self._capture_calib_phalanx_transforms()
        self._prescribed_hand_frames: dict[str, BushingState] | None = None
        self._prescribed_club_frames: dict[str, BushingState] | None = None
        self._last_sampled_states: BodyStates | None = None

    @property
    def model_id(self) -> str:
        """Manifest ``model_id``."""
        return self._model_id

    def _configure_physics_and_pose(self) -> None:
        mujoco = self._mujoco
        # Keep muscle actuators strictly inactive (muscle recruitment is out of scope)
        disable_act = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        opt = self._model.opt
        opt.disableflags = int(opt.disableflags) | disable_act
        self._data.ctrl[:] = 0.0

        if self._contact_mode == "shared":
            self._model.geom_contype[:] = 0
            self._model.geom_conaffinity[:] = 0

        targets = pose_to_myohand_joint_targets(
            self._grip_pose,
            trigger_finger=self._grip_spec.trigger_finger,
            grip_radius_m=self._topology.grip_radius_m,
        )
        obj_jnt = mujoco.mjtObj.mjOBJ_JOINT
        obj_eq = mujoco.mjtObj.mjOBJ_EQUALITY
        for hand, prefix in ((Hand.LEAD, "lh_"), (Hand.TRAIL, "rh_")):
            hand_targets = targets[hand]
            for jname, q_target in hand_targets.items():
                full_jnt = f"{prefix}{jname}"
                jid = int(mujoco.mj_name2id(self._model, obj_jnt, full_jnt))
                qadr = int(self._model.jnt_qposadr[jid])
                self._data.qpos[qadr] = q_target
                self._model.qpos0[qadr] = q_target
                eq_id = int(mujoco.mj_name2id(self._model, obj_eq, f"lock_{full_jnt}"))
                if eq_id >= 0:
                    self._model.eq_data[eq_id, 0] = q_target
                    active_flag = 1 if self._frozen_fingers else 0
                    self._model.eq_active0[eq_id] = active_flag
                    self._data.eq_active[eq_id] = active_flag
        mujoco.mj_forward(self._model, self._data)

    def _capture_calib_phalanx_transforms(self) -> dict[str, np.ndarray]:
        mujoco = self._mujoco
        obj_body = mujoco.mjtObj.mjOBJ_BODY
        lead_id = int(
            mujoco.mj_name2id(self._model, obj_body, self._binding.lead_hand_body)
        )
        trail_id = int(
            mujoco.mj_name2id(self._model, obj_body, self._binding.trail_hand_body)
        )
        t_lead_inv = np.linalg.inv(_body_transform(self._data, lead_id))
        t_trail_inv = np.linalg.inv(_body_transform(self._data, trail_id))
        calib: dict[str, np.ndarray] = {}
        for sb in self._binding.sites:
            bid = int(mujoco.mj_name2id(self._model, obj_body, sb.body_id))
            t_b = _body_transform(self._data, bid)
            t_root_inv = t_lead_inv if sb.site_id.startswith("lead_") else t_trail_inv
            calib[sb.body_id] = t_root_inv @ t_b
        return calib

    def interface(self) -> GripInterface:
        """Return the shared :class:`GripInterface` for this model."""
        return self._interface

    def topology(self) -> ContactTopology:
        """Return the active :class:`ContactTopology`."""
        return self._topology

    def topology_model(self) -> TopologyContactModel:
        """Return the active :class:`TopologyContactModel`."""
        return self._topology_model

    def resolve_binding(
        self,
        topology: ContactTopology,
        *,
        attachment_mode: str | None = None,
    ) -> HandClubBinding:
        """Resolve and validate the phalanx-mapped :class:`HandClubBinding`."""
        manifest = dict(self._manifest)
        if attachment_mode is not None:
            manifest["attachment_mode"] = attachment_mode
        binding = binding_from_manifest(manifest, topology, self._interface)
        validate_resolved_binding(binding, topology, self._interface)
        return binding

    def bind_topology(
        self,
        topology: ContactTopology,
        *,
        attachment_mode: str | None = None,
    ) -> HandClubBinding:
        """Alias for :meth:`resolve_binding`."""
        return self.resolve_binding(topology, attachment_mode=attachment_mode)

    def capabilities(self) -> GripCapabilities:
        """Return honest capability descriptor for the selected contact mode."""
        is_shared = self._contact_mode == "shared"
        modes = (
            ("separated_v2", "contact", "contact_v1", "bushing")
            if is_shared
            else ("separated_v2", "contact")
        )
        return GripCapabilities(
            attachment_modes=modes,
            supports_physical_pairs=True,
            supports_wrench_application=is_shared,
            native_contact=not is_shared,
            available_bodies=self._binding.required_bodies(),
            unavailable_reason=(
                None
                if is_shared
                else "contact_mode='native' uses native MuJoCo collision; external wrench application is disabled to prevent double contact."
            ),
        )

    def actuators_active(self) -> bool:
        """Return True if muscle actuation is enabled (always False in M8)."""
        mujoco = self._mujoco
        disable_act = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        opt = self._model.opt
        return (int(opt.disableflags) & disable_act) == 0

    def max_actuator_force_n(self) -> float:
        """Return maximum magnitude of generalized actuator force in qfrc_actuator."""
        return float(np.max(np.abs(self._data.qfrc_actuator)))

    def finger_joint_positions(self) -> np.ndarray:
        """Return vector of all 46 (lh_* and rh_*) hand joint coordinates (rad)."""
        mujoco = self._mujoco
        obj_jnt = mujoco.mjtObj.mjOBJ_JOINT
        vals: list[float] = []
        for prefix in ("lh_", "rh_"):
            for jname in MYOHAND_CANONICAL_JOINTS:
                jid = int(mujoco.mj_name2id(self._model, obj_jnt, f"{prefix}{jname}"))
                vals.append(float(self._data.qpos[int(self._model.jnt_qposadr[jid])]))
        return np.asarray(vals, dtype=float)

    def registered_grip_collision_count(self) -> int:
        """Return number of active collision-enabled geoms in the MuJoCo model."""
        contype = np.asarray(self._model.geom_contype, dtype=int)
        conaff = np.asarray(self._model.geom_conaffinity, dtype=int)
        return int(np.sum((contype != 0) | (conaff != 0)))

    def total_applied_xfrc_force_n(self) -> np.ndarray:
        """Return sum of all applied external forces across all bodies (action-reaction check)."""
        return np.asarray(np.sum(self._data.xfrc_applied[:, :3], axis=0), dtype=float)

    def set_prescribed_grip_state(
        self,
        hand_frames: Mapping[str, BushingState],
        club_frames: Mapping[str, BushingState],
    ) -> None:
        """Set prescribed hand and club grip frame states and sync free golf_club state."""
        self._prescribed_hand_frames = dict(hand_frames)
        self._prescribed_club_frames = dict(club_frames)
        left_club = club_frames["L"]
        left_frame = self._interface.left
        t_grip_l = left_frame.matrix()
        rot_grip_l = t_grip_l[:3, :3]
        pos_grip_l = t_grip_l[:3, 3]
        rot_club = np.asarray(left_club.rotation, dtype=float) @ rot_grip_l.T
        pos_club = np.asarray(left_club.position_m, dtype=float) - rot_club @ pos_grip_l
        omega_club = np.asarray(left_club.omega_rad_s, dtype=float)
        vel_club = np.asarray(left_club.velocity_m_s, dtype=float) - np.cross(
            omega_club, rot_club @ pos_grip_l
        )

        from scipy.spatial.transform import Rotation

        q_xyzw = Rotation.from_matrix(rot_club).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        mujoco = self._mujoco
        jid = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "club_free")
        )
        qadr = int(self._model.jnt_qposadr[jid])
        vadr = int(self._model.jnt_dofadr[jid])
        self._data.qpos[qadr : qadr + 3] = pos_club
        self._data.qpos[qadr + 3 : qadr + 7] = q_wxyz
        self._data.qvel[vadr : vadr + 3] = vel_club
        self._data.qvel[vadr + 3 : vadr + 6] = rot_club.T @ omega_club
        mujoco.mj_forward(self._model, self._data)

    def _current_grip_frames(
        self,
    ) -> tuple[dict[str, BushingState], dict[str, BushingState]]:
        if (
            self._prescribed_hand_frames is not None
            and self._prescribed_club_frames is not None
        ):
            return self._prescribed_hand_frames, self._prescribed_club_frames

        club_state = RigidBodyState(
            rotation=np.eye(3, dtype=float),
            position_m=np.zeros(3, dtype=float),
            velocity_m_s=np.zeros(3, dtype=float),
            omega_rad_s=np.zeros(3, dtype=float),
        )
        base_frames = hand_frame_states(club_state, self._interface)
        preload = np.array([0.0, -0.0015, 0.0], dtype=float)
        hand_frames = {
            side: BushingState(
                rotation=st.rotation.copy(),
                position_m=st.position_m + st.rotation @ preload,
                velocity_m_s=st.velocity_m_s.copy(),
                omega_rad_s=st.omega_rad_s.copy(),
            )
            for side, st in base_frames.items()
        }
        return hand_frames, base_frames

    def _current_club_state(
        self, club_frames: Mapping[str, BushingState]
    ) -> RigidBodyState:
        left_club = club_frames["L"]
        left_frame = self._interface.left
        t_grip_l = left_frame.matrix()
        rot_grip_l = t_grip_l[:3, :3]
        pos_grip_l = t_grip_l[:3, 3]
        rot_club = np.asarray(left_club.rotation, dtype=float) @ rot_grip_l.T
        pos_club = np.asarray(left_club.position_m, dtype=float) - rot_club @ pos_grip_l
        omega_club = np.asarray(left_club.omega_rad_s, dtype=float)
        vel_club = np.asarray(left_club.velocity_m_s, dtype=float) - np.cross(
            omega_club, rot_club @ pos_grip_l
        )
        return RigidBodyState(
            rotation=rot_club,
            position_m=pos_club,
            velocity_m_s=vel_club,
            omega_rad_s=omega_club,
        )

    def _phalanx_state_for_body(
        self,
        body_id: str,
        base_hand: BushingState,
        root_pose: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if self._frozen_fingers:
            return _state_to_pose_and_vel(base_hand)
        mujoco = self._mujoco
        bid = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        b_pose = _body_transform(self._data, bid)
        rel_cur = np.linalg.inv(root_pose) @ b_pose
        rel_cal = self._calib_rel_transforms.get(body_id, rel_cur)
        delta_mj = rel_cur @ np.linalg.inv(rel_cal)
        hand_rb = RigidBodyState(
            rotation=base_hand.rotation,
            position_m=base_hand.position_m,
            velocity_m_s=base_hand.velocity_m_s,
            omega_rad_s=base_hand.omega_rad_s,
        )
        return _state_to_pose_and_vel(hand_rb.frame(delta_mj))

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and spatial velocities for the club and all bound phalanges."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        hand_frames, club_frames = self._current_grip_frames()
        if self._prescribed_hand_frames is None:
            self.set_prescribed_grip_state(hand_frames, club_frames)
        club_state = self._current_club_state(club_frames)

        mujoco = self._mujoco
        obj_body = mujoco.mjtObj.mjOBJ_BODY
        l_id = int(mujoco.mj_name2id(self._model, obj_body, binding.lead_hand_body))
        r_id = int(mujoco.mj_name2id(self._model, obj_body, binding.trail_hand_body))
        l_root = _body_transform(self._data, l_id)
        r_root = _body_transform(self._data, r_id)

        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        c_pose, c_vel = _state_to_pose_and_vel(club_state)
        poses[binding.club_body] = c_pose
        velocities[binding.club_body] = c_vel

        for body_id in binding.required_bodies():
            if body_id == binding.club_body:
                continue
            is_lead = body_id.startswith("lh_") or body_id == binding.lead_hand_body
            side_char = "L" if is_lead else "R"
            root_pose = l_root if is_lead else r_root
            b_pose, b_vel = self._phalanx_state_for_body(
                body_id, hand_frames[side_char], root_pose
            )
            poses[body_id] = b_pose
            velocities[body_id] = b_vel

        states = BodyStates(poses, velocities=velocities)
        self._last_sampled_states = states
        return states

    def synthesise_closure_from_net_wrench(
        self,
        force_n: np.ndarray | None = None,
        torque_nm: np.ndarray | None = None,
    ) -> Any:
        """Build a closure stub for allocated hand-hand reporting in contact mode."""
        f_vec = (
            np.asarray(force_n, dtype=float)
            if force_n is not None
            else np.array([250.0, -120.0, 80.0], dtype=float)
        )
        t_vec = (
            np.asarray(torque_nm, dtype=float)
            if torque_nm is not None
            else np.array([6.0, -4.0, 2.0], dtype=float)
        )
        w_item = SimpleNamespace(
            force_n=f_vec.copy(),
            torque_nm=t_vec.copy(),
            point_m=np.zeros(3, dtype=float),
        )
        return SimpleNamespace(available=True, reason="", wrenches=[w_item])

    def evaluate_contacts(
        self,
        topology_model: TopologyContactModel | None = None,
        binding: HandClubBinding | None = None,
        *,
        closure: Any | None = None,
        prior: AllocationPrior | None = None,
    ) -> GripStepReport:
        """Evaluate physical pad-grip and phalanx hand-hand contact wrenches."""
        resolved_model = (
            topology_model if topology_model is not None else self._topology_model
        )
        resolved_binding = (
            binding
            if binding is not None
            else self.resolve_binding(resolved_model.topology)
        )
        states = self.sample_states(resolved_binding)
        return evaluate_contacts(
            resolved_model,
            resolved_binding,
            states,
            closure=closure,
            prior=prior,
        )

    def apply_wrenches(
        self,
        wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Apply physical body wrenches to ``data.xfrc_applied`` (shared contact mode only)."""
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        if self._contact_mode != "shared":
            raise SchemaFeatureUnavailableError(
                self._model_id,
                consumer="HRG-GS-M8 (#13330)",
                feature=(
                    "cannot apply shared wrenches when native_contact is enabled "
                    "(contact_mode='native')"
                ),
                child="myohand_bilateral_adapter",
                issue=13330,
            )
        mujoco = self._mujoco
        obj_body = mujoco.mjtObj.mjOBJ_BODY
        self._data.xfrc_applied[:] = 0.0
        for body_name, (force_n, moment_nm) in wrenches.items():
            bid = int(mujoco.mj_name2id(self._model, obj_body, body_name))
            if bid < 0:
                raise ValueError(f"Unknown body {body_name!r} in {self._model_id}")
            f_arr = np.asarray(force_n, dtype=float)
            m_arr = np.asarray(moment_nm, dtype=float)
            xpos = np.asarray(self._data.xpos[bid], dtype=float)
            xipos = np.asarray(self._data.xipos[bid], dtype=float)
            m_com = m_arr + np.cross(xpos - xipos, f_arr)
            self._data.xfrc_applied[bid, :3] = f_arr
            self._data.xfrc_applied[bid, 3:6] = m_com

    def apply_report(self, report: GripStepReport) -> None:
        """Validate capabilities and apply physical body wrenches from ``report``."""
        apply_report(self, report)

    def step_with_contacts(self, dt_s: float = FIXTURE_DT_S) -> GripStepReport:
        """Evaluate contacts, apply wrenches, and advance one step while holding frozen fingers."""
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"dt_s must be finite and > 0, got {dt_s}")
        q_frozen = self.finger_joint_positions().copy()
        report = self.evaluate_contacts()
        self.apply_report(report)
        self._data.ctrl[:] = 0.0
        mujoco = self._mujoco
        mujoco.mj_step(self._model, self._data)
        if self._frozen_fingers:
            obj_jnt = mujoco.mjtObj.mjOBJ_JOINT
            idx = 0
            for prefix in ("lh_", "rh_"):
                for jname in MYOHAND_CANONICAL_JOINTS:
                    jid = int(
                        mujoco.mj_name2id(self._model, obj_jnt, f"{prefix}{jname}")
                    )
                    qadr = int(self._model.jnt_qposadr[jid])
                    vadr = int(self._model.jnt_dofadr[jid])
                    self._data.qpos[qadr] = q_frozen[idx]
                    self._data.qvel[vadr] = 0.0
                    idx += 1
            mujoco.mj_forward(self._model, self._data)
        return report

    def club_mass_kg(self) -> float:
        """Return mass (kg) of the free golf_club body."""
        mujoco = self._mujoco
        bid = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, self._binding.club_body
            )
        )
        return float(self._model.body_mass[bid])

    def club_moment_about_com(
        self, force_n: np.ndarray, moment_origin_nm: np.ndarray
    ) -> np.ndarray:
        """Transfer a club wrench from the club body origin (xpos) to the club COM (xipos)."""
        mujoco = self._mujoco
        bid = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, self._binding.club_body
            )
        )
        xpos = np.asarray(self._data.xpos[bid], dtype=float)
        xipos = np.asarray(self._data.xipos[bid], dtype=float)
        return np.asarray(moment_origin_nm, dtype=float) + np.cross(
            xpos - xipos, np.asarray(force_n, dtype=float)
        )

    def club_inverse_dynamics_requirement(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (F_req, M_req_com, M_req_origin) = (m(a - g), I alpha + w x Iw, ...) on golf_club."""
        mujoco = self._mujoco
        mujoco.mj_forward(self._model, self._data)
        mujoco.mj_rnePostConstraint(self._model, self._data)
        bid = int(
            mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, self._binding.club_body
            )
        )
        mass = float(self._model.body_mass[bid])
        cvel = np.asarray(self._data.cvel[bid], dtype=float)
        cacc = np.asarray(self._data.cacc[bid], dtype=float)
        omega_w = cvel[:3]
        alpha_w = cacc[:3]
        a_com_minus_g = cacc[3:6] + np.cross(omega_w, cvel[3:6])
        f_req = mass * a_com_minus_g

        r_imat = np.asarray(self._data.ximat[bid], dtype=float).reshape(3, 3)
        i_com = r_imat @ np.diag(self._model.body_inertia[bid]) @ r_imat.T
        m_req_com = i_com @ alpha_w + np.cross(omega_w, i_com @ omega_w)

        xpos = np.asarray(self._data.xpos[bid], dtype=float)
        xipos = np.asarray(self._data.xipos[bid], dtype=float)
        m_req_origin = m_req_com + np.cross(xipos - xpos, f_req)
        return f_req, m_req_com, m_req_origin


def build_prescribed_grip_trajectory(
    interface: GripInterface,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
) -> tuple[np.ndarray, list[dict[str, BushingState]], list[dict[str, BushingState]]]:
    """Build deterministic 2 ms prescribed hand/club trajectory for M3 parity & Newton checks."""
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    times = np.arange(n_steps, dtype=float) * float(dt_s)
    hand_seq: list[dict[str, BushingState]] = []
    club_seq: list[dict[str, BushingState]] = []

    for t in times:
        phase = 2.0 * math.pi * float(t) / (float(n_steps - 1) * float(dt_s))
        club_state = RigidBodyState(
            rotation=np.eye(3, dtype=float),
            position_m=np.array([0.002 * math.sin(phase), 0.0, 0.0], dtype=float),
            velocity_m_s=np.array([0.02 * math.cos(phase), 0.0, 0.0], dtype=float),
            omega_rad_s=np.array([0.0, 0.0, 0.15 * math.sin(phase)], dtype=float),
        )
        c_frames = hand_frame_states(club_state, interface)
        delta_p = np.array(
            [
                0.0004 * math.sin(phase),
                -0.0015 - 0.0003 * math.cos(phase),
                0.0002 * math.sin(phase),
            ],
            dtype=float,
        )
        delta_v = np.array(
            [
                0.015 * math.cos(phase),
                0.010 * math.sin(phase),
                0.005 * math.cos(phase),
            ],
            dtype=float,
        )
        h_frames = {
            side: BushingState(
                rotation=st.rotation.copy(),
                position_m=st.position_m + st.rotation @ delta_p,
                velocity_m_s=st.velocity_m_s + st.rotation @ delta_v,
                omega_rad_s=st.omega_rad_s.copy(),
            )
            for side, st in c_frames.items()
        }
        hand_seq.append(h_frames)
        club_seq.append(c_frames)
    return times, hand_seq, club_seq


def run_myohand_frozen_qualification_series(
    adapter: MyoHandBilateralGripAdapter,
    hand_frames_seq: Sequence[Mapping[str, BushingState]],
    club_frames_seq: Sequence[Mapping[str, BushingState]],
    times: np.ndarray,
    *,
    attachment_mode: str = "contact",
) -> TopologyKineticsSeries:
    """Evaluate frozen-finger bilateral myoHand on prescribed states and return M3 series."""
    topo_model = adapter.topology_model()
    binding = adapter.resolve_binding(
        adapter.topology(), attachment_mode=attachment_mode
    )
    validate_binding(adapter.topology(), binding, adapter.capabilities())
    closure = (
        adapter.synthesise_closure_from_net_wrench()
        if attachment_mode != "separated_v2"
        else None
    )
    reports: list[GripStepReport] = []
    for h_frames, c_frames in zip(hand_frames_seq, club_frames_seq, strict=True):
        adapter.set_prescribed_grip_state(h_frames, c_frames)
        rep = adapter.evaluate_contacts(topo_model, binding, closure=closure)
        adapter.apply_report(rep)
        reports.append(rep)

    m_hash = topology_mapping_hash(topo_model)
    return TopologyKineticsSeries.from_step_reports(
        times,
        reports,
        engine="myosuite_myohand",
        model_id=binding.model_id,
        mapping_hash=m_hash,
        binding=binding,
    )


def _synthesize_club_grasp(
    report: GripStepReport,
    r_l: np.ndarray,
    r_r: np.ndarray,
) -> BimanualGraspSynthesis:
    sw_map = report.side_wrenches
    sw_l = sw_map["L"]
    sw_r = sw_map["R"]
    f_l = np.asarray(sw_l.force_n, dtype=float)
    tau_l = np.asarray(sw_l.moment_nm, dtype=float)
    f_r = np.asarray(sw_r.force_n, dtype=float)
    tau_r = np.asarray(sw_r.moment_nm, dtype=float)
    r_mid = 0.5 * (r_l + r_r)
    y_vec = np.concatenate([f_l, tau_l, f_r, tau_r])
    g_mat = grasp_map(
        NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=r_mid),
        r_l,
        r_r,
    )
    w_club = g_mat @ y_vec
    f_net = f_l + f_r
    m_mid = np.cross(r_l - r_mid, f_l) + tau_l + np.cross(r_r - r_mid, f_r) + tau_r
    w_direct = np.concatenate([f_net, m_mid])
    res_inf = float(np.max(np.abs(w_club - w_direct)))
    return BimanualGraspSynthesis(g_mat, y_vec, w_direct, res_inf)


def evaluate_grasp_resultant_newton_check(
    adapter: MyoHandBilateralGripAdapter,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
) -> GraspResultantNewtonReport:
    """Verify net club contact wrench against inverse dynamics m(a - g) and moment (HRG-GS-M8)."""
    iface = adapter.interface()
    times, hand_seq, club_seq = build_prescribed_grip_trajectory(
        iface, n_steps=n_steps, dt_s=dt_s
    )
    f_hand_seq = np.zeros((n_steps, 3), dtype=float)
    f_req_seq = np.zeros((n_steps, 3), dtype=float)
    m_hand_com_seq = np.zeros((n_steps, 3), dtype=float)
    m_req_com_seq = np.zeros((n_steps, 3), dtype=float)
    last_report: GripStepReport | None = None
    topo = adapter.topology()
    binding = adapter.resolve_binding(topo)
    club_body = binding.club_body

    for k in range(n_steps):
        adapter.set_prescribed_grip_state(hand_seq[k], club_seq[k])
        rep = adapter.evaluate_contacts()
        adapter.apply_report(rep)
        f_req, m_req_com, _ = adapter.club_inverse_dynamics_requirement()
        body_w = rep.body_wrenches()
        f_net, m_origin = body_w[club_body]
        m_hand_com = adapter.club_moment_about_com(f_net, m_origin)

        f_hand_seq[k] = f_net
        f_req_seq[k] = f_req
        m_hand_com_seq[k] = m_hand_com
        m_req_com_seq[k] = m_req_com
        last_report = rep

    assert last_report is not None
    f_err = compute_quantity_error(f_hand_seq, f_req_seq, name="club_net_force_n")
    m_err = compute_quantity_error(
        m_hand_com_seq, m_req_com_seq, name="club_net_moment_com_nm"
    )

    f_last = f_hand_seq[-1]
    f_req_last = f_req_seq[-1]
    m_last = m_hand_com_seq[-1]
    m_req_last = m_req_com_seq[-1]
    newton_f_rel = float(
        np.linalg.norm(f_last - f_req_last)
        / max(float(np.linalg.norm(f_req_last)), 1e-9)
    )
    newton_m_res = float(np.linalg.norm(m_last - m_req_last))

    topo_model = adapter.topology_model()
    law = topo_model.law
    mu_s = float(law.static_friction)
    cone_ratios: dict[str, float | None] = {}
    for pname, pw in last_report.pair_wrenches.items():
        fn = float(pw.normal_force_mag_n)
        ft = float(pw.tangential_force_mag_n)
        if fn > 1e-12:
            cone_ratios[pname] = float(ft / (mu_s * fn))
        else:
            cone_ratios[pname] = None

    last_club_l = club_seq[-1]["L"]
    last_club_r = club_seq[-1]["R"]
    r_l = np.asarray(last_club_l.position_m, dtype=float)
    r_r = np.asarray(last_club_r.position_m, dtype=float)
    synthesis = _synthesize_club_grasp(last_report, r_l, r_r)
    club_mass = adapter.club_mass_kg()
    style_enum = topo.style

    return GraspResultantNewtonReport(
        model_id=adapter.model_id,
        style=style_enum.value,
        club_mass_kg=club_mass,
        force_peak_rel_error=float(f_err.peak_error),
        force_rms_rel_error=float(f_err.rms_error),
        moment_peak_rel_error=float(m_err.peak_error),
        moment_rms_rel_error=float(m_err.rms_error),
        newton_force_rel_error=newton_f_rel,
        newton_moment_com_residual_nm=newton_m_res,
        net_hand_force_n=(float(f_last[0]), float(f_last[1]), float(f_last[2])),
        required_force_n=(
            float(f_req_last[0]),
            float(f_req_last[1]),
            float(f_req_last[2]),
        ),
        net_hand_moment_com_nm=(float(m_last[0]), float(m_last[1]), float(m_last[2])),
        required_moment_com_nm=(
            float(m_req_last[0]),
            float(m_req_last[1]),
            float(m_req_last[2]),
        ),
        grasp_synthesis=synthesis,
        friction_cone_ratios=cone_ratios,
        per_hand_split_status="INDETERMINATE",
        squeeze_status="INDETERMINATE",
    )
