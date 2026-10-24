"""MyoSuite golfer grip adapter and bimanual grasp synthesis (HRG-GS-M7, #13329).

Binds ``hand_l``, ``hand_r``, and ``golf_club`` on the MS-51 MyoSuite golfer
scenes (driver and 7-iron) to the shared :mod:`src.shared.python.grip_contact`
architecture in both dual-site weld mode and 6-DoF contact mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final

import numpy as np
from numpy import linalg as npla

from src.engines.physics_engines.myosuite.python.golfer_scene import (
    _DEFAULT_CONTACT,
    REPO_ROOT_DEFAULT,
    ClubKind,
    resolve_golfer_scene,
)
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    SchemaFeatureUnavailableError,
    evaluate_contacts,
    validate_binding,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    contact_topology,
)
from src.shared.python.grip_contact.hand_hand_allocation import (
    AllocatedPairForce,
    AllocationPrior,
    allocate_pair_force,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    map_site_to_hand_frame,
)
from src.shared.python.loop_nullspace.evidence import (
    GripFrame as NullspaceGripFrame,
)
from src.shared.python.loop_nullspace.grip_split import grasp_map
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    resolve_visual_pose,
)

try:
    import mujoco

    from src.engines.physics_engines.mujoco.python.grip_efc import (
        DEFAULT_GRIP_WELDS,
        grip_analysis_from_efc,
    )
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        WeldedGripBodyError,
        _calibration_grip_frames,
        _sample_body_pose_and_vel,
        _state_to_pose_and_vel,
    )
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]
    DEFAULT_GRIP_WELDS = {"L": "grip_weld_l", "R": "grip_weld_r"}

    class WeldedGripBodyError(SchemaFeatureUnavailableError, ValueError):  # type: ignore[no-redef]
        """Raised when physical wrenches or modes are requested on welded grip bodies."""


_MANIFESTS_DIR: Final[Path] = REPO_ROOT_DEFAULT / "src" / "config" / "grip_bindings"
ADMITTED_MYOSUITE_GOLFER_MODELS: Final[tuple[str, ...]] = (
    "myosuite_golfer_driver",
    "myosuite_golfer_iron",
)
_MYOSUITE_MODEL_ID_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "driver": "myosuite_golfer_driver",
        "golfer_myobody_driver": "myosuite_golfer_driver",
        "iron": "myosuite_golfer_iron",
        "golfer_myobody_iron": "myosuite_golfer_iron",
    }
)
MYOSUITE_MANIFEST_PATHS: Final[Mapping[str, Path]] = MappingProxyType(
    {
        model_id: _MANIFESTS_DIR / f"{model_id}.json"
        for model_id in ADMITTED_MYOSUITE_GOLFER_MODELS
    }
)
_MYOSUITE_CONSUMER: Final[str] = "myosuite.golfer_grip_adapter"
_MYOSUITE_CHILD: Final[str] = "HRG-GS-M7"
_MYOSUITE_ISSUE: Final[int] = 13329

_WELD_SITE_PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("grip_weld_l", "grip_site_hand_l", "grip_site_club_l"),
    ("grip_weld_r", "grip_site_hand_r", "grip_site_club_r"),
)

_MODEL_CACHE: dict[tuple[str, bool], tuple[Any, np.ndarray]] = {}


@dataclass(frozen=True)
class BimanualGraspSynthesis:
    """Bimanual grasp synthesis result ``G y = w_club`` (HRG-GS-M7, #13329)."""

    grasp_matrix: np.ndarray
    hand_wrench_vector: np.ndarray
    club_wrench_vector: np.ndarray
    residual_inf_n: float


@dataclass(frozen=True)
class SettledWeldNewtonReport:
    """Newton-Euler balance report for a settled welded club in fixed body pose."""

    club_mass_kg: float
    gravity_m_s2: np.ndarray
    com_position_m: np.ndarray
    com_accel_m_s2: np.ndarray
    lead_force_n: np.ndarray
    trail_force_n: np.ndarray
    net_force_n: np.ndarray
    expected_force_n: np.ndarray
    force_relative_error: float
    net_moment_about_com_nm: np.ndarray
    com_moment_residual_nm: float
    steps: int


def _require_mujoco_sdk() -> Any:
    if mujoco is None:  # pragma: no cover
        raise ImportError("mujoco SDK is required for MyoSuiteGolferGripAdapter")
    return mujoco


def _raise_myosuite_error(cls_err: Any, model_id: str, feature: str) -> None:
    raise cls_err(
        model_id,
        consumer=_MYOSUITE_CONSUMER,
        feature=feature,
        child=_MYOSUITE_CHILD,
        issue=_MYOSUITE_ISSUE,
    )


def _canonical_myosuite_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    resolved = _MYOSUITE_MODEL_ID_ALIASES.get(model_id, model_id)
    if resolved not in ADMITTED_MYOSUITE_GOLFER_MODELS:
        raise ValueError(f"Unknown MyoSuite golfer model_id {model_id!r}")
    return resolved


def _load_myosuite_manifest(
    model_id: str, manifest_override: Mapping[str, Any] | Path | None = None
) -> dict[str, Any]:
    if manifest_override is None:
        path = MYOSUITE_MANIFEST_PATHS[_canonical_myosuite_model_id(model_id)]
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(manifest_override, Path):
        raw = json.loads(manifest_override.read_text(encoding="utf-8"))
    elif isinstance(manifest_override, Mapping):
        raw = dict(manifest_override)
    else:
        raise TypeError(
            f"manifest must be Mapping, Path, or None, got {type(manifest_override).__name__}"
        )
    validate_manifest(raw)
    return dict(raw)


def _validate_bound_bodies_exist(
    model: Any, model_id: str, bodies: tuple[str, ...]
) -> None:
    mj = _require_mujoco_sdk()
    body_obj = mj.mjtObj.mjOBJ_BODY
    for body_id in bodies:
        if int(mj.mj_name2id(model, body_obj, body_id)) < 0:
            _raise_myosuite_error(
                SchemaFeatureUnavailableError,
                model_id,
                f"missing required model body {body_id!r}",
            )


def _resolve_grip_weld_ids(model: Any) -> dict[str, int]:
    mj = _require_mujoco_sdk()
    eq_obj = mj.mjtObj.mjOBJ_EQUALITY
    return {
        s: int(mj.mj_name2id(model, eq_obj, n))
        for s, n in DEFAULT_GRIP_WELDS.items()
        if int(mj.mj_name2id(model, eq_obj, n)) >= 0
    }


def _configure_grip_welds(
    model: Any, data: Any, weld_ids: Mapping[str, int], *, active: bool
) -> None:
    flag = 1 if active else 0
    eq_active0, eq_active = model.eq_active0, data.eq_active
    for eq_id in weld_ids.values():
        eq_active0[eq_id] = flag
        eq_active[eq_id] = flag


def _hand_wrench_namespace(side: str, hw: Any) -> SimpleNamespace:
    return SimpleNamespace(
        hand=side,
        force_n=np.asarray(hw.force_on_club_n, dtype=float).copy(),
        torque_nm=np.asarray(hw.torque_on_club_nm, dtype=float).copy(),
        point_m=np.asarray(hw.point_m, dtype=float).copy(),
    )


def _non_club_dof_and_qpos_indices(
    model: Any, club_body_id: int
) -> tuple[list[int], list[int]]:
    club_jnt = int(model.body_jntadr[club_body_id])
    club_dof = int(model.jnt_dofadr[club_jnt])
    club_qpos = int(model.jnt_qposadr[club_jnt])
    dof_indices = [
        i for i in range(int(model.nv)) if not (club_dof <= i < club_dof + 6)
    ]
    qpos_indices = [
        i for i in range(int(model.nq)) if not (club_qpos <= i < club_qpos + 7)
    ]
    return dof_indices, qpos_indices


class MyoSuiteGolferGripAdapter:
    """MyoSuite golfer :class:`ModelGripAdapter` binding ``hand_l``, ``hand_r``, and ``golf_club`` (HRG-GS-M7, #13329)."""

    def __init__(
        self,
        model: Any,
        data: Any | None = None,
        *,
        manifest: Mapping[str, Any] | Path | None = None,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        contact_mode: bool = False,
    ) -> None:
        mj = _require_mujoco_sdk()
        spec = grip_spec if grip_spec is not None else GripSpec()
        if not isinstance(spec, GripSpec):
            raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")
        if not isinstance(contact_mode, bool):
            raise TypeError("contact_mode must be a bool")

        resolved_manifest = _load_myosuite_manifest("myosuite_golfer_driver", manifest)
        self._manifest = resolved_manifest
        self.model_id = str(resolved_manifest["model_id"])
        self.club_body = str(resolved_manifest["club_body"])
        self.lead_hand_body = str(resolved_manifest["lead_hand_body"])
        self.trail_hand_body = str(resolved_manifest["trail_hand_body"])
        self.model = model
        self.data = data if data is not None else mj.MjData(model)
        _validate_bound_bodies_exist(
            self.model,
            self.model_id,
            (self.club_body, self.lead_hand_body, self.trail_hand_body),
        )
        self._grip_weld_ids = _resolve_grip_weld_ids(self.model)
        self.contact_mode = bool(contact_mode)
        _configure_grip_welds(
            self.model, self.data, self._grip_weld_ids, active=not self.contact_mode
        )
        mj.mj_forward(self.model, self.data)

        self.grip_spec = spec
        self.visual_pose = resolve_visual_pose(spec)
        self.topology = contact_topology(spec)
        self._interface = interface
        self._prescribed_states: BodyStates | None = None
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cal_grip_frames = _calibration_grip_frames(
            self.model, self.club_body, self.lead_hand_body, self.trail_hand_body
        )

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "myosuite_golfer_driver",
        *,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        contact_mode: bool = False,
        models_dir: Path | None = None,
        repo_root: Path | None = None,
    ) -> MyoSuiteGolferGripAdapter:
        """Load an admitted MyoSuite golfer scene (driver or iron) into a :class:`MyoSuiteGolferGripAdapter`."""
        mj = _require_mujoco_sdk()
        canonical_id = _canonical_myosuite_model_id(model_id)
        manifest = _load_myosuite_manifest(canonical_id)
        club_kind = (
            ClubKind.DRIVER if canonical_id.endswith("_driver") else ClubKind.IRON
        )
        scene = resolve_golfer_scene(
            club=club_kind, models_dir=models_dir, repo_root=repo_root
        )
        if scene.is_placeholder:
            raise FileNotFoundError(
                f"MyoSuite golfer scene {canonical_id!r} requires pinned myo_sim assets"
            )
        cache_key = (str(scene.xml_path), bool(contact_mode))
        if cache_key not in _MODEL_CACHE:
            loaded = mj.MjModel.from_xml_path(str(scene.xml_path))
            orig_arm = np.asarray(loaded.dof_armature, dtype=float).copy()
            _MODEL_CACHE[cache_key] = (loaded, orig_arm)
        mj_model, orig_arm = _MODEL_CACHE[cache_key]
        mj_model.dof_armature[:] = orig_arm
        return cls(
            mj_model,
            mj.MjData(mj_model),
            manifest=manifest,
            grip_spec=grip_spec,
            interface=interface,
            contact_mode=contact_mode,
        )

    @property
    def grip_weld_ids(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._grip_weld_ids))

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    def active_grip_weld_names(self) -> tuple[str, ...]:
        """Return the names of currently active grip weld equalities in ``data.eq_active``."""
        eq_active = self.data.eq_active
        return tuple(
            DEFAULT_GRIP_WELDS[s]
            for s in ("L", "R")
            if s in self._grip_weld_ids and int(eq_active[self._grip_weld_ids[s]]) != 0
        )

    def weld_residuals_at_qpos0(self) -> dict[str, tuple[float, float]]:
        """Measure ``qpos0`` site distance (m) and orientation angle (deg) for each club weld."""
        mj = _require_mujoco_sdk()
        model = self.model
        probe = mj.MjData(model)
        mj.mj_forward(model, probe)
        site_obj = mj.mjtObj.mjOBJ_SITE
        site_xpos, site_xmat = probe.site_xpos, probe.site_xmat
        residuals: dict[str, tuple[float, float]] = {}
        for weld_name, hand_site, club_site in _WELD_SITE_PAIRS:
            sid_h = int(mj.mj_name2id(model, site_obj, hand_site))
            sid_c = int(mj.mj_name2id(model, site_obj, club_site))
            pos_h = np.asarray(site_xpos[sid_h], dtype=float)
            pos_c = np.asarray(site_xpos[sid_c], dtype=float)
            dist_m = float(npla.norm(pos_h - pos_c))
            rot_h = np.asarray(site_xmat[sid_h], dtype=float).reshape(3, 3)
            rot_c = np.asarray(site_xmat[sid_c], dtype=float).reshape(3, 3)
            cos_theta = float(
                np.clip((np.trace(rot_h.T @ rot_c) - 1.0) / 2.0, -1.0, 1.0)
            )
            angle_deg = float(np.degrees(np.arccos(cos_theta)))
            residuals[weld_name] = (dist_m, angle_deg)
        return residuals

    @precondition(
        lambda self, steps=1000: not self.contact_mode and int(steps) > 0,
        "settle_welded_club requires weld mode and positive step count",
    )
    def settle_welded_club(self, *, steps: int = 1000) -> SettledWeldNewtonReport:
        """Hold non-club bodies at ``qpos0``, settle the welded club under gravity, and verify Newton-Euler balance."""
        mj = _require_mujoco_sdk()
        model, data = self.model, self.data
        body_obj = mj.mjtObj.mjOBJ_BODY
        club_bid = int(mj.mj_name2id(model, body_obj, self.club_body))
        non_club_dofs, non_club_qpos = _non_club_dof_and_qpos_indices(model, club_bid)

        mj.mj_resetData(model, data)
        _configure_grip_welds(model, data, self._grip_weld_ids, active=True)
        qpos0 = np.asarray(model.qpos0, dtype=float).copy()
        dof_armature = model.dof_armature
        qpos, qvel, ctrl = data.qpos, data.qvel, data.ctrl
        dof_armature[non_club_dofs] = 1e9
        for _ in range(int(steps)):
            qpos[non_club_qpos] = qpos0[non_club_qpos]
            qvel[non_club_dofs] = 0.0
            ctrl[:] = 0.0
            mj.mj_step(model, data)
        qpos[non_club_qpos] = qpos0[non_club_qpos]
        qvel[non_club_dofs] = 0.0
        mj.mj_forward(model, data)
        grip = grip_analysis_from_efc(model, data)

        left_hw, right_hw = grip.left, grip.right
        if left_hw is None or right_hw is None:
            raise ValueError(
                grip.unavailable_reason or "grip weld efc_force unavailable"
            )

        club_jnt = int(model.body_jntadr[club_bid])
        club_dof = int(model.jnt_dofadr[club_jnt])
        m_club = float(model.body_mass[club_bid])
        opt = model.opt
        gravity = np.asarray(opt.gravity, dtype=float).copy()
        com = np.asarray(data.xipos[club_bid], dtype=float).copy()
        a_com = np.asarray(data.qacc[club_dof : club_dof + 3], dtype=float).copy()

        f_l = np.asarray(left_hw.force_on_club_n, dtype=float).copy()
        f_r = np.asarray(right_hw.force_on_club_n, dtype=float).copy()
        t_l = np.asarray(
            left_hw.torque_on_club_nm or (0.0, 0.0, 0.0), dtype=float
        ).copy()
        t_r = np.asarray(
            right_hw.torque_on_club_nm or (0.0, 0.0, 0.0), dtype=float
        ).copy()
        pt_l = np.asarray(left_hw.point_m, dtype=float)
        pt_r = np.asarray(right_hw.point_m, dtype=float)

        f_net = f_l + f_r
        f_expected = m_club * (a_com - gravity)
        rel_err = float(
            npla.norm(f_net - f_expected) / max(float(npla.norm(f_expected)), 1e-12)
        )
        m_com = np.cross(pt_l - com, f_l) + t_l + np.cross(pt_r - com, f_r) + t_r
        return SettledWeldNewtonReport(
            club_mass_kg=m_club,
            gravity_m_s2=gravity,
            com_position_m=com,
            com_accel_m_s2=a_com,
            lead_force_n=f_l,
            trail_force_n=f_r,
            net_force_n=f_net,
            expected_force_n=f_expected,
            force_relative_error=rel_err,
            net_moment_about_com_nm=m_com,
            com_moment_residual_nm=float(npla.norm(m_com)),
            steps=int(steps),
        )

    def _club_grip_frames(self) -> dict[str, np.ndarray]:
        iface = self._interface
        if iface is not None:
            left_f, right_f = iface.left, iface.right
            return {"L": left_f.matrix(), "R": right_f.matrix()}
        return dict(self._cal_grip_frames)

    def capabilities(self) -> GripCapabilities:
        """Return declared :class:`GripCapabilities` for site-weld or contact mode."""
        avail = (self.club_body, self.lead_hand_body, self.trail_hand_body)
        if self.contact_mode:
            return GripCapabilities(
                attachment_modes=("contact", "contact_v1", "bushing"),
                supports_physical_pairs=True,
                supports_wrench_application=True,
                native_contact=False,
                available_bodies=avail,
            )
        welded_reason = (
            f"club body {self.club_body!r} is site-welded to hands "
            f"({self.lead_hand_body!r}, {self.trail_hand_body!r}) via "
            f"grip_weld_l and grip_weld_r in MJCF {self.model_id!r}; "
            "pass contact_mode=True to disable grip welds and enable physical pad contact"
        )
        return GripCapabilities(
            attachment_modes=("weld",),
            supports_physical_pairs=False,
            supports_wrench_application=False,
            native_contact=False,
            available_bodies=avail,
            unavailable_reason=welded_reason,
        )

    def resolve_binding(
        self, topology: ContactTopology, *, attachment_mode: str | None = None
    ) -> HandClubBinding:
        """Resolve and validate a :class:`HandClubBinding` against :meth:`capabilities`."""
        manifest = dict(self._manifest)
        if attachment_mode is not None:
            manifest["attachment_mode"] = attachment_mode
        elif self.contact_mode:
            manifest["attachment_mode"] = "contact"
        grip_frames = self._club_grip_frames()
        binding = binding_from_manifest(manifest, topology, grip_frames)
        validate_resolved_binding(binding, topology, grip_frames)
        validate_binding(topology, binding, self.capabilities())
        return binding

    def set_prescribed_grip_states(
        self,
        *,
        hand_states: Mapping[str, BushingState],
        club_state: RigidBodyState | BushingState,
    ) -> None:
        """Set prescribed hand and club grip states for M3 harness or hold evaluation."""
        for side_key in ("L", "R"):
            if side_key not in hand_states:
                raise ValueError(f"hand_states missing side {side_key!r}")
        c_pose, c_vel = _state_to_pose_and_vel(club_state)
        l_pose, l_vel = _state_to_pose_and_vel(hand_states["L"])
        r_pose, r_vel = _state_to_pose_and_vel(hand_states["R"])
        poses = {
            self.club_body: c_pose,
            self.lead_hand_body: l_pose,
            self.trail_hand_body: r_pose,
        }
        vels = {
            self.club_body: c_vel,
            self.lead_hand_body: l_vel,
            self.trail_hand_body: r_vel,
        }
        self._prescribed_states = BodyStates(poses, velocities=vels)

    def clear_prescribed_grip_states(self) -> None:
        self._prescribed_states = None

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and spatial velocities for bound bodies."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        if self._prescribed_states is not None:
            return self._prescribed_states
        mj = _require_mujoco_sdk()
        mj.mj_forward(self.model, self.data)
        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_id in binding.required_bodies():
            poses[body_id], velocities[body_id] = _sample_body_pose_and_vel(
                self.model, self.data, body_id
            )
        return BodyStates(poses, velocities=velocities)

    def extract_weld_closure(self) -> Any:
        """Extract per-hand weld constraint wrenches from ``data.efc_force`` via :func:`grip_analysis_from_efc`."""
        mj = _require_mujoco_sdk()
        mj.mj_forward(self.model, self.data)
        grip = grip_analysis_from_efc(self.model, self.data)
        left_hw, right_hw = grip.left, grip.right
        if left_hw is None or right_hw is None:
            reason = grip.unavailable_reason or "grip weld efc_force unavailable"
            return SimpleNamespace(available=False, reason=reason, wrenches=[])
        w_left = _hand_wrench_namespace("lead", left_hw)
        w_right = _hand_wrench_namespace("trail", right_hw)
        return SimpleNamespace(
            available=True, reason="", wrenches=[w_left, w_right], grip_analysis=grip
        )

    def synthesise_closure_from_net_wrench(
        self, force_n: np.ndarray | None = None, torque_nm: np.ndarray | None = None
    ) -> Any:
        """Build a closure stub for allocated hand-hand reporting in contact mode."""
        f_vec = np.asarray(
            force_n if force_n is not None else [250.0, -120.0, 80.0], dtype=float
        )
        t_vec = np.asarray(
            torque_nm if torque_nm is not None else [6.0, -4.0, 2.0], dtype=float
        )
        w_item = SimpleNamespace(
            force_n=f_vec.copy(),
            torque_nm=t_vec.copy(),
            point_m=np.zeros(3, dtype=float),
        )
        return SimpleNamespace(available=True, reason="", wrenches=[w_item])

    def allocate_weld_pad_forces(
        self,
        topology: ContactTopology | None = None,
        *,
        prior: AllocationPrior | None = None,
    ) -> dict[str, AllocatedPairForce]:
        """Allocate per-hand site-weld constraint wrenches across pad-grip pairs."""
        topo = topology if topology is not None else self.topology
        mj = _require_mujoco_sdk()
        mj.mj_forward(self.model, self.data)
        grip = grip_analysis_from_efc(self.model, self.data)
        left_hw, right_hw = grip.left, grip.right
        if left_hw is None or right_hw is None:
            reason = grip.unavailable_reason or "grip weld efc_force unavailable"
            return {
                p.name: AllocatedPairForce(
                    available=False,
                    reason=reason,
                    pair_name=p.name,
                    label=p.allocated_label,
                )
                for p in topo.hand_grip_pairs()
            }
        mu = float(_DEFAULT_CONTACT.static_friction)
        results: dict[str, AllocatedPairForce] = {}
        for role, side_char, hw in (("lead", "L", left_hw), ("trail", "R", right_hw)):
            side_pairs = [
                p for p in topo.hand_grip_pairs() if topo.site(p.site_a).hand == role
            ]
            n_side = max(1, len(side_pairs))
            f_vec = np.asarray(hw.force_on_club_n, dtype=float)
            t_vec = np.asarray(hw.torque_on_club_nm or (0.0, 0.0, 0.0), dtype=float)
            ref_pt, c_mag = np.asarray(hw.point_m, dtype=float), float(npla.norm(f_vec))
            for pair in side_pairs:
                mapped = map_site_to_hand_frame(topo, topo.site(pair.site_a), side_char)
                centre = np.asarray(mapped.centre_m, dtype=float)
                rad = np.array([centre[0], 0.0, centre[2]], dtype=float)
                rad_norm = float(npla.norm(rad))
                normal = (
                    rad / rad_norm
                    if rad_norm > 1e-12
                    else np.array([0.0, 0.0, 1.0], dtype=float)
                )
                weight = (
                    float(prior.weights[pair.name])
                    if prior is not None and pair.name in prior.weights
                    else 1.0 / n_side
                )
                results[pair.name] = allocate_pair_force(
                    force_on_lead_n=weight * f_vec,
                    torque_on_lead_nm=weight * t_vec,
                    reference_point_m=ref_pt,
                    patch=(ref_pt + centre, normal),
                    static_friction=mu,
                    pair_name=pair.name,
                    label=pair.allocated_label,
                    closure_force_n=c_mag,
                )
        return results

    def evaluate_contacts(
        self,
        topology_model: TopologyContactModel,
        binding: HandClubBinding | None = None,
        *,
        closure: Any | None = None,
        prior: AllocationPrior | None = None,
    ) -> GripStepReport:
        """Evaluate grip contacts at current sampled states, extracting weld ``efc_force`` in weld mode."""
        rb = (
            binding
            if binding is not None
            else self.resolve_binding(topology_model.topology)
        )
        rc = (
            self.extract_weld_closure()
            if closure is None and not self.contact_mode
            else closure
        )
        return evaluate_contacts(
            topology_model, rb, self.sample_states(rb), closure=rc, prior=prior
        )

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Apply physical body wrenches to ``data.xfrc_applied`` in contact mode."""
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        if not self.contact_mode:
            _raise_myosuite_error(
                WeldedGripBodyError,
                self.model_id,
                f"cannot apply physical wrenches to welded club/hand bodies in {self.model_id!r}",
            )
        active_welds = self.active_grip_weld_names()
        if active_welds:
            _raise_myosuite_error(
                WeldedGripBodyError,
                self.model_id,
                f"cannot apply shared-law wrenches while grip weld equalities {active_welds} are active",
            )
        mj = _require_mujoco_sdk()
        body_obj = mj.mjtObj.mjOBJ_BODY
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        xfrc, xpos, xipos = self.data.xfrc_applied, self.data.xpos, self.data.xipos
        for body_id, pair in wrenches.items():
            f_arr, m_arr = (
                np.asarray(pair[0], dtype=float),
                np.asarray(pair[1], dtype=float),
            )
            if f_arr.shape != (3,) or not bool(np.isfinite(f_arr).all()):
                raise ValueError(f"force for {body_id!r} must be a finite 3-vector")
            if m_arr.shape != (3,) or not bool(np.isfinite(m_arr).all()):
                raise ValueError(f"moment for {body_id!r} must be a finite 3-vector")
            b_idx = int(mj.mj_name2id(self.model, body_obj, body_id))
            if b_idx < 0:
                raise ValueError(f"Body {body_id!r} not found in MjModel")
            arm = np.asarray(xpos[b_idx], dtype=float) - np.asarray(
                xipos[b_idx], dtype=float
            )
            xfrc[b_idx, :3] = f_arr
            xfrc[b_idx, 3:] = m_arr + np.cross(arm, f_arr)
            cleaned[body_id] = (f_arr.copy(), m_arr.copy())
        self._applied_wrenches = cleaned

    @postcondition(
        lambda result: result.residual_inf_n <= 1e-9,
        "bimanual grasp synthesis residual ||G y - w_club||_inf <= 1e-9 N",
    )
    def synthesize_bimanual_grasp(
        self,
        report: GripStepReport | None = None,
        *,
        binding: HandClubBinding | None = None,
    ) -> BimanualGraspSynthesis:
        """Synthesize bimanual grasp map ``G y = w_club`` and verify residual ``<= 1e-9 N``."""
        if not self.contact_mode:
            mj = _require_mujoco_sdk()
            mj.mj_forward(self.model, self.data)
            grip = grip_analysis_from_efc(self.model, self.data)
            left_hw, right_hw = grip.left, grip.right
            if (
                left_hw is None
                or right_hw is None
                or grip.net_force_n is None
                or grip.couple_at_midpoint_nm is None
            ):
                raise ValueError(
                    grip.unavailable_reason
                    or "grip weld efc_force unavailable for synthesis"
                )
            r_l, r_r = (
                np.asarray(left_hw.point_m, dtype=float),
                np.asarray(right_hw.point_m, dtype=float),
            )
            origin = 0.5 * (r_l + r_r)
            g_mat = grasp_map(
                NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=origin),
                r_l,
                r_r,
            )
            y_vec = np.concatenate(
                [
                    left_hw.force_on_club_n,
                    left_hw.torque_on_club_nm or (0.0, 0.0, 0.0),
                    right_hw.force_on_club_n,
                    right_hw.torque_on_club_nm or (0.0, 0.0, 0.0),
                ]
            )
            w_club = np.concatenate([grip.net_force_n, grip.couple_at_midpoint_nm])
            return BimanualGraspSynthesis(
                g_mat, y_vec, w_club, float(np.max(np.abs(g_mat @ y_vec - w_club)))
            )

        if report is None:
            raise ValueError(
                "report is required for bimanual grasp synthesis in contact mode"
            )
        rb = (
            binding
            if binding is not None
            else self.resolve_binding(self.topology, attachment_mode="contact")
        )
        states = self.sample_states(rb)
        r_l = states[self.lead_hand_body][:3, 3].copy()
        r_r = states[self.trail_hand_body][:3, 3].copy()
        r_c = states[self.club_body][:3, 3].copy()
        origin = 0.5 * (r_l + r_r)
        g_mat = grasp_map(
            NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=origin),
            r_l,
            r_r,
        )
        bw = report.body_wrenches()
        f_hl, m_hl = bw.get(self.lead_hand_body, (np.zeros(3), np.zeros(3)))
        f_hr, m_hr = bw.get(self.trail_hand_body, (np.zeros(3), np.zeros(3)))
        f_c, m_c = bw.get(self.club_body, (np.zeros(3), np.zeros(3)))
        y_vec = np.concatenate([-f_hl, -m_hl, -f_hr, -m_hr])
        w_club = np.concatenate([f_c, m_c + np.cross(r_c - origin, f_c)])
        return BimanualGraspSynthesis(
            g_mat, y_vec, w_club, float(np.max(np.abs(g_mat @ y_vec - w_club)))
        )


__all__ = [
    "ADMITTED_MYOSUITE_GOLFER_MODELS",
    "MYOSUITE_MANIFEST_PATHS",
    "BimanualGraspSynthesis",
    "MyoSuiteGolferGripAdapter",
    "SettledWeldNewtonReport",
    "WeldedGripBodyError",
]
