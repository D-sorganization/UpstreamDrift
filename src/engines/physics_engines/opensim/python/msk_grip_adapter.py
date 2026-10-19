"""OpenSim MSK grip adapter and hold simulation (HRG-GS-M4, #13326).

Split from ``msk_club`` (which builds the club XML) to keep both modules within
the file-size budget. This module admits an OpenSim golf model for grip
contact, binds the shared ``grip_contact`` topology to its hands and club, and
runs the weld/bushing hold simulation against the shared contact evaluation.
"""

from __future__ import annotations

import copy
import json
import tempfile
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # construction only
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as SafeET
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.msk_club import (
    CLUB_BODY,
    DEFAULT_GRIP_MODEL,
    GRAVITY_M_S2,
    GRIP_BUSHINGS,
    GRIP_MODELS,
    LEAD_JOINT,
    MANIFESTS_DIR,
    MODELS_DIR,
    TRAIL_CONSTRAINT,
    GripCalibration,
    MskClub,
    _objects,
    attach_club,
    grip_model_of,
    load_calibration,
    load_msk_club,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.grip_contact import (
    BushingState,
    ClubDynamics,
    ContactTopology,
    GripCapabilities,
    GripStepReport,
    GripStyle,
    HandClubBinding,
    ModelGripAdapter,
    TopologyContactModel,
    apply_report,
    build_topology_model,
    bushing_wrench,
    evaluate_contacts,
    topology_side_wrench,
    validate_binding,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
)
from src.shared.python.grip_contact.parity import QuantityError, compute_quantity_error
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)


@dataclass(frozen=True)
class OpenSimModelAdmission:
    """Admission record for an OpenSim model under the two-hand grip contract."""

    model_id: str
    manifest_path: Path
    osim_path: Path | None
    calibration_model: str
    lead_hand_body: str
    trail_hand_body: str
    club_body: str
    locked_forearm_wrist_coordinates: tuple[str, ...]
    note: str


ADMITTED_OPENSIM_MODELS: dict[str, OpenSimModelAdmission] = {
    "golf_humanoid": OpenSimModelAdmission(
        "golf_humanoid",
        MANIFESTS_DIR / "opensim_golf_humanoid.json",
        MODELS_DIR / "golf_humanoid.osim",
        "golf_humanoid",
        "hand_l",
        "hand_r",
        CLUB_BODY,
        (),
        "Rajagopal full-body golf model with unlocked forearm and wrist coordinates.",
    ),
    "golf_humanoid_scaled": OpenSimModelAdmission(
        "golf_humanoid_scaled",
        MANIFESTS_DIR / "opensim_golf_humanoid_scaled.json",
        MODELS_DIR / "golf_humanoid_scaled.osim",
        "golf_humanoid_scaled",
        "hand_l",
        "hand_r",
        CLUB_BODY,
        (),
        "Scaled Rajagopal full-body golf model with unlocked forearm and wrist coordinates.",
    ),
    "bruno2015_fullbody_os4_v2": OpenSimModelAdmission(
        "bruno2015_fullbody_os4_v2",
        MANIFESTS_DIR / "bruno2015_fullbody_os4_v2.json",
        None,
        "golf_humanoid",
        "hand_L",
        "hand_R",
        CLUB_BODY,
        (
            "pro_sup_l",
            "wrist_flex_l",
            "wrist_dev_l",
            "pro_sup_r",
            "wrist_flex_r",
            "wrist_dev_r",
        ),
        "Bruno 2015 thoracolumbar full-body OpenSim 4 model; forearm (pro_sup_l/r) and wrist (wrist_flex_l/r, wrist_dev_l/r) are locked by default.",
    ),
}


def _raise_unavailable(model_id: str, feature: str) -> None:
    raise SchemaFeatureUnavailableError(
        model_id,
        consumer="opensim.msk_club",
        feature=feature,
        child="HRG-GS-M4",
        issue=13326,
    )


def admit_opensim_grip_model(model_id: str) -> OpenSimModelAdmission:
    """Return the admission record for ``model_id`` or fail closed."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if model_id not in ADMITTED_OPENSIM_MODELS:
        _raise_unavailable(
            model_id,
            f"OpenSim model {model_id!r} lacks admitted two-hand grip bodies (hand_l/hand_r or hand_L/hand_R)",
        )
    return ADMITTED_OPENSIM_MODELS[model_id]


def _hand_interface_transforms(
    club: MskClub, calibration: GripCalibration
) -> dict[str, np.ndarray]:
    """Compute 4x4 transform of each hand's ``GripInterface`` frame in the hand body."""
    transforms: dict[str, np.ndarray] = {}
    iface = club.interface
    for side in ("L", "R"):
        rot_t = club.grip_rotation.T
        inv_shaft = np.eye(4)
        inv_shaft[:3, :3] = rot_t
        inv_shaft[:3, 3] = -(rot_t @ club.grip_points[side])
        side_iface = iface.frame(side)
        hand_shaft = calibration.hand_frames[side]
        transforms[side] = hand_shaft @ inv_shaft @ side_iface.matrix()
    return transforms


def _try_import_opensim() -> Any | None:
    """Import ``opensim`` if available as a real runtime module."""
    try:
        osim = import_opensim()

        if hasattr(osim, "Model"):
            return osim
    except ImportError:
        pass
    return None


def _build_opensim_runtime(
    model_tree: ET.ElementTree,
    lead_hand_body: str,
    trail_hand_body: str,
    grip_model: str,
) -> tuple[Any | None, Any | None, dict[str, Any], dict[str, Any]]:
    """Instantiate an OpenSim ``Model``, state, bushings, and ``PrescribedForce`` map."""
    osim = _try_import_opensim()
    if osim is None:
        return None, None, {}, {}
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".osim", dir=MODELS_DIR, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        model_tree.write(tmp, encoding="utf-8", xml_declaration=True)
    try:
        osim_model = osim.Model(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    prescribed: dict[str, Any] = {}
    bushings: dict[str, Any] = {}
    if grip_model == "bushing":
        force_set = osim_model.getForceSet()
        body_set = osim_model.getBodySet()
        for side, b_name in GRIP_BUSHINGS.items():
            bushings[side] = osim.BushingForce.safeDownCast(force_set.get(b_name))
        for body_name in (lead_hand_body, trail_hand_body, CLUB_BODY):
            pf = osim.PrescribedForce(
                f"shared_grip_force_{body_name}", body_set.get(body_name)
            )
            pf.set_pointIsGlobal(False)
            pf.set_forceIsGlobal(True)
            pf.setForceFunctions(
                osim.Constant(0.0), osim.Constant(0.0), osim.Constant(0.0)
            )
            pf.setPointFunctions(
                osim.Constant(0.0), osim.Constant(0.0), osim.Constant(0.0)
            )
            pf.setTorqueFunctions(
                osim.Constant(0.0), osim.Constant(0.0), osim.Constant(0.0)
            )
            osim_model.addForce(pf)
            prescribed[body_name] = pf
    state = osim_model.initSystem()
    osim_model.realizeVelocity(state)
    return osim_model, state, bushings, prescribed


def _osim_body_state(
    osim_model: Any, state: Any, body_name: str, local_offset: np.ndarray
) -> BushingState:
    """Read an OpenSim body's world pose and velocity transformed by ``local_offset``."""
    body_set = osim_model.getBodySet()
    body = body_set.get(body_name)
    xform = body.getTransformInGround(state)
    rot_osim, pos_osim = xform.R(), xform.p()
    rot_body = np.array(
        [[float(rot_osim.get(i, j)) for j in range(3)] for i in range(3)]
    )
    pos_body = np.array([float(pos_osim.get(i)) for i in range(3)])
    vel_spatial = body.getVelocityInGround(state)
    ang_osim, lin_osim = vel_spatial.get(0), vel_spatial.get(1)
    omega_w = np.array([float(ang_osim.get(i)) for i in range(3)])
    vel_w = np.array([float(lin_osim.get(i)) for i in range(3)])
    r_world = rot_body @ local_offset[:3, 3]
    return BushingState(
        rot_body @ local_offset[:3, :3],
        pos_body + r_world,
        vel_w + np.cross(omega_w, r_world),
        omega_w,
    )


def _update_prescribed_force(
    pf: Any, force_w: np.ndarray, moment_w: np.ndarray, point_local: np.ndarray
) -> None:
    """Update the constant force, torque, and application point of a ``PrescribedForce``."""
    osim = _try_import_opensim()
    if osim is None:
        return
    cast = osim.Constant.safeDownCast
    f_set, t_set, p_set = (
        pf.get_forceFunctions(),
        pf.get_torqueFunctions(),
        pf.get_pointFunctions(),
    )
    for idx in range(3):
        cast(f_set.get(idx)).setValue(float(force_w[idx]))
        cast(t_set.get(idx)).setValue(float(moment_w[idx]))
        cast(p_set.get(idx)).setValue(float(point_local[idx]))


class OpenSimMskGripAdapter(ModelGripAdapter):
    """Native OpenSim MSK ``ModelGripAdapter`` for both ``weld`` and ``bushing`` modes."""

    def __init__(
        self,
        model_id: str,
        club: MskClub,
        calibration: GripCalibration,
        manifest: Mapping[str, Any],
        *,
        grip_model: str = DEFAULT_GRIP_MODEL,
        osim_runtime: tuple[Any | None, Any | None, dict[str, Any], dict[str, Any]]
        | None = None,
    ) -> None:
        if grip_model not in GRIP_MODELS:
            raise ValueError(
                f"grip_model must be one of {GRIP_MODELS}, got {grip_model!r}"
            )
        validate_manifest(manifest)
        self.model_id = model_id
        self.club = club
        self.calibration = calibration
        self.manifest = dict(manifest)
        self.grip_model = grip_model
        self.lead_hand_body = str(manifest["lead_hand_body"])
        self.trail_hand_body = str(manifest["trail_hand_body"])
        self.club_body = str(manifest["club_body"])
        self._hand_iface_transforms = _hand_interface_transforms(club, calibration)
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._bushings_enabled = grip_model == "bushing"
        if osim_runtime is None:
            self.osim_model, self.state = None, None
            self._bushings: dict[str, Any] = {}
            self._prescribed: dict[str, Any] = {}
        else:
            self.osim_model, self.state, self._bushings, self._prescribed = osim_runtime

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        *,
        club: str = "driver",
        grip_model: str = DEFAULT_GRIP_MODEL,
        grip_style: GripStyle | str = GripStyle.OVERLAP,
        trigger_finger: bool = False,
    ) -> OpenSimMskGripAdapter:
        """Build an adapter for an admitted OpenSim model ID."""
        admission = admit_opensim_grip_model(model_id)
        msk = load_msk_club(club, grip_style=grip_style, trigger_finger=trigger_finger)
        calib = load_calibration(admission.calibration_model, club)
        m_path = admission.manifest_path
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
        runtime = None
        if admission.osim_path is not None and admission.osim_path.is_file():
            raw_tree = SafeET.parse(str(admission.osim_path))
            root = copy.deepcopy(raw_tree.getroot())
            model_el = root.find("Model")
            if model_el is None:
                raise ValueError(f"No <Model> in {admission.osim_path}")
            attach_club(model_el, msk, calib, grip_model=grip_model)
            runtime = _build_opensim_runtime(
                ET.ElementTree(root),
                admission.lead_hand_body,
                admission.trail_hand_body,
                grip_model,
            )
        return cls(
            model_id, msk, calib, manifest, grip_model=grip_model, osim_runtime=runtime
        )

    @classmethod
    def from_xml_element(
        cls,
        model_el: ET.Element,
        *,
        model_id: str = "golf_humanoid",
        club: str = "driver",
        grip_style: GripStyle | str = GripStyle.OVERLAP,
        trigger_finger: bool = False,
    ) -> OpenSimMskGripAdapter:
        """Build an adapter from an in-memory OpenSim ``<Model>`` XML element."""
        body_names = {b.get("name") for b in _objects(model_el, "BodySet")}
        has_lower = "hand_l" in body_names and "hand_r" in body_names
        has_upper = "hand_L" in body_names and "hand_R" in body_names
        if not (has_lower or has_upper):
            _raise_unavailable(
                model_id,
                f"XML model {model_id!r} is missing required hand bodies (hand_l/hand_r)",
            )
        manifest_id = "bruno2015_fullbody_os4_v2" if has_upper else "golf_humanoid"
        admission = admit_opensim_grip_model(manifest_id)
        m_path = admission.manifest_path
        manifest = dict(json.loads(m_path.read_text(encoding="utf-8")))
        manifest["model_id"] = model_id
        msk = load_msk_club(club, grip_style=grip_style, trigger_finger=trigger_finger)
        calib = load_calibration(admission.calibration_model, club)
        return cls(model_id, msk, calib, manifest, grip_model=grip_model_of(model_el))

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Copy of the most recently applied physical body wrenches."""
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    def native_bushings_enabled(self) -> bool:
        """Whether the native OpenSim ``BushingForce`` elements are currently enabled."""
        if self.osim_model is not None and self.state is not None and self._bushings:
            return any(
                bool(bf.appliesForce(self.state)) for bf in self._bushings.values()
            )
        return self._bushings_enabled

    def capabilities(self) -> GripCapabilities:
        """Return declared capabilities for the active club attachment mode."""
        bodies = (self.club_body, self.lead_hand_body, self.trail_hand_body)
        if self.grip_model == "weld":
            return GripCapabilities(
                attachment_modes=("weld",),
                supports_physical_pairs=False,
                supports_wrench_application=True,
                native_contact=False,
                available_bodies=bodies,
                unavailable_reason=(
                    f"OpenSim MSK weld mode attaches {self.club_body} with lead "
                    f"WeldJoint ({LEAD_JOINT}) and trail WeldConstraint "
                    f"({TRAIL_CONSTRAINT}); pairs are allocated/reporting only."
                ),
            )
        return GripCapabilities(
            attachment_modes=("contact", "bushing"),
            supports_physical_pairs=True,
            supports_wrench_application=True,
            native_contact=False,
            available_bodies=bodies,
        )

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        """Resolve and validate the :class:`HandClubBinding` for ``topology``."""
        iface = self.club.interface
        raw_binding = binding_from_manifest(self.manifest, topology, iface)
        mode = "weld" if self.grip_model == "weld" else "contact"
        binding = replace(raw_binding, attachment_mode=mode)
        validate_binding(topology, binding, self.capabilities())
        return binding

    def sample_states(
        self, binding: HandClubBinding
    ) -> Mapping[str, np.ndarray | BushingState]:
        """Sample world states for ``lead_hand_body``, ``trail_hand_body``, and ``club_body``."""
        if self.osim_model is not None and self.state is not None:
            return {
                binding.lead_hand_body: _osim_body_state(
                    self.osim_model,
                    self.state,
                    binding.lead_hand_body,
                    self._hand_iface_transforms["L"],
                ),
                binding.trail_hand_body: _osim_body_state(
                    self.osim_model,
                    self.state,
                    binding.trail_hand_body,
                    self._hand_iface_transforms["R"],
                ),
                binding.club_body: _osim_body_state(
                    self.osim_model, self.state, binding.club_body, np.eye(4)
                ),
            }
        club_pose = np.asarray(self.calibration.club_in_ground, dtype=float)
        iface = self.club.interface
        left_frame, right_frame = iface.left, iface.right
        return {
            binding.lead_hand_body: club_pose @ left_frame.matrix(),
            binding.trail_hand_body: club_pose @ right_frame.matrix(),
            binding.club_body: club_pose.copy(),
        }

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Apply physical body wrenches or raise ``ValueError`` in ``weld`` mode."""
        if self.grip_model == "weld":
            raise ValueError(
                f"Cannot apply grip wrenches to welded bodies {sorted(wrenches)} "
                f"in OpenSim weld mode (WeldJoint {LEAD_JOINT} + WeldConstraint {TRAIL_CONSTRAINT})."
            )
        self._bushings_enabled = False
        if self.osim_model is not None and self.state is not None:
            for bf in self._bushings.values():
                bf.setAppliesForce(self.state, False)
        local_points = {
            self.lead_hand_body: self._hand_iface_transforms["L"][:3, 3],
            self.trail_hand_body: self._hand_iface_transforms["R"][:3, 3],
            self.club_body: np.zeros(3),
        }
        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_name, (force, moment) in wrenches.items():
            f_arr, m_arr = (
                np.asarray(force, dtype=float),
                np.asarray(moment, dtype=float),
            )
            if f_arr.shape != (3,) or m_arr.shape != (3,):
                raise ValueError(
                    f"wrench for {body_name!r} must be a pair of 3-vectors"
                )
            if not (np.all(np.isfinite(f_arr)) and np.all(np.isfinite(m_arr))):
                raise ValueError(f"wrench for {body_name!r} must be finite")
            cleaned[body_name] = (f_arr.copy(), m_arr.copy())
            if body_name in self._prescribed:
                pt = local_points.get(body_name, np.zeros(3))
                _update_prescribed_force(self._prescribed[body_name], f_arr, m_arr, pt)
        self._applied_wrenches = cleaned


@dataclass(frozen=True)
class MskGripHoldResult:
    """Result of comparing the native OpenSim MSK grip hold against the bushing path."""

    time_s: np.ndarray
    reports: tuple[GripStepReport, ...]
    binding: HandClubBinding
    topology_model: TopologyContactModel
    hand_states: Mapping[str, Sequence[BushingState]]
    club_states: Mapping[str, Sequence[BushingState]]
    bushing_club_forces_n: np.ndarray
    bushing_club_moments_nm: np.ndarray
    contact_club_forces_n: np.ndarray
    contact_club_moments_nm: np.ndarray
    force_error: QuantityError
    moment_error: QuantityError


def _club_side_bushing_state(
    rot_c: np.ndarray,
    pos_c: np.ndarray,
    vel_c: np.ndarray,
    omega_c: np.ndarray,
    grip_local: np.ndarray,
) -> BushingState:
    """World :class:`BushingState` of one club grip frame given club origin state."""
    r_w = rot_c @ grip_local[:3, 3]
    return BushingState(
        rot_c @ grip_local[:3, :3], pos_c + r_w, vel_c + np.cross(omega_c, r_w), omega_c
    )


def _make_hold_wrench_fns(
    topo_model: TopologyContactModel,
    bushing_params: Any,
    hand_fixed: Mapping[str, BushingState],
    grip_local: Mapping[str, np.ndarray],
) -> tuple[Any, Any]:
    """Build bushing and shared-law contact wrench callbacks for :func:`simulate_msk_grip_hold`."""

    def _bushing_fn(
        rot: np.ndarray, pos: np.ndarray, vel: np.ndarray, omg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        f_tot, m_tot = np.zeros(3), np.zeros(3)
        for side in ("L", "R"):
            c_state = _club_side_bushing_state(rot, pos, vel, omg, grip_local[side])
            bw = bushing_wrench(bushing_params, hand_fixed[side], c_state)
            f_tot = f_tot + bw.force_n
            m_tot = (
                m_tot + bw.moment_nm + np.cross(c_state.position_m - pos, bw.force_n)
            )
        return f_tot, m_tot

    def _contact_fn(
        rot: np.ndarray, pos: np.ndarray, vel: np.ndarray, omg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        f_tot, m_tot = np.zeros(3), np.zeros(3)
        for side in ("L", "R"):
            c_state = _club_side_bushing_state(rot, pos, vel, omg, grip_local[side])
            sw = topology_side_wrench(topo_model, side, hand_fixed[side], c_state)
            f_tot = f_tot + sw.force_n
            m_tot = (
                m_tot + sw.moment_nm + np.cross(c_state.position_m - pos, sw.force_n)
            )
        return f_tot, m_tot

    return _bushing_fn, _contact_fn


def _make_hold_rhs(wrench_fn: Any, dyn: ClubDynamics, rot0: np.ndarray) -> Any:
    """Build ODE RHS ``dy/dt`` for 6-DOF free ``Club`` under gravity and ``wrench_fn``."""
    mass, com_local = float(dyn.mass_kg), np.asarray(dyn.com_m, dtype=float)
    inertia_local, grav = (
        np.asarray(dyn.inertia_com_kg_m2, dtype=float),
        np.asarray(GRAVITY_M_S2, dtype=float),
    )

    def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
        pos, vel, omg = y[:3], y[6:9], y[9:12]
        rot_rel = Rotation.from_rotvec(y[3:6])
        rot = rot_rel.as_matrix() @ rot0
        f_orig, m_orig = wrench_fn(rot, pos, vel, omg)
        r_com = rot @ com_local
        m_com = m_orig - np.cross(r_com, f_orig)
        i_world = rot @ inertia_local @ rot.T
        alpha = np.linalg.solve(i_world, m_com - np.cross(omg, i_world @ omg))
        acc_orig = (
            (f_orig + mass * grav) / mass
            - np.cross(alpha, r_com)
            - np.cross(omg, np.cross(omg, r_com))
        )
        return np.concatenate([vel, omg, acc_orig, alpha])

    return _rhs


def _as_bushing_state(state: object) -> BushingState:
    """Return ``state`` as a ``BushingState``; raise ``TypeError`` otherwise."""
    if not isinstance(state, BushingState):
        raise TypeError(
            f"OpenSim MSK hold expects BushingState samples, got {type(state).__name__}"
        )
    return state


def _held_still(state: object) -> BushingState:
    """The sampled hand pose with zero velocity: the hold fixes both hands."""
    raw = _as_bushing_state(state)
    return BushingState(raw.rotation, raw.position_m, np.zeros(3), np.zeros(3))


def _integrate_hold(rhs: Any, y0: np.ndarray, t_eval: np.ndarray) -> Any:
    """Integrate a club hold RHS from rest over [0, t_eval[-1]] (stiff Radau)."""
    return solve_ivp(
        rhs,
        (0.0, float(t_eval[-1])),
        y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        max_step=1e-3,
    )


def simulate_msk_grip_hold(
    *,
    model_id: str = "golf_humanoid",
    club: str = "driver",
    grip_style: GripStyle | str = GripStyle.OVERLAP,
    trigger_finger: bool = False,
    squeeze_per_hand_n: float = 1100.0,
) -> MskGripHoldResult:
    """Compare the native OpenSim MSK grip adapter against the ``msk_club`` bushing hold."""
    adapter = OpenSimMskGripAdapter.from_model_id(
        model_id,
        club=club,
        grip_model="bushing",
        grip_style=grip_style,
        trigger_finger=trigger_finger,
    )
    msk = adapter.club
    topo, iface = msk.topology, msk.interface
    topo_model = build_topology_model(
        iface, topo, squeeze_per_hand_n=squeeze_per_hand_n
    )
    binding = adapter.resolve_binding(topo)
    init_states = adapter.sample_states(binding)
    hand_fixed = {
        "L": _held_still(init_states[binding.lead_hand_body]),
        "R": _held_still(init_states[binding.trail_hand_body]),
    }
    club_raw = _as_bushing_state(init_states[binding.club_body])
    left_iface, right_iface = iface.left, iface.right
    grip_local = {"L": left_iface.matrix(), "R": right_iface.matrix()}
    bushing_fn, contact_fn = _make_hold_wrench_fns(
        topo_model, iface.bushing, hand_fixed, grip_local
    )
    rot0 = club_raw.rotation.copy()
    y0 = np.zeros(12)
    y0[:3] = club_raw.position_m
    t_eval = np.linspace(0.08, 0.12, 21)
    sol_b = _integrate_hold(_make_hold_rhs(bushing_fn, msk.dynamics, rot0), y0, t_eval)
    sol_c = _integrate_hold(_make_hold_rhs(contact_fn, msk.dynamics, rot0), y0, t_eval)
    b_forces, b_moments, c_forces, c_moments = [], [], [], []
    reports: list[GripStepReport] = []
    club_seq: dict[str, list[BushingState]] = {"L": [], "R": []}
    for k in range(len(t_eval)):
        yb, yc = sol_b.y[:, k], sol_c.y[:, k]
        rb = Rotation.from_rotvec(yb[3:6]).as_matrix() @ rot0
        rc = Rotation.from_rotvec(yc[3:6]).as_matrix() @ rot0
        fb, mb = bushing_fn(rb, yb[:3], yb[6:9], yb[9:12])
        fc, mc_val = contact_fn(rc, yc[:3], yc[6:9], yc[9:12])
        b_forces.append(fb)
        b_moments.append(mb)
        c_forces.append(fc)
        c_moments.append(mc_val)
        step_map = {
            binding.lead_hand_body: hand_fixed["L"],
            binding.trail_hand_body: hand_fixed["R"],
            binding.club_body: BushingState(rc, yc[:3], yc[6:9], yc[9:12]),
        }
        rep = evaluate_contacts(topo_model, binding, step_map)
        apply_report(adapter, rep)
        reports.append(rep)
        for side in ("L", "R"):
            club_seq[side].append(
                _club_side_bushing_state(
                    rc, yc[:3], yc[6:9], yc[9:12], grip_local[side]
                )
            )
    b_f_arr, b_m_arr = np.asarray(b_forces, float), np.asarray(b_moments, float)
    c_f_arr, c_m_arr = np.asarray(c_forces, float), np.asarray(c_moments, float)
    times = t_eval - t_eval[0]
    hand_seq = {
        "L": [hand_fixed["L"] for _ in times],
        "R": [hand_fixed["R"] for _ in times],
    }
    return MskGripHoldResult(
        time_s=times,
        reports=tuple(reports),
        binding=binding,
        topology_model=topo_model,
        hand_states=hand_seq,
        club_states=club_seq,
        bushing_club_forces_n=b_f_arr,
        bushing_club_moments_nm=b_m_arr,
        contact_club_forces_n=c_f_arr,
        contact_club_moments_nm=c_m_arr,
        force_error=compute_quantity_error(c_f_arr, b_f_arr, name="net_force"),
        moment_error=compute_quantity_error(c_m_arr, b_m_arr, name="net_moment"),
    )
