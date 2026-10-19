"""Native MuJoCo humanoid golf grip adapter for rigid-hand MJCFs (HRG-GS-M5, #13327).

In the native MuJoCo humanoid golf MJCFs (``golfer.xml``, ``UPPER_BODY_GOLF_SWING_XML``,
``FULL_BODY_GOLF_SWING_XML``, ``ADVANCED_BIOMECHANICAL_GOLF_SWING_XML``, and
``CANONICAL_GOLF_HUMANOID_XML``), the club is parented to one hand body and
linked to the other hand body by an equality ``<weld>``.

This adapter implements :class:`~src.shared.python.grip_contact.binding.ModelGripAdapter`:

- Attaches non-colliding zero-mass visual style proxy geoms from :class:`GripSpec`
  (``overlap``, ``interlock``, ``ten_finger``, ``trigger_finger``) without altering
  body mass, inertia, or welded forward dynamics.
- Extracts the equality weld constraint wrench from ``data.efc_force`` via
  :func:`~src.engines.physics_engines.mujoco.python.grip_efc.weld_wrench_on_club`
  and feeds it to :func:`~src.shared.python.grip_contact.binding.evaluate_contacts`
  so hand-hand outputs are labelled ``"allocated"`` with the prior name.
- Declares ``supports_physical_pairs=False`` with an explicit ``unavailable_reason``
  for welded bodies and raises :class:`WeldedGripBodyError` (subclass of both
  :class:`SchemaFeatureUnavailableError` and :class:`ValueError`) if physical
  wrench application is attempted on welded bodies.
- Behind the explicit ``attachment_redesign`` flag (``"off"`` by default),
  supports ``"free_club_shared"`` (free club with shared-law ``xfrc_applied``)
  and ``"free_club_native"`` (free club with ``contype="0"``/``<exclude>`` and
  explicit ``<pair>`` pad-grip geoms), never applying both native and shared
  contact on the same pair.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final, Literal

import mujoco
import numpy as np
from numpy import linalg as npla

from src.engines.physics_engines.mujoco._golf_swing_advanced_xml import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_canonical_xml import (
    CANONICAL_GOLF_HUMANOID_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
    FULL_BODY_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
    UPPER_BODY_GOLF_SWING_XML,
)
from src.engines.physics_engines.mujoco.python.grip_efc import (
    _eq_body,
    dense_efc_jacobian,
    weld_wrench_on_club,
)
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
from src.shared.python.grip_contact.hand_hand_allocation import AllocationPrior
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    map_site_to_hand_frame,
)
from src.shared.python.model_appearance.grip_pose import GripPose
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    resolve_visual_pose,
)

__all__ = [
    "ADMITTED_HUMANOID_MODELS",
    "ATTACHMENT_REDESIGN_MODES",
    "MANIFEST_PATHS",
    "AttachmentRedesignMode",
    "MuJoCoHumanoidGripAdapter",
    "WeldedGripBodyError",
    "build_humanoid_grip_mjcf",
    "extract_weld_closure_wrench",
    "load_humanoid_mjcf_xml",
]

AttachmentRedesignMode = Literal["off", "free_club_shared", "free_club_native"]
ATTACHMENT_REDESIGN_MODES: Final[tuple[AttachmentRedesignMode, ...]] = (
    "off",
    "free_club_shared",
    "free_club_native",
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
_MANIFESTS_DIR: Final[Path] = _REPO_ROOT / "src" / "config" / "grip_bindings"
_GOLFER_XML_PATH: Final[Path] = (
    _REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "mujoco"
    / "models"
    / "generated"
    / "golfer.xml"
)

ADMITTED_HUMANOID_MODELS: Final[tuple[str, ...]] = (
    "mujoco_golfer",
    "mujoco_upper_body_swing",
    "mujoco_full_body_swing",
    "mujoco_advanced_biomechanical_swing",
    "mujoco_canonical_golf_humanoid",
)

_MODEL_ID_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "golfer": "mujoco_golfer",
        "mujoco/legacy-golfer": "mujoco_golfer",
        "golf_upper_body_swing": "mujoco_upper_body_swing",
        "golf_full_body_swing": "mujoco_full_body_swing",
        "advanced_biomechanical_golf_swing": "mujoco_advanced_biomechanical_swing",
        "canonical_golf_humanoid": "mujoco_canonical_golf_humanoid",
    }
)

MANIFEST_PATHS: Final[Mapping[str, Path]] = MappingProxyType(
    {
        model_id: _MANIFESTS_DIR / f"{model_id}.json"
        for model_id in ADMITTED_HUMANOID_MODELS
    }
)

_CONSUMER: Final[str] = "mujoco.humanoid_grip_adapter"
_CHILD: Final[str] = "HRG-GS-M5"
_ISSUE: Final[int] = 13327


class WeldedGripBodyError(SchemaFeatureUnavailableError, ValueError):
    """Raised when physical wrenches or modes are requested on welded or native-contact bodies."""


def _raise_welded_error(model_id: str, feature: str) -> None:
    raise WeldedGripBodyError(
        model_id,
        consumer=_CONSUMER,
        feature=feature,
        child=_CHILD,
        issue=_ISSUE,
    )


def _canonical_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    resolved = _MODEL_ID_ALIASES.get(model_id, model_id)
    if resolved not in ADMITTED_HUMANOID_MODELS:
        raise ValueError(
            f"Unknown MuJoCo humanoid model_id {model_id!r}; "
            f"choose from {ADMITTED_HUMANOID_MODELS}"
        )
    return resolved


def load_humanoid_mjcf_xml(model_id: str) -> str:
    """Return the raw MJCF XML string for an admitted MuJoCo humanoid golf model."""
    resolved = _canonical_model_id(model_id)
    if resolved == "mujoco_golfer":
        return _GOLFER_XML_PATH.read_text(encoding="utf-8")
    if resolved == "mujoco_upper_body_swing":
        return UPPER_BODY_GOLF_SWING_XML
    if resolved == "mujoco_full_body_swing":
        return FULL_BODY_GOLF_SWING_XML
    if resolved == "mujoco_advanced_biomechanical_swing":
        return ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
    return CANONICAL_GOLF_HUMANOID_XML


def _load_manifest_for_model(
    model_id: str,
    manifest_override: Mapping[str, Any] | Path | None = None,
) -> dict[str, Any]:
    if manifest_override is None:
        resolved = _canonical_model_id(model_id)
        path = MANIFEST_PATHS[resolved]
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


def _format_floats(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.17g}" for v in values)


def _find_body_element(root: ET.Element, body_name: str) -> ET.Element:
    for elem in root.iter("body"):
        if elem.get("name") == body_name:
            return elem
    raise ValueError(f"Body {body_name!r} not found in MJCF XML")


def _attach_visual_style_geoms(
    root: ET.Element,
    lead_body_name: str,
    trail_body_name: str,
    grip_spec: GripSpec,
) -> tuple[str, ...]:
    """Attach zero-mass, non-colliding visual proxy geoms for ``grip_spec`` onto both hands."""
    topo = contact_topology(grip_spec)
    pose = resolve_visual_pose(grip_spec)
    lead_el = _find_body_element(root, lead_body_name)
    trail_el = _find_body_element(root, trail_body_name)

    geom_names: list[str] = []
    for site in topo.sites:
        is_lead = site.hand == "lead"
        side_char = "L" if is_lead else "R"
        target_el = lead_el if is_lead else trail_el
        mapped = map_site_to_hand_frame(topo, site, side_char)
        pos_str = _format_floats(np.asarray(mapped.centre_m, dtype=float))
        radius_str = f"{float(site.radius_m):.17g}"
        rgba = "0.20 0.65 0.90 0.35" if is_lead else "0.95 0.55 0.20 0.35"
        g_name = f"grip_vis_{pose.style.value}_{site.name}"
        geom_el = ET.Element(
            "geom",
            {
                "name": g_name,
                "type": "sphere",
                "size": radius_str,
                "pos": pos_str,
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
                "group": "1",
                "rgba": rgba,
            },
        )
        target_el.append(geom_el)
        geom_names.append(g_name)

    for link_tag in pose.linked_fingers():
        link_name = f"grip_vis_link_{pose.style.value}_{link_tag}"
        link_el = ET.Element(
            "geom",
            {
                "name": link_name,
                "type": "sphere",
                "size": "0.006",
                "pos": "0 0 -0.04",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
                "group": "1",
                "rgba": "0.95 0.85 0.15 0.45",
            },
        )
        trail_el.append(link_el)
        geom_names.append(link_name)

    return tuple(geom_names)


def _detach_club_to_worldbody(
    root: ET.Element,
    club_body: str,
    lead_body: str,
    trail_body: str,
    pos0: np.ndarray,
    quat0: np.ndarray,
) -> ET.Element:
    """Move ``club_body`` to ``<worldbody>`` with a ``<freejoint/>`` and remove hand-club welds."""
    parent_map = {child: parent for parent in root.iter() for child in parent}
    club_el = _find_body_element(root, club_body)

    removed_joints = {
        j.get("name") for j in list(club_el.findall("joint")) if j.get("name")
    }
    for j_el in list(club_el.findall("joint")):
        club_el.remove(j_el)

    for act_parent in root.findall("actuator"):
        for act_el in list(act_parent):
            if act_el.get("joint") in removed_joints:
                act_parent.remove(act_el)

    for eq_parent in root.findall("equality"):
        for weld_el in list(eq_parent.findall("weld")):
            b1 = weld_el.get("body1", "")
            b2 = weld_el.get("body2", "")
            s1 = weld_el.get("site1", "")
            s2 = weld_el.get("site2", "")
            if club_body in (b1, b2) or "grip" in s1 or "grip" in s2:
                eq_parent.remove(weld_el)

    for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
        club_el.attrib.pop(attr, None)
    club_el.set("pos", _format_floats(pos0))
    club_el.set("quat", _format_floats(quat0))

    freejoint_el = ET.Element("freejoint", {"name": f"{club_body}_freejoint"})
    club_el.insert(0, freejoint_el)

    old_parent = parent_map[club_el]
    old_parent.remove(club_el)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF XML is missing <worldbody>")
    worldbody.append(club_el)

    contact_el = root.find("contact")
    if contact_el is None:
        contact_el = ET.SubElement(root, "contact")
    for h_body in (lead_body, trail_body):
        ET.SubElement(
            contact_el,
            "exclude",
            {
                "name": f"exclude_{h_body}_{club_body}",
                "body1": h_body,
                "body2": club_body,
            },
        )
    return club_el


def _attach_native_contact_pairs(
    root: ET.Element,
    club_el: ET.Element,
    lead_body: str,
    trail_body: str,
    topology_model: TopologyContactModel,
) -> tuple[str, ...]:
    """Attach ``contype="0" conaffinity="0"`` pad and grip geoms with explicit ``<pair>`` tags."""
    lead_el = _find_body_element(root, lead_body)
    trail_el = _find_body_element(root, trail_body)
    contact_el = root.find("contact")
    if contact_el is None:
        contact_el = ET.SubElement(root, "contact")

    cyl = topology_model.cylinder
    ax_lo, ax_hi = cyl.axial_range_m
    grip_radius_str = f"{float(cyl.radius_m):.17g}"
    fromto_str = f"{float(ax_lo):.17g} 0 0 {float(ax_hi):.17g} 0 0"
    grip_geom_name = "grip_native_cylinder"
    ET.SubElement(
        club_el,
        "geom",
        {
            "name": grip_geom_name,
            "type": "cylinder",
            "fromto": fromto_str,
            "size": grip_radius_str,
            "contype": "0",
            "conaffinity": "0",
            "mass": "0",
        },
    )

    law = topology_model.law
    mu = float(law.static_friction)
    torsional = mu * 0.008
    topo = topology_model.topology
    pair_names: list[str] = []

    for pair in topo.hand_grip_pairs():
        site = topo.site(pair.site_a)
        is_lead = site.hand == "lead"
        side_char = "L" if is_lead else "R"
        target_el = lead_el if is_lead else trail_el
        mapped = map_site_to_hand_frame(topo, site, side_char)
        pos_str = _format_floats(np.asarray(mapped.centre_m, dtype=float))
        pad_geom_name = f"pad_native_{site.name}"
        ET.SubElement(
            target_el,
            "geom",
            {
                "name": pad_geom_name,
                "type": "sphere",
                "size": f"{float(site.radius_m):.17g}",
                "pos": pos_str,
                "contype": "0",
                "conaffinity": "0",
                "mass": "0",
            },
        )
        pair_el_name = f"pair_native_{pair.name}"
        ET.SubElement(
            contact_el,
            "pair",
            {
                "name": pair_el_name,
                "geom1": pad_geom_name,
                "geom2": grip_geom_name,
                "condim": "6",
                "friction": f"{mu:.17g} {mu:.17g} {torsional:.17g} 1e-5 1e-5",
            },
        )
        pair_names.append(pair_el_name)

    return tuple(pair_names)


def build_humanoid_grip_mjcf(
    model_id_or_xml: str,
    *,
    manifest: Mapping[str, Any] | None = None,
    grip_spec: GripSpec | None = None,
    attachment_redesign: AttachmentRedesignMode = "off",
    topology_model: TopologyContactModel | None = None,
    attach_visual_style: bool = True,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Build MJCF XML with visual style from ``GripSpec`` and optional attachment redesign."""
    if attachment_redesign not in ATTACHMENT_REDESIGN_MODES:
        raise ValueError(
            f"attachment_redesign must be one of {ATTACHMENT_REDESIGN_MODES}, "
            f"got {attachment_redesign!r}"
        )
    spec = grip_spec if grip_spec is not None else GripSpec()
    if not isinstance(spec, GripSpec):
        raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")

    if model_id_or_xml.lstrip().startswith("<"):
        raw_xml = model_id_or_xml
        resolved_manifest = dict(manifest) if manifest is not None else {}
    else:
        canonical_id = _canonical_model_id(model_id_or_xml)
        raw_xml = load_humanoid_mjcf_xml(canonical_id)
        resolved_manifest = _load_manifest_for_model(canonical_id, manifest)

    club_body = str(resolved_manifest.get("club_body", "club_shaft"))
    lead_body = str(resolved_manifest.get("lead_hand_body", "hand_left"))
    trail_body = str(resolved_manifest.get("trail_hand_body", "hand_right"))

    root = ET.fromstring(raw_xml)
    visual_geoms: tuple[str, ...] = ()
    if attach_visual_style:
        visual_geoms = _attach_visual_style_geoms(root, lead_body, trail_body, spec)

    native_pairs: tuple[str, ...] = ()
    if attachment_redesign != "off":
        base_model = mujoco.MjModel.from_xml_string(raw_xml)
        base_data = mujoco.MjData(base_model)
        mujoco.mj_forward(base_model, base_data)
        cid = int(mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, club_body))
        pos0 = np.asarray(base_data.xpos[cid], dtype=float).copy()
        quat0 = np.asarray(base_data.xquat[cid], dtype=float).copy()
        club_el = _detach_club_to_worldbody(
            root, club_body, lead_body, trail_body, pos0, quat0
        )
        if attachment_redesign == "free_club_native":
            if topology_model is None:
                raise ValueError(
                    "topology_model is required when attachment_redesign='free_club_native'"
                )
            native_pairs = _attach_native_contact_pairs(
                root, club_el, lead_body, trail_body, topology_model
            )

    xml_out = ET.tostring(root, encoding="unicode")
    return xml_out, visual_geoms, native_pairs


def _find_club_weld_eq_id(model: mujoco.MjModel, club_body_id: int) -> int | None:
    """Find the equality index of the ``<weld>`` attached to ``club_body_id``."""
    weld_type = int(mujoco.mjtEq.mjEQ_WELD)
    for eq_id in range(int(model.neq)):
        if int(model.eq_type[eq_id]) != weld_type:
            continue
        b1, _ = _eq_body(model, eq_id, 1)
        b2, _ = _eq_body(model, eq_id, 2)
        if club_body_id in (b1, b2):
            return eq_id
    return None


def extract_weld_closure_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eq_id: int,
    club_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract world-frame 6D wrench ``(F, tau)`` and anchor point on ``club_body_id``.

    Uses :func:`weld_wrench_on_club` when the 6D relative weld Jacobian has full
    rank (``tol=1e-3``), and falls back to truncated SVD least-squares
    (``rcond=1e-3``) when the kinematic chain has locked out-of-plane DOFs.
    """
    b1, _ = _eq_body(model, eq_id, 1)
    b2, site2 = _eq_body(model, eq_id, 2)
    if club_body_id not in (b1, b2):
        raise ValueError(
            f"club body {club_body_id} is not an object of equality {eq_id}"
        )
    site_xpos = data.site_xpos
    xpos = data.xpos
    point = np.array(site_xpos[site2] if site2 is not None else xpos[b2], dtype=float)
    j1 = np.zeros((6, int(model.nv)))
    j2 = np.zeros((6, int(model.nv)))
    mujoco.mj_jac(model, data, j1[:3], j1[3:], point, b1)
    mujoco.mj_jac(model, data, j2[:3], j2[3:], point, b2)
    jdiff = j2 - j1
    if int(npla.matrix_rank(jdiff, tol=1e-3)) == 6:
        return weld_wrench_on_club(model, data, eq_id, club_body_id)

    eq_cnstr = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    rows = np.flatnonzero((data.efc_type == eq_cnstr) & (data.efc_id == eq_id))
    if rows.size == 0:
        raise ValueError(f"equality {eq_id} is inactive (no constraint rows)")
    qfrc = dense_efc_jacobian(model, data)[rows].T @ data.efc_force[rows]
    wrench_on_obj2, *_ = npla.lstsq(jdiff.T, qfrc, rcond=1e-3)
    wrench = wrench_on_obj2 if club_body_id == b2 else -wrench_on_obj2
    return np.asarray(wrench, dtype=float), point


def _sample_body_pose_and_vel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Sample 4x4 world pose and ``(v_world, omega_world)`` of ``body_name``."""
    b_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if b_id < 0:
        raise ValueError(f"Body {body_name!r} not found in MjModel")
    pose = np.eye(4, dtype=float)
    xmat = data.xmat
    xpos = data.xpos
    pose[:3, :3] = np.asarray(xmat[b_id], dtype=float).reshape(3, 3)
    pose[:3, 3] = np.asarray(xpos[b_id], dtype=float)

    nv = int(model.nv)
    jacp = np.zeros((3, nv), dtype=float)
    jacr = np.zeros((3, nv), dtype=float)
    mujoco.mj_jacBody(model, data, jacp, jacr, b_id)
    qvel = np.asarray(data.qvel, dtype=float)
    v_world = jacp @ qvel
    omega_world = jacr @ qvel
    return pose, (v_world, omega_world)


def _calibration_grip_frames(
    model: mujoco.MjModel,
    club_body: str,
    lead_body: str,
    trail_body: str,
) -> dict[str, np.ndarray]:
    """Compute club-local 4x4 grip frames ``{'L': T_L, 'R': T_R}`` at ``qpos0``."""
    cal_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cal_data)
    c_pose, _ = _sample_body_pose_and_vel(model, cal_data, club_body)
    l_pose, _ = _sample_body_pose_and_vel(model, cal_data, lead_body)
    r_pose, _ = _sample_body_pose_and_vel(model, cal_data, trail_body)
    c_inv = npla.inv(c_pose)
    return {
        "L": c_inv @ l_pose,
        "R": c_inv @ r_pose,
    }


def _state_to_pose_and_vel(
    state: BushingState | RigidBodyState,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = state.rotation
    pose[:3, 3] = state.position_m
    vel = (state.velocity_m_s.copy(), state.omega_rad_s.copy())
    return pose, vel


class MuJoCoHumanoidGripAdapter:
    """Native MuJoCo humanoid golf :class:`ModelGripAdapter` (rigid hands)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        manifest: Mapping[str, Any] | Path | None = None,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        attachment_redesign: AttachmentRedesignMode = "off",
        visual_style_geoms: tuple[str, ...] = (),
        native_pair_names: tuple[str, ...] = (),
    ) -> None:
        if attachment_redesign not in ATTACHMENT_REDESIGN_MODES:
            raise ValueError(
                f"attachment_redesign must be one of {ATTACHMENT_REDESIGN_MODES}, "
                f"got {attachment_redesign!r}"
            )
        spec = grip_spec if grip_spec is not None else GripSpec()
        if not isinstance(spec, GripSpec):
            raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")

        resolved_manifest = _load_manifest_for_model("mujoco_golfer", manifest)
        self._manifest = resolved_manifest
        self._model_id = str(resolved_manifest["model_id"])
        self._club_body = str(resolved_manifest["club_body"])
        self._lead_hand_body = str(resolved_manifest["lead_hand_body"])
        self._trail_hand_body = str(resolved_manifest["trail_hand_body"])

        self._model = model
        self._data = data if data is not None else mujoco.MjData(model)
        mujoco.mj_forward(self._model, self._data)

        self._grip_spec = spec
        self._visual_pose = resolve_visual_pose(spec)
        self._topology = contact_topology(spec)
        self._interface = interface
        self._attachment_redesign: AttachmentRedesignMode = attachment_redesign
        self._visual_style_geoms = tuple(visual_style_geoms)
        self._native_pair_names = tuple(native_pair_names)
        self._prescribed_states: BodyStates | None = None
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._cal_grip_frames = _calibration_grip_frames(
            self._model,
            self._club_body,
            self._lead_hand_body,
            self._trail_hand_body,
        )

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "mujoco_golfer",
        *,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        topology_model: TopologyContactModel | None = None,
        attachment_redesign: AttachmentRedesignMode = "off",
        attach_visual_style: bool = True,
    ) -> MuJoCoHumanoidGripAdapter:
        """Construct an adapter for one of the 5 admitted humanoid golf MJCFs."""
        canonical_id = _canonical_model_id(model_id)
        manifest = _load_manifest_for_model(canonical_id)
        xml_str, vis_geoms, native_pairs = build_humanoid_grip_mjcf(
            canonical_id,
            manifest=manifest,
            grip_spec=grip_spec,
            attachment_redesign=attachment_redesign,
            topology_model=topology_model,
            attach_visual_style=attach_visual_style,
        )
        mj_model = mujoco.MjModel.from_xml_string(xml_str)
        mj_data = mujoco.MjData(mj_model)
        return cls(
            mj_model,
            mj_data,
            manifest=manifest,
            grip_spec=grip_spec,
            interface=interface,
            attachment_redesign=attachment_redesign,
            visual_style_geoms=vis_geoms,
            native_pair_names=native_pairs,
        )

    @property
    def model(self) -> mujoco.MjModel:
        """Underlying ``mujoco.MjModel``."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """Underlying ``mujoco.MjData``."""
        return self._data

    @property
    def model_id(self) -> str:
        """Manifest ``model_id``."""
        return self._model_id

    @property
    def club_body(self) -> str:
        """Bound club body name."""
        return self._club_body

    @property
    def lead_hand_body(self) -> str:
        """Bound lead hand body name."""
        return self._lead_hand_body

    @property
    def trail_hand_body(self) -> str:
        """Bound trail hand body name."""
        return self._trail_hand_body

    @property
    def grip_spec(self) -> GripSpec:
        """Active :class:`GripSpec`."""
        return self._grip_spec

    @property
    def visual_pose(self) -> GripPose:
        """Resolved visual :class:`GripPose` for :attr:`grip_spec`."""
        return self._visual_pose

    @property
    def topology(self) -> ContactTopology:
        """Active :class:`ContactTopology` for :attr:`grip_spec`."""
        return self._topology

    @property
    def attachment_redesign(self) -> AttachmentRedesignMode:
        """Active attachment redesign mode (``'off'``, ``'free_club_shared'``, or ``'free_club_native'``)."""
        return self._attachment_redesign

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Most recently applied external body wrenches ``{body_id: (force_n, moment_nm)}``."""
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    def visual_style_geom_names(self) -> tuple[str, ...]:
        """Names of non-colliding visual proxy geoms attached for :attr:`grip_spec`."""
        return self._visual_style_geoms

    def registered_grip_collision_count(self) -> int:
        """Number of native MuJoCo ``<pair>`` collision entries registered for grip pads."""
        return len(self._native_pair_names)

    def _club_grip_frames(self) -> dict[str, np.ndarray]:
        if self._interface is not None:
            left_f = self._interface.left
            right_f = self._interface.right
            return {"L": left_f.matrix(), "R": right_f.matrix()}
        return dict(self._cal_grip_frames)

    def capabilities(self) -> GripCapabilities:
        """Return declared :class:`GripCapabilities` for the current attachment mode."""
        avail = (self._club_body, self._lead_hand_body, self._trail_hand_body)
        if self._attachment_redesign == "free_club_shared":
            return GripCapabilities(
                attachment_modes=("contact", "contact_v1", "bushing"),
                supports_physical_pairs=True,
                supports_wrench_application=True,
                native_contact=False,
                available_bodies=avail,
            )
        if self._attachment_redesign == "free_club_native":
            return GripCapabilities(
                attachment_modes=("contact",),
                supports_physical_pairs=True,
                supports_wrench_application=False,
                native_contact=True,
                available_bodies=avail,
            )
        welded_reason = (
            f"club body {self._club_body!r} is welded/constrained to hands "
            f"({self._lead_hand_body!r}, {self._trail_hand_body!r}) in MJCF "
            f"{self._model_id!r}; pass attachment_redesign='free_club_shared' "
            "or 'free_club_native' to enable physical hand-club contact"
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
        self,
        topology: ContactTopology,
        *,
        attachment_mode: str | None = None,
    ) -> HandClubBinding:
        """Resolve and validate a :class:`HandClubBinding` against :meth:`capabilities`."""
        manifest = dict(self._manifest)
        if attachment_mode is not None:
            manifest["attachment_mode"] = attachment_mode
        elif self._attachment_redesign != "off":
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
        """Clear prescribed states and resume sampling from ``MjData``."""
        self._prescribed_states = None

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and spatial velocities for bound bodies."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        if self._prescribed_states is not None:
            return self._prescribed_states

        mujoco.mj_forward(self._model, self._data)
        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body_id in binding.required_bodies():
            pose, vel = _sample_body_pose_and_vel(self._model, self._data, body_id)
            poses[body_id] = pose
            velocities[body_id] = vel
        return BodyStates(poses, velocities=velocities)

    def extract_weld_closure(self) -> Any:
        """Extract the equality weld constraint wrench from ``data.efc_force``."""
        cid = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._club_body)
        )
        eq_id = _find_club_weld_eq_id(self._model, cid)
        if eq_id is None:
            return SimpleNamespace(
                available=False,
                reason=f"no hand-club weld equality found for {self._club_body!r}",
                wrenches=[],
            )
        try:
            wrench, point = extract_weld_closure_wrench(
                self._model, self._data, eq_id, cid
            )
        except ValueError as exc:
            return SimpleNamespace(available=False, reason=str(exc), wrenches=[])

        w_item = SimpleNamespace(
            force_n=wrench[:3].copy(),
            torque_nm=wrench[3:].copy(),
            point_m=point.copy(),
        )
        return SimpleNamespace(available=True, reason="", wrenches=[w_item])

    def synthesise_closure_from_net_wrench(
        self,
        force_n: np.ndarray | None = None,
        torque_nm: np.ndarray | None = None,
    ) -> Any:
        """Build a closure stub for allocated hand-hand reporting in free-club mode."""
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
        topology_model: TopologyContactModel,
        binding: HandClubBinding | None = None,
        *,
        closure: Any | None = None,
        prior: AllocationPrior | None = None,
    ) -> GripStepReport:
        """Evaluate grip contacts at the current sampled states, extracting weld ``efc_force`` in weld mode."""
        resolved_binding = (
            binding
            if binding is not None
            else self.resolve_binding(topology_model.topology)
        )
        states = self.sample_states(resolved_binding)
        resolved_closure = closure
        if resolved_closure is None and self._attachment_redesign == "off":
            resolved_closure = self.extract_weld_closure()
        return evaluate_contacts(
            topology_model,
            resolved_binding,
            states,
            closure=resolved_closure,
            prior=prior,
        )

    def apply_wrenches(
        self,
        wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Apply physical body wrenches to ``data.xfrc_applied`` in ``free_club_shared`` mode.

        Raises:
            WeldedGripBodyError: when in welded mode (``attachment_redesign='off'``)
                or native contact mode (``attachment_redesign='free_club_native'``).
        """
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        if self._attachment_redesign == "off":
            _raise_welded_error(
                self._model_id,
                f"cannot apply physical wrenches to welded club/hand bodies in {self._model_id!r}",
            )
        if self._attachment_redesign == "free_club_native":
            _raise_welded_error(
                self._model_id,
                "cannot apply shared-law wrenches when native_contact is enabled",
            )

        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        xfrc = self._data.xfrc_applied
        xpos = self._data.xpos
        xipos = self._data.xipos

        for body_id, pair in wrenches.items():
            f_arr = np.asarray(pair[0], dtype=float)
            m_arr = np.asarray(pair[1], dtype=float)
            if f_arr.shape != (3,) or not bool(np.isfinite(f_arr).all()):
                raise ValueError(f"force for {body_id!r} must be a finite 3-vector")
            if m_arr.shape != (3,) or not bool(np.isfinite(m_arr).all()):
                raise ValueError(f"moment for {body_id!r} must be a finite 3-vector")
            b_idx = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            )
            if b_idx < 0:
                raise ValueError(f"Body {body_id!r} not found in MjModel")
            arm = np.asarray(xpos[b_idx], dtype=float) - np.asarray(
                xipos[b_idx], dtype=float
            )
            tau_com = m_arr + np.cross(arm, f_arr)
            xfrc[b_idx, :3] = f_arr
            xfrc[b_idx, 3:] = tau_com
            cleaned[body_id] = (f_arr.copy(), m_arr.copy())

        self._applied_wrenches = cleaned
