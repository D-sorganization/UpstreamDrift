"""MuJoCo Shadow Hand and Wonik Allegro grip laboratory adapter (HRG-GS-M6, #13328).

IMPORTANT QUALIFICATION NOTE:
This module builds and drives **a contact fixture, not a qualified golfer**.
It couples a free cylindrical club handle (``club_handle``) with bimanual
articulated hand models (Shadow Dexterous Hand and Wonik Allegro Hand) driven
by mocap palm welds (``lh_mocap``, ``rh_mocap``), reusing:
- :mod:`src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._grip_modelling_xml`
- :mod:`src.shared.python.grip_contact.binding`
- :mod:`src.shared.python.grip_contact.binding_manifest`
- :mod:`src.shared.python.grip_contact.contact_topology`
- :mod:`src.shared.python.grip_contact.pair_contact`
- :mod:`src.shared.python.grip_contact.topology_contact`
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Literal

import defusedxml.ElementTree as DefusedET
import mujoco
import numpy as np
import numpy.linalg as npla

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._grip_modelling_xml import (
    inject_cylinder_object,
    inject_mocap_bodies,
    prepare_scene_xml,
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
from src.shared.python.grip_contact.interface import GripFrame, GripInterface
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.topology_contact import TopologyContactModel
from src.shared.python.motion_matching.grip_contract import GripSpec

__all__ = [
    "FIXTURE_DISCLAIMER",
    "LAB_MANIFEST_PATHS",
    "LAB_MODEL_IDS",
    "LabContactMode",
    "MuJoCoGripLabAdapter",
    "build_grip_lab_mjcf",
    "check_hand_assets_available",
]

FIXTURE_DISCLAIMER: Final[str] = (
    "MuJoCo Shadow/Allegro grip laboratory is a contact fixture, not a qualified golfer."
)
LAB_MODEL_IDS: Final[tuple[str, ...]] = (
    "mujoco_shadow_hand_lab",
    "mujoco_allegro_hand_lab",
)
LabContactMode = Literal["shared", "native"]
LAB_CONTACT_MODES: Final[tuple[LabContactMode, ...]] = ("shared", "native")

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
_MANIFESTS_DIR: Final[Path] = _REPO_ROOT / "src/config/grip_bindings"
_DEFAULT_ASSETS_ROOT: Final[Path] = (
    Path(__file__).resolve().parent / "mujoco_humanoid_golf/hand_assets"
)
_SPEC_PATH: Final[Path] = (
    _REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)

LAB_MANIFEST_PATHS: Final[dict[str, Path]] = {
    model_id: _MANIFESTS_DIR / f"{model_id}.json" for model_id in LAB_MODEL_IDS
}

_HAND_ASSET_DIRS: Final[dict[str, str]] = {
    "mujoco_shadow_hand_lab": "shadow_hand",
    "mujoco_allegro_hand_lab": "wonik_allegro",
}
_REQUIRED_ASSET_FILES: Final[dict[str, tuple[str, ...]]] = {
    "mujoco_shadow_hand_lab": (
        "scene_both.xml",
        "left_hand.xml",
        "right_hand.xml",
    ),
    "mujoco_allegro_hand_lab": (
        "left_hand.xml",
        "right_hand.xml",
    ),
}


def _raise_lab_unavailable(model_id: str, feature: str) -> None:
    raise SchemaFeatureUnavailableError(
        model_id,
        consumer="HRG-GS-M6 (#13328)",
        feature=feature,
        child="grip_lab_adapter",
        issue=13328,
    )


def _canonical_lab_model_id(model_id: str) -> str:
    if model_id in LAB_MODEL_IDS:
        return model_id
    aliases = {
        "shadow_hand": "mujoco_shadow_hand_lab",
        "shadow_hand_lab": "mujoco_shadow_hand_lab",
        "wonik_allegro": "mujoco_allegro_hand_lab",
        "allegro_hand_lab": "mujoco_allegro_hand_lab",
    }
    if model_id in aliases:
        return aliases[model_id]
    raise ValueError(
        f"Unknown grip laboratory model_id {model_id!r}; expected one of {LAB_MODEL_IDS}"
    )


def check_hand_assets_available(
    model_id: str,
    *,
    assets_root: Path | None = None,
) -> Path:
    """Verify local Shadow/Allegro hand assets exist without downloading anything.

    Raises:
        SchemaFeatureUnavailableError: when the asset directory, XMLs, or mesh
            folder is missing.
    """
    canonical_id = _canonical_lab_model_id(model_id)
    root_dir = assets_root if assets_root is not None else _DEFAULT_ASSETS_ROOT
    subdir_name = _HAND_ASSET_DIRS[canonical_id]
    hand_dir = root_dir / subdir_name
    if not hand_dir.is_dir():
        _raise_lab_unavailable(
            canonical_id,
            f"hand assets directory missing at {hand_dir} (never download assets)",
        )
    for req_file in _REQUIRED_ASSET_FILES[canonical_id]:
        req_path = hand_dir / req_file
        if not req_path.is_file():
            _raise_lab_unavailable(
                canonical_id,
                f"hand assets XML file missing at {req_path}",
            )
    mesh_dir = hand_dir / "assets"
    if not mesh_dir.is_dir():
        _raise_lab_unavailable(
            canonical_id,
            f"hand assets mesh directory missing at {mesh_dir}",
        )
    return hand_dir


def _load_lab_manifest(
    model_id: str,
    manifest_override: Mapping[str, Any] | Path | None = None,
) -> dict[str, Any]:
    canonical_id = _canonical_lab_model_id(model_id)
    if manifest_override is None:
        path = LAB_MANIFEST_PATHS[canonical_id]
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


def _rewrite_compiler_meshdir(root: ET.Element, mesh_dir: Path) -> None:
    compilers = list(root.findall("compiler"))
    if not compilers:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
        compilers = [compiler]
    compilers[0].set("meshdir", str(mesh_dir.resolve()))
    for extra in compilers[1:]:
        root.remove(extra)


def _prefix_allegro_tree(hand_root: ET.Element, prefix: str) -> None:
    """Prefix all body, joint, geom, actuator, default-class, and asset names in an Allegro XML."""
    name_tags = {"body", "joint", "geom", "site", "mesh", "material", "position"}
    ref_attrs = ("class", "childclass", "mesh", "material", "joint", "body1", "body2")
    for elem in hand_root.iter():
        if elem.tag == "mesh" and "name" not in elem.attrib and "file" in elem.attrib:
            file_val = elem.attrib["file"]
            elem.set("name", Path(file_val).stem)
        if elem.tag in name_tags and "name" in elem.attrib:
            old_name = elem.attrib["name"]
            elem.set("name", f"{prefix}{old_name}")
        for attr in ref_attrs:
            if attr in elem.attrib:
                old_val = elem.attrib[attr]
                elem.set(attr, f"{prefix}{old_val}")


def _build_bimanual_allegro_xml(hand_dir: Path) -> str:
    """Combine left and right Wonik Allegro hands with mocap palm welds and free handle."""
    left_root = DefusedET.parse(str(hand_dir / "left_hand.xml")).getroot()
    right_root = DefusedET.parse(str(hand_dir / "right_hand.xml")).getroot()
    if left_root is None or right_root is None:
        raise ValueError("Invalid Wonik Allegro hand XML")

    _prefix_allegro_tree(left_root, "lh_")
    _prefix_allegro_tree(right_root, "rh_")

    combined = ET.Element("mujoco", {"model": "bimanual_allegro_grip_lab"})
    compiler = left_root.find("compiler")
    if compiler is not None:
        combined.append(compiler)
    _rewrite_compiler_meshdir(combined, hand_dir / "assets")

    for section_tag in ("default", "asset", "worldbody", "contact", "actuator"):
        sec_el = ET.SubElement(combined, section_tag)
        for src_root in (left_root, right_root):
            src_sec = src_root.find(section_tag)
            if src_sec is not None:
                for child in list(src_sec):
                    sec_el.append(child)

    wb = combined.find("worldbody")
    if wb is not None:
        for body_el in wb.findall("body"):
            b_name = body_el.get("name")
            if b_name in ("lh_palm", "rh_palm"):
                pos_str = "0 -0.15 0.25" if b_name == "lh_palm" else "0 0.15 0.25"
                body_el.set("pos", pos_str)
                body_el.insert(0, ET.Element("freejoint", {"name": f"{b_name}_free"}))

    xml_str = ET.tostring(combined, encoding="unicode")
    xml_str = inject_cylinder_object(xml_str)
    xml_str = inject_mocap_bodies(xml_str, hand_dir / "left_hand.xml", is_both=True)
    xml_str = xml_str.replace('body2="rh_forearm"', 'body2="rh_palm"')
    xml_str = xml_str.replace('body2="lh_forearm"', 'body2="lh_palm"')
    return xml_str


def build_grip_lab_mjcf(
    model_id: str,
    *,
    assets_root: Path | None = None,
) -> str:
    """Build the bimanual MuJoCo grip laboratory XML string for ``model_id``."""
    canonical_id = _canonical_lab_model_id(model_id)
    hand_dir = check_hand_assets_available(canonical_id, assets_root=assets_root)
    if canonical_id == "mujoco_shadow_hand_lab":
        scene_path = hand_dir / "scene_both.xml"
        xml_str = prepare_scene_xml(scene_path, hand_dir, is_both=True)
        root = DefusedET.fromstring(xml_str)
        _rewrite_compiler_meshdir(root, hand_dir / "assets")
        for body_el in root.iter("body"):
            if body_el.get("name") == "object":
                body_el.set("name", "club_handle")
        return ET.tostring(root, encoding="unicode")
    return _build_bimanual_allegro_xml(hand_dir)


def _default_interface() -> GripInterface:
    if _SPEC_PATH.is_file():
        spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
        return GripInterface.from_spec(spec)
    rot = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    left = GripFrame("L", (-0.0762, 0.0, 0.014), rot)
    right = GripFrame("R", (0.0, 0.0, -0.014), rot)
    return GripInterface(left=left, right=right)


def _sample_mj_pose_and_vel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
    return pose, (jacp @ qvel, jacr @ qvel)


def _state_to_pose_and_vel(
    state: BushingState | RigidBodyState,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = state.rotation
    pose[:3, 3] = state.position_m
    vel = (state.velocity_m_s.copy(), state.omega_rad_s.copy())
    return pose, vel


_MODEL_CACHE: dict[tuple[str, str], mujoco.MjModel] = {}
_CAL_POSE_CACHE: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}


def _get_or_compile_lab_model(
    canonical_id: str,
    *,
    assets_root: Path | None = None,
) -> mujoco.MjModel:
    hand_dir = check_hand_assets_available(canonical_id, assets_root=assets_root)
    cache_key = (canonical_id, str(hand_dir.resolve()))
    if cache_key not in _MODEL_CACHE:
        xml_str = build_grip_lab_mjcf(canonical_id, assets_root=assets_root)
        _MODEL_CACHE[cache_key] = mujoco.MjModel.from_xml_string(xml_str)
    return _MODEL_CACHE[cache_key]


def _sample_mj_pose_only(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray:
    b_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if b_id < 0:
        raise ValueError(f"Body {body_name!r} not found in MjModel")
    pose = np.eye(4, dtype=float)
    xmat = data.xmat
    xpos = data.xpos
    pose[:3, :3] = np.asarray(xmat[b_id], dtype=float).reshape(3, 3)
    pose[:3, 3] = np.asarray(xpos[b_id], dtype=float)
    return pose


def _calibration_relative_poses(
    model: mujoco.MjModel,
    lead_root: str,
    trail_root: str,
) -> dict[str, np.ndarray]:
    """Compute calibration pose T_{H_s -> B_i}^0 for every body in MjModel at qpos0."""
    cache_key = (id(model), lead_root, trail_root)
    if cache_key in _CAL_POSE_CACHE:
        return _CAL_POSE_CACHE[cache_key]
    cal_data = mujoco.MjData(model)
    mujoco.mj_forward(model, cal_data)
    l_pose = _sample_mj_pose_only(model, cal_data, lead_root)
    r_pose = _sample_mj_pose_only(model, cal_data, trail_root)
    l_inv = npla.inv(l_pose)
    r_inv = npla.inv(r_pose)

    rel_poses: dict[str, np.ndarray] = {}
    nbody = int(model.nbody)
    for b_id in range(1, nbody):
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)
        if not b_name:
            continue
        b_pose = _sample_mj_pose_only(model, cal_data, b_name)
        if b_name.startswith("lh_"):
            rel_poses[b_name] = l_inv @ b_pose
        elif b_name.startswith("rh_"):
            rel_poses[b_name] = r_inv @ b_pose
    _CAL_POSE_CACHE[cache_key] = rel_poses
    return rel_poses


class MuJoCoGripLabAdapter:
    """MuJoCo Shadow/Allegro grip laboratory adapter: a contact fixture, not a qualified golfer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        manifest: Mapping[str, Any] | Path | None = None,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        contact_mode: LabContactMode = "shared",
        frozen_fingers: bool = False,
    ) -> None:
        if contact_mode not in LAB_CONTACT_MODES:
            raise ValueError(
                f"contact_mode must be one of {LAB_CONTACT_MODES}, got {contact_mode!r}"
            )
        spec = grip_spec if grip_spec is not None else GripSpec()
        if not isinstance(spec, GripSpec):
            raise TypeError(f"grip_spec must be GripSpec, got {type(spec).__name__}")

        resolved_manifest = _load_lab_manifest("mujoco_shadow_hand_lab", manifest)
        self._manifest = resolved_manifest
        self._model_id = str(resolved_manifest["model_id"])
        self._club_body = str(resolved_manifest["club_body"])
        self._lead_hand_body = str(resolved_manifest["lead_hand_body"])
        self._trail_hand_body = str(resolved_manifest["trail_hand_body"])
        self._segment_map: dict[str, str] = dict(
            resolved_manifest.get("segment_map", {})
        )

        self._model = model
        self._data = data if data is not None else mujoco.MjData(model)
        mujoco.mj_forward(self._model, self._data)
        self._qpos_cal = np.asarray(self._data.qpos, dtype=float).copy()

        self._grip_spec = spec
        self._topology = contact_topology(spec)
        self._interface = interface if interface is not None else _default_interface()
        self._contact_mode: LabContactMode = contact_mode
        self._frozen_fingers = bool(frozen_fingers)

        init_club = RigidBodyState(
            rotation=np.eye(3),
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            omega_rad_s=np.zeros(3),
        )
        self._club_state: RigidBodyState = init_club
        self._hand_states: dict[str, BushingState] = dict(
            hand_frame_states(init_club, self._interface)
        )
        self._phalanx_deltas: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._applied_wrenches: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cal_rel_poses = _calibration_relative_poses(
            self._model, self._lead_hand_body, self._trail_hand_body
        )

    @classmethod
    def from_model_id(
        cls,
        model_id: str = "mujoco_shadow_hand_lab",
        *,
        grip_spec: GripSpec | None = None,
        interface: GripInterface | None = None,
        contact_mode: LabContactMode = "shared",
        frozen_fingers: bool = False,
        assets_root: Path | None = None,
    ) -> MuJoCoGripLabAdapter:
        """Build a grip laboratory contact fixture adapter from ``model_id``."""
        canonical_id = _canonical_lab_model_id(model_id)
        mj_model = _get_or_compile_lab_model(canonical_id, assets_root=assets_root)
        manifest = _load_lab_manifest(canonical_id)
        mj_data = mujoco.MjData(mj_model)
        return cls(
            mj_model,
            mj_data,
            manifest=manifest,
            grip_spec=grip_spec,
            interface=interface,
            contact_mode=contact_mode,
            frozen_fingers=frozen_fingers,
        )

    @property
    def fixture_disclaimer(self) -> str:
        """Explicit qualification disclaimer for this laboratory fixture."""
        return FIXTURE_DISCLAIMER

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
        """Bound club handle body name."""
        return self._club_body

    @property
    def lead_hand_body(self) -> str:
        """Bound lead palm body name."""
        return self._lead_hand_body

    @property
    def trail_hand_body(self) -> str:
        """Bound trail palm body name."""
        return self._trail_hand_body

    @property
    def frozen_fingers(self) -> bool:
        """True when finger joints are locked to their calibration pose."""
        return self._frozen_fingers

    @property
    def contact_mode(self) -> LabContactMode:
        """Active contact mode (``'shared'`` or ``'native'``)."""
        return self._contact_mode

    @property
    def applied_wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Most recently applied body wrenches ``{body_id: (force_n, moment_nm)}``."""
        return {k: (f.copy(), m.copy()) for k, (f, m) in self._applied_wrenches.items()}

    def _available_body_names(self) -> tuple[str, ...]:
        names: list[str] = []
        nbody = int(self._model.nbody)
        for b_id in range(1, nbody):
            b_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, b_id)
            if b_name:
                names.append(b_name)
        return tuple(names)

    def capabilities(self) -> GripCapabilities:
        """Return declared :class:`GripCapabilities` for this laboratory fixture."""
        avail = self._available_body_names()
        if self._contact_mode == "native":
            return GripCapabilities(
                attachment_modes=("separated_v2", "contact"),
                supports_physical_pairs=True,
                supports_wrench_application=False,
                native_contact=True,
                available_bodies=avail,
            )
        return GripCapabilities(
            attachment_modes=("separated_v2", "contact", "contact_v1", "bushing"),
            supports_physical_pairs=True,
            supports_wrench_application=True,
            native_contact=False,
            available_bodies=avail,
        )

    def _club_grip_frames(self) -> dict[str, np.ndarray]:
        left_f = self._interface.left
        right_f = self._interface.right
        return {"L": left_f.matrix(), "R": right_f.matrix()}

    def resolve_binding(
        self,
        topology: ContactTopology,
        *,
        attachment_mode: str | None = None,
    ) -> HandClubBinding:
        """Resolve and validate a segment-mapped phalanx :class:`HandClubBinding`."""
        manifest = dict(self._manifest)
        if attachment_mode is not None:
            manifest["attachment_mode"] = attachment_mode
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
        """Set hand and club grip frame states for trajectory or fixture evaluation."""
        for side_key in ("L", "R"):
            if side_key not in hand_states:
                raise ValueError(f"hand_states missing side {side_key!r}")
        self._hand_states = {"L": hand_states["L"], "R": hand_states["R"]}
        if isinstance(club_state, RigidBodyState):
            self._club_state = club_state
        else:
            self._club_state = RigidBodyState(
                rotation=club_state.rotation,
                position_m=club_state.position_m,
                velocity_m_s=club_state.velocity_m_s,
                omega_rad_s=club_state.omega_rad_s,
            )

    def clear_prescribed_grip_states(self) -> None:
        """Reset hand/club states and clear any phalanx articulation offsets."""
        init_club = RigidBodyState(
            rotation=np.eye(3),
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            omega_rad_s=np.zeros(3),
        )
        self._club_state = init_club
        self._hand_states = dict(hand_frame_states(init_club, self._interface))
        self._phalanx_deltas.clear()
        self._data.qpos[:] = self._qpos_cal
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def articulate_site_phalanx(
        self,
        site_id: str,
        *,
        delta_pos_m: np.ndarray | None = None,
        delta_vel_m_s: np.ndarray | None = None,
        joint_delta_rad: float = 0.25,
    ) -> None:
        """Articulate the phalanx body mapped to ``site_id`` in ``MjData`` (no-op if frozen)."""
        if self._frozen_fingers:
            return
        if site_id not in self._segment_map:
            raise KeyError(f"Unknown site_id {site_id!r} in segment_map")
        body_name = self._segment_map[site_id]
        b_id = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if b_id < 0:
            raise ValueError(f"Body {body_name!r} not found in MjModel")

        jnt_adr = int(self._model.body_jntadr[b_id])
        jnt_num = int(self._model.body_jntnum[b_id])
        if jnt_num > 0 and jnt_adr >= 0:
            qpos_adr = int(self._model.jnt_qposadr[jnt_adr])
            self._data.qpos[qpos_adr] += float(joint_delta_rad)
            mujoco.mj_forward(self._model, self._data)

        d_pos = (
            np.asarray(delta_pos_m, dtype=float).copy()
            if delta_pos_m is not None
            else np.zeros(3, dtype=float)
        )
        d_vel = (
            np.asarray(delta_vel_m_s, dtype=float).copy()
            if delta_vel_m_s is not None
            else np.zeros(3, dtype=float)
        )
        self._phalanx_deltas[body_name] = (d_pos, d_vel, np.zeros(3, dtype=float))

    def _phalanx_state_for_body(
        self,
        body_id: str,
        base_hand: BushingState,
        root_pose: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if self._frozen_fingers:
            return _state_to_pose_and_vel(base_hand)

        b_pose = _sample_mj_pose_only(self._model, self._data, body_id)
        rel_cur = npla.inv(root_pose) @ b_pose
        rel_cal = self._cal_rel_poses.get(body_id, rel_cur)
        delta_mj = rel_cur @ npla.inv(rel_cal)

        hand_rb = RigidBodyState(
            rotation=base_hand.rotation,
            position_m=base_hand.position_m,
            velocity_m_s=base_hand.velocity_m_s,
            omega_rad_s=base_hand.omega_rad_s,
        )
        art_state = hand_rb.frame(delta_mj)
        pose, (v_lin, v_ang) = _state_to_pose_and_vel(art_state)
        if body_id in self._phalanx_deltas:
            d_pos, d_vel, d_omega = self._phalanx_deltas[body_id]
            pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ d_pos
            v_lin = v_lin + pose[:3, :3] @ d_vel
            v_ang = v_ang + d_omega
        return pose, (v_lin, v_ang)

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        """Sample 4x4 world poses and spatial velocities for the handle and all bound phalanges."""
        if not isinstance(binding, HandClubBinding):
            raise TypeError(
                f"binding must be HandClubBinding, got {type(binding).__name__}"
            )
        if self._frozen_fingers:
            self._data.qpos[:] = self._qpos_cal
            self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        l_root_pose = _sample_mj_pose_only(
            self._model, self._data, self._lead_hand_body
        )
        r_root_pose = _sample_mj_pose_only(
            self._model, self._data, self._trail_hand_body
        )

        poses: dict[str, np.ndarray] = {}
        velocities: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        c_pose, c_vel = _state_to_pose_and_vel(self._club_state)
        poses[binding.club_body] = c_pose
        velocities[binding.club_body] = c_vel

        for body_id in binding.required_bodies():
            if body_id == binding.club_body:
                continue
            is_lead = body_id.startswith("lh_") or body_id == binding.lead_hand_body
            side_char = "L" if is_lead else "R"
            root_pose = l_root_pose if is_lead else r_root_pose
            b_pose, b_vel = self._phalanx_state_for_body(
                body_id, self._hand_states[side_char], root_pose
            )
            poses[body_id] = b_pose
            velocities[body_id] = b_vel

        return BodyStates(poses, velocities=velocities)

    def synthesise_closure_from_net_wrench(
        self,
        force_n: np.ndarray | None = None,
        torque_nm: np.ndarray | None = None,
    ) -> Any:
        """Build a closure stub for allocated hand-hand reporting in frozen-finger contact mode."""
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
        """Evaluate physical hand-grip and phalanx hand-hand contacts at the sampled states."""
        resolved_binding = (
            binding
            if binding is not None
            else self.resolve_binding(topology_model.topology)
        )
        states = self.sample_states(resolved_binding)
        return evaluate_contacts(
            topology_model,
            resolved_binding,
            states,
            closure=closure,
            prior=prior,
        )

    def apply_wrenches(
        self,
        wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Apply equal-and-opposite physical wrenches to phalanx and handle bodies in ``xfrc_applied``."""
        if not isinstance(wrenches, Mapping):
            raise TypeError("wrenches must be a Mapping")
        if self._contact_mode == "native":
            _raise_lab_unavailable(
                self._model_id,
                "cannot apply shared-law wrenches when native_contact is enabled",
            )

        cleaned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        xfrc = self._data.xfrc_applied
        xfrc[:] = 0.0
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
