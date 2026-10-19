"""Compose bilateral MyoHub myoHand + free golf-club MJCF scenes (HRG-GS-M8, #13330).

Pure-function generator patterned on :mod:`golfer_scene` that composes a pinned
two-``myoHand`` scene (right hand ``rh_*``, sagittally mirrored left hand ``lh_*``,
and one 6-DOF free ``golf_club`` body for driver and 7-iron). References the
pinned ``shared/models/myosuite/myo_sim`` assets by relative path and never
copies meshes or textures.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as StdET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only; parse via DefusedET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import defusedxml.ElementTree as DefusedET
import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.grip_contact.binding import SchemaFeatureUnavailableError
from src.shared.python.model_appearance.grip_pose import DEFAULT_GRIP_POSE, Hand
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec

__all__ = [
    "HAND_SCENE_DIR",
    "LeftHandMirrorReport",
    "MYOHAND_CANONICAL_BODIES",
    "MYOHAND_CANONICAL_JOINTS",
    "MYO_SIM_PIN_SHA",
    "MyoHandClubKind",
    "GeneratedMyoHandBilateralScenePaths",
    "check_myohand_assets_available",
    "generate_myohand_bilateral_scene",
    "verify_left_hand_sagittal_reflection",
]

_THIS_FILE: Final[Path] = Path(__file__).resolve()
_REPO_ROOT: Final[Path] = _THIS_FILE.parents[5]

MYO_SIM_PIN_SHA: Final[str] = "33f3ded946f55adbdcf963c99999587aadaf975f"
MYO_SIM_REL_ROOT: Final[Path] = Path("shared/models/myosuite/myo_sim")
HAND_SCENE_REL_DIR: Final[Path] = Path("shared/models/myosuite/golf/hand")
HAND_SCENE_DIR: Final[Path] = _REPO_ROOT / HAND_SCENE_REL_DIR

#: World sagittal reflection S_W = diag(-1, 1, 1) across x = 0.
SAGITTAL_WORLD_REFLECTION: Final[np.ndarray] = np.diag([-1.0, 1.0, 1.0])
#: Body-local reflection S_B = diag(1, 1, -1) across local z_B.
BODY_LOCAL_REFLECTION: Final[np.ndarray] = np.diag([1.0, 1.0, -1.0])

#: Root thorax x-offset (m) placing rh_lunate at x = -0.0381 m and lh_lunate at x = +0.0381 m
#: (separated by DEFAULT_GRIP_POSE.hand_spacing_m = 0.0762 m across x = 0).
THORAX_SAGITTAL_X_M: Final[float] = 0.164223

MYOHAND_CANONICAL_BODIES: Final[tuple[str, ...]] = (
    "thorax",
    "clavicle",
    "clavphant",
    "scapula",
    "scapphant",
    "humphant",
    "humphant1",
    "humerus",
    "ulna",
    "radius",
    "lunate",
    "scaphoid",
    "pisiform",
    "triquetrum",
    "capitate",
    "trapezium",
    "firstmc",
    "proximal_thumb",
    "distal_thumb",
    "trapezoid",
    "hamate",
    "secondmc",
    "proxph2",
    "midph2",
    "distph2",
    "thirdmc",
    "proxph3",
    "midph3",
    "distph3",
    "fourthmc",
    "proxph4",
    "midph4",
    "distph4",
    "fifthmc",
    "proxph5",
    "midph5",
    "distph5",
)

MYOHAND_CANONICAL_JOINTS: Final[tuple[str, ...]] = (
    "pro_sup",
    "deviation",
    "flexion",
    "cmc_abduction",
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "mcp2_flexion",
    "mcp2_abduction",
    "pm2_flexion",
    "md2_flexion",
    "mcp3_flexion",
    "mcp3_abduction",
    "pm3_flexion",
    "md3_flexion",
    "mcp4_flexion",
    "mcp4_abduction",
    "pm4_flexion",
    "md4_flexion",
    "mcp5_flexion",
    "mcp5_abduction",
    "pm5_flexion",
    "md5_flexion",
)


class MyoHandClubKind(str, Enum):
    """Supported club specs for the bilateral myoHand fixture."""

    DRIVER = "driver"
    IRON = "iron"


@dataclass(frozen=True)
class GeneratedMyoHandBilateralScenePaths:
    """Paths written by :func:`generate_myohand_bilateral_scene`."""

    driver: Path
    iron: Path
    receipt: Path
    right_chain: Path
    left_chain: Path


@dataclass(frozen=True)
class LeftHandMirrorReport:
    """Quantitative sagittal-reflection verification report for the left myoHand."""

    supported: bool
    body_count: int
    joint_count: int
    max_xpos_error_m: float
    max_xipos_error_m: float
    max_rotation_error: float
    max_mass_error_kg: float
    max_world_inertia_error_kg_m2: float
    max_articulated_xpos_error_m: float
    max_articulated_xipos_error_m: float
    max_normal_reflection_error: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_floats(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()], dtype=float)


def _fmt_floats(arr: np.ndarray) -> str:
    return " ".join(f"{float(x):.12g}" for x in arr)


def _elem_rotation(elem: Any) -> np.ndarray:
    attrib = elem.attrib
    if "euler" in attrib:
        eul = _parse_floats(attrib["euler"])
        return np.asarray(
            Rotation.from_euler("XYZ", eul, degrees=False).as_matrix(),
            dtype=float,
        )
    if "quat" in attrib:
        q = _parse_floats(attrib["quat"])
        q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=float)
        return np.asarray(Rotation.from_quat(q_xyzw).as_matrix(), dtype=float)
    return np.eye(3, dtype=float)


def _set_elem_rotation(elem: Any, rot: np.ndarray) -> None:
    attrib = elem.attrib
    attrib.pop("euler", None)
    q_xyzw = Rotation.from_matrix(rot).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    if q_wxyz[0] < 0.0:
        q_wxyz = -q_wxyz
    elem.set("quat", _fmt_floats(q_wxyz))


def check_myohand_assets_available(repo_root: Path | None = None) -> Path:
    """Verify that the pinned MyoHub myoHand XML assets exist on disk.

    Raises:
        SchemaFeatureUnavailableError: if any required myo_sim hand XML is absent.
    """
    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    myo_sim_dir = root / MYO_SIM_REL_ROOT
    required_rel = (
        Path("hand/myohand.xml"),
        Path("hand/assets/myohand_assets.xml"),
        Path("hand/assets/myohand_body.xml"),
        Path("scene/myosuite_scene.xml"),
    )
    for rel in required_rel:
        candidate = myo_sim_dir / rel
        if not candidate.is_file():
            raise SchemaFeatureUnavailableError(
                "myosuite_myohand_bilateral",
                consumer="HRG-GS-M8 (#13330)",
                feature=(
                    f"Required MyoHub asset {candidate} is missing; "
                    "run scripts/setup_myosuite_models.sh to populate myo_sim."
                ),
                child="myohand_bilateral_scene",
                issue=13330,
            )
    return myo_sim_dir


def _ensure_explicit_inertials(root_body: Any) -> None:
    """Add explicit phantom inertial to bodies lacking <inertial> (thorax, clavphant)."""
    for body in root_body.iter("body"):
        if body.find("inertial") is None:
            StdET.SubElement(
                body,
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "0.0001",
                    "fullinertia": "1e-05 1e-05 1e-05 0 0 0",
                },
            )


def _prefix_right_hand_tree(body: Any, *, is_root: bool = False) -> None:
    """Prefix right-hand body, joint, geom, site, and camera names with 'rh_'."""
    attrib = body.attrib
    attrib["name"] = "rh_" + attrib["name"]
    if is_root and "pos" in attrib:
        pos = _parse_floats(attrib["pos"])
        pos[0] = THORAX_SAGITTAL_X_M
        body.set("pos", _fmt_floats(pos))
    rot = _elem_rotation(body)
    _set_elem_rotation(body, rot)
    for tag in ("joint", "geom", "site", "camera"):
        for child in body.findall(tag):
            c_attrib = child.attrib
            if "name" in c_attrib:
                child.set("name", "rh_" + c_attrib["name"])
            if tag in ("geom", "site") and ("euler" in c_attrib or "quat" in c_attrib):
                _set_elem_rotation(child, _elem_rotation(child))
    for sub in body.findall("body"):
        _prefix_right_hand_tree(sub, is_root=False)


def _mirror_left_hand_inertial(body: Any) -> None:
    inert = body.find("inertial")
    if inert is None:
        return
    i_attrib = inert.attrib
    if "pos" in i_attrib:
        inert.set(
            "pos", _fmt_floats(BODY_LOCAL_REFLECTION @ _parse_floats(i_attrib["pos"]))
        )
    if "fullinertia" in i_attrib:
        fi = _parse_floats(i_attrib["fullinertia"])
        fi[4] = -fi[4]
        fi[5] = -fi[5]
        inert.set("fullinertia", _fmt_floats(fi))


def _mirror_left_hand_children(body: Any) -> None:
    for cam in body.findall("camera"):
        if "name" in cam.attrib:
            cam.set("name", "lh_" + cam.attrib["name"])
    for jnt in body.findall("joint"):
        j_attrib = jnt.attrib
        jnt.set("name", "lh_" + j_attrib["name"])
        if "pos" in j_attrib:
            jnt.set(
                "pos",
                _fmt_floats(BODY_LOCAL_REFLECTION @ _parse_floats(j_attrib["pos"])),
            )
        if "axis" in j_attrib:
            jnt.set(
                "axis",
                _fmt_floats(-BODY_LOCAL_REFLECTION @ _parse_floats(j_attrib["axis"])),
            )
    for geom in body.findall("geom"):
        g_attrib = geom.attrib
        if "name" in g_attrib:
            geom.set("name", "lh_" + g_attrib["name"])
        if "mesh" in g_attrib:
            geom.set("mesh", "lh_" + g_attrib["mesh"])
        if "pos" in g_attrib:
            geom.set(
                "pos",
                _fmt_floats(BODY_LOCAL_REFLECTION @ _parse_floats(g_attrib["pos"])),
            )
        if "fromto" in g_attrib:
            ft = _parse_floats(g_attrib["fromto"])
            ft[2] = -ft[2]
            ft[5] = -ft[5]
            geom.set("fromto", _fmt_floats(ft))
        if "euler" in g_attrib or "quat" in g_attrib:
            _set_elem_rotation(
                geom,
                BODY_LOCAL_REFLECTION @ _elem_rotation(geom) @ BODY_LOCAL_REFLECTION,
            )
    for site in body.findall("site"):
        s_attrib = site.attrib
        if "name" in s_attrib:
            site.set("name", "lh_" + s_attrib["name"])
        if "pos" in s_attrib:
            site.set(
                "pos",
                _fmt_floats(BODY_LOCAL_REFLECTION @ _parse_floats(s_attrib["pos"])),
            )
        if "euler" in s_attrib or "quat" in s_attrib:
            _set_elem_rotation(
                site,
                BODY_LOCAL_REFLECTION @ _elem_rotation(site) @ BODY_LOCAL_REFLECTION,
            )


def _mirror_left_hand_tree(body: Any, *, is_root: bool = False) -> None:
    """Mirror a canonical right-hand body tree into the sagittal-reflected left hand 'lh_'."""
    attrib = body.attrib
    attrib["name"] = "lh_" + attrib["name"]
    if "pos" in attrib:
        pos = _parse_floats(attrib["pos"])
        if is_root:
            pos[0] = THORAX_SAGITTAL_X_M
            body.set("pos", _fmt_floats(SAGITTAL_WORLD_REFLECTION @ pos))
        else:
            body.set("pos", _fmt_floats(BODY_LOCAL_REFLECTION @ pos))
    rot = _elem_rotation(body)
    reflected_rot = (
        SAGITTAL_WORLD_REFLECTION @ rot @ BODY_LOCAL_REFLECTION
        if is_root
        else BODY_LOCAL_REFLECTION @ rot @ BODY_LOCAL_REFLECTION
    )
    _set_elem_rotation(body, reflected_rot)
    _mirror_left_hand_inertial(body)
    _mirror_left_hand_children(body)
    for sub in body.findall("body"):
        _mirror_left_hand_tree(sub, is_root=False)


def _build_hand_chain_files(
    myo_sim_dir: Path, assets_out_dir: Path
) -> tuple[Path, Path]:
    """Write assets/myohand_right_chain.xml and assets/myohand_left_chain.xml."""
    body_xml_path = myo_sim_dir / "hand/assets/myohand_body.xml"
    parsed = DefusedET.parse(body_xml_path)
    root = parsed.getroot()
    thorax_orig = root.find(".//body[@name='thorax']")
    if thorax_orig is None:
        raise ValueError(f"thorax body not found in {body_xml_path}")

    rh_thorax = DefusedET.fromstring(StdET.tostring(thorax_orig, encoding="unicode"))
    lh_thorax = DefusedET.fromstring(StdET.tostring(thorax_orig, encoding="unicode"))
    _ensure_explicit_inertials(rh_thorax)
    _ensure_explicit_inertials(lh_thorax)
    _prefix_right_hand_tree(rh_thorax, is_root=True)
    _mirror_left_hand_tree(lh_thorax, is_root=True)

    assets_out_dir.mkdir(parents=True, exist_ok=True)
    right_path = assets_out_dir / "myohand_right_chain.xml"
    left_path = assets_out_dir / "myohand_left_chain.xml"

    rh_wrap = StdET.Element("mujocoinclude", {"model": "myohand_right_chain"})
    rh_body_root = StdET.SubElement(
        rh_wrap, "body", {"name": "rh_full_body", "childclass": "myohand"}
    )
    rh_body_root.append(rh_thorax)

    lh_wrap = StdET.Element("mujocoinclude", {"model": "myohand_left_chain"})
    lh_body_root = StdET.SubElement(
        lh_wrap, "body", {"name": "lh_full_body", "childclass": "myohand"}
    )
    lh_body_root.append(lh_thorax)

    StdET.indent(rh_wrap, space="  ")
    StdET.indent(lh_wrap, space="  ")
    right_path.write_text(
        StdET.tostring(rh_wrap, encoding="unicode") + "\n", encoding="utf-8"
    )
    left_path.write_text(
        StdET.tostring(lh_wrap, encoding="unicode") + "\n", encoding="utf-8"
    )
    return right_path, left_path


def _build_bilateral_assets_xml(myo_sim_dir: Path) -> tuple[str, str, str, str]:
    """Return (<default>, <asset>, <contact>, <actuator>/<tendon>) XML blocks."""
    assets_path = myo_sim_dir / "hand/assets/myohand_assets.xml"
    root = DefusedET.parse(assets_path).getroot()
    default_el = root.find("default")
    asset_el = root.find("asset")
    contact_el = root.find("contact")
    tendon_el = root.find("tendon")
    actuator_el = root.find("actuator")
    if default_el is None or asset_el is None or contact_el is None:
        raise ValueError(f"Missing required sections in {assets_path}")

    for mesh_el in list(asset_el.findall("mesh")):
        m_copy = DefusedET.fromstring(StdET.tostring(mesh_el, encoding="unicode"))
        file_attr = m_copy.attrib["file"]
        base_name = m_copy.attrib.get("name") or Path(file_attr).stem
        m_copy.set("name", "lh_" + base_name)
        m_copy.set("scale", "1 1 -1")
        asset_el.append(m_copy)

    new_contact = StdET.Element("contact")
    for exc in contact_el.findall("exclude"):
        b1, b2 = exc.attrib["body1"], exc.attrib["body2"]
        StdET.SubElement(
            new_contact, "exclude", {"body1": f"rh_{b1}", "body2": f"rh_{b2}"}
        )
        StdET.SubElement(
            new_contact, "exclude", {"body1": f"lh_{b1}", "body2": f"lh_{b2}"}
        )

    new_tendon = StdET.Element("tendon")
    if tendon_el is not None:
        for prefix in ("rh_", "lh_"):
            for sp in tendon_el.findall("spatial"):
                sp_copy = DefusedET.fromstring(StdET.tostring(sp, encoding="unicode"))
                sp_copy.set("name", prefix + sp_copy.attrib["name"])
                for child in sp_copy:
                    if "site" in child.attrib:
                        child.set("site", prefix + child.attrib["site"])
                    if " sidesite" in child.attrib:
                        child.set("sidesite", prefix + child.attrib["sidesite"])
                    if "sidesite" in child.attrib:
                        child.set("sidesite", prefix + child.attrib["sidesite"])
                    if "geom" in child.attrib:
                        child.set("geom", prefix + child.attrib["geom"])
                new_tendon.append(sp_copy)

    new_actuator = StdET.Element("actuator")
    if actuator_el is not None:
        for prefix in ("rh_", "lh_"):
            for mus in actuator_el.findall("muscle"):
                m_copy = DefusedET.fromstring(StdET.tostring(mus, encoding="unicode"))
                m_copy.set("name", prefix + m_copy.attrib["name"])
                m_copy.set("tendon", prefix + m_copy.attrib["tendon"])
                new_actuator.append(m_copy)

    return (
        StdET.tostring(default_el, encoding="unicode"),
        StdET.tostring(asset_el, encoding="unicode"),
        StdET.tostring(new_contact, encoding="unicode"),
        StdET.tostring(new_tendon, encoding="unicode")
        + "\n"
        + StdET.tostring(new_actuator, encoding="unicode"),
    )


def _club_worldbody_xml(club: ClubSpec) -> str:
    """Build free-jointed golf_club body aligned with the bilateral wrist centres."""
    lead_below = DEFAULT_GRIP_POSE.distance_below_butt_m(Hand.LEAD)
    trail_below = DEFAULT_GRIP_POSE.distance_below_butt_m(Hand.TRAIL)
    y_l = -(club.length_m - lead_below)
    y_r = -(club.length_m - trail_below)
    # Place club so grip_site_club_l coincides with lh_lunate (+0.0381, -0.446507, 1.41751)
    # when rotated 90 deg about +z (+y_club -> -x_world): x_club - y_l = +0.0381
    club_x = 0.0381 + y_l
    club_y = -0.446507
    club_z = 1.41751
    head = club.head_half_size_m
    head_type = "ellipsoid" if club.head_shape == "ellipsoid" else "box"
    shaft_grip_mass = club.shaft_mass_kg + club.grip_mass_kg
    return (
        f'  <body name="golf_club" pos="{club_x:.6f} {club_y:.6f} {club_z:.6f}" '
        f'quat="0.70710678 0 0 0.70710678">\n'
        '    <freejoint name="club_free"/>\n'
        f'    <site name="grip_site_club_l" pos="0 {y_l:.6g} 0" size="0.006"/>\n'
        f'    <site name="grip_site_club_r" pos="0 {y_r:.6g} 0" size="0.006"/>\n'
        f'    <geom name="club_shaft" type="capsule" fromto="0 {-club.length_m:.6g} 0 0 0 0" '
        f'size="{club.shaft_radius_m:.6g}" mass="{shaft_grip_mass:.6g}" '
        'contype="1" conaffinity="1" rgba="0.75 0.75 0.8 1"/>\n'
        f'    <geom name="club_head" type="{head_type}" pos="0 0 0" '
        f'size="{head[0]:.6g} {head[1]:.6g} {head[2]:.6g}" mass="{club.head_mass_kg:.6g}" '
        'contype="1" conaffinity="1" rgba="0.1 0.1 0.15 1"/>\n'
        "  </body>"
    )


def _equality_joint_locks_xml() -> str:
    """Build <equality> joint locks for all 23 lh_* and 23 rh_* joints."""
    lines = ["<equality>"]
    for prefix in ("lh_", "rh_"):
        for jnt in MYOHAND_CANONICAL_JOINTS:
            full_jnt = f"{prefix}{jnt}"
            lines.append(
                f'  <joint name="lock_{full_jnt}" joint1="{full_jnt}" '
                'polycoef="0 0 0 0 0" active="true" solref="0.002 1" solimp="0.99 0.999 0.0001"/>'
            )
    lines.append("</equality>")
    return "\n".join(lines)


def _compose_scene_xml(
    club: ClubSpec,
    default_xml: str,
    asset_xml: str,
    contact_xml: str,
    tendon_actuator_xml: str,
) -> str:
    """Compose a complete bilateral myoHand + free club MJCF document."""
    club_xml = _club_worldbody_xml(club)
    eq_xml = _equality_joint_locks_xml()
    return (
        f'<mujoco model="myohand_bilateral_{club.name}">\n'
        '  <compiler inertiafromgeom="auto" angle="radian" balanceinertia="true" '
        'boundmass="0.001" boundinertia=".0001" meshdir="../../myo_sim" texturedir="../../myo_sim"/>\n'
        '  <size njmax="1000" nconmax="400" nuser_jnt="1"/>\n'
        '  <option timestep="0.002"/>\n'
        '  <include file="../../myo_sim/scene/myosuite_scene.xml"/>\n'
        f"  {default_xml}\n"
        f"  {asset_xml}\n"
        f"  {contact_xml}\n"
        "  <worldbody>\n"
        '    <include file="assets/myohand_right_chain.xml"/>\n'
        '    <include file="assets/myohand_left_chain.xml"/>\n'
        f"{club_xml}\n"
        "  </worldbody>\n"
        f"  {eq_xml}\n"
        f"  {tendon_actuator_xml}\n"
        "</mujoco>\n"
    )


def generate_myohand_bilateral_scene(
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> GeneratedMyoHandBilateralScenePaths:
    """Compose and write the bilateral myoHand driver/iron scenes and receipt JSON.

    Preconditions:
        ``shared/models/myosuite/myo_sim`` is populated at ``MYO_SIM_PIN_SHA``.
    Postconditions:
        Writes ``myohand_bilateral_driver.xml``, ``myohand_bilateral_iron.xml``,
        ``myohand_bilateral_receipt.json``, and ``assets/myohand_{right,left}_chain.xml``
        without copying any mesh or texture from ``myo_sim``.
    """
    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    myo_sim_dir = check_myohand_assets_available(root)
    out_dir = (root / HAND_SCENE_REL_DIR) if output_root is None else Path(output_root)
    assets_out = out_dir / "assets"
    right_chain, left_chain = _build_hand_chain_files(myo_sim_dir, assets_out)
    default_xml, asset_xml, contact_xml, ta_xml = _build_bilateral_assets_xml(
        myo_sim_dir
    )

    driver_spec = CLUBS["driver"]
    iron_spec = CLUBS["iron7"]
    driver_path = out_dir / "myohand_bilateral_driver.xml"
    iron_path = out_dir / "myohand_bilateral_iron.xml"
    driver_path.write_text(
        _compose_scene_xml(driver_spec, default_xml, asset_xml, contact_xml, ta_xml),
        encoding="utf-8",
    )
    iron_path.write_text(
        _compose_scene_xml(iron_spec, default_xml, asset_xml, contact_xml, ta_xml),
        encoding="utf-8",
    )

    source_hashes = {
        "hand/myohand.xml": _sha256(myo_sim_dir / "hand/myohand.xml"),
        "hand/assets/myohand_assets.xml": _sha256(
            myo_sim_dir / "hand/assets/myohand_assets.xml"
        ),
        "hand/assets/myohand_body.xml": _sha256(
            myo_sim_dir / "hand/assets/myohand_body.xml"
        ),
        "scene/myosuite_scene.xml": _sha256(myo_sim_dir / "scene/myosuite_scene.xml"),
    }
    receipt = {
        "schema_version": 1,
        "issue": 13330,
        "myo_sim_pin": MYO_SIM_PIN_SHA,
        "source_xml_sha256": source_hashes,
        "generated_assets_sha256": {
            "assets/myohand_right_chain.xml": _sha256(right_chain),
            "assets/myohand_left_chain.xml": _sha256(left_chain),
        },
        "clubs": {
            "driver": {
                "name": driver_spec.name,
                "length_m": driver_spec.length_m,
                "total_mass_kg": driver_spec.total_mass_kg,
                "scene_xml": driver_path.name,
                "sha256": _sha256(driver_path),
            },
            "iron": {
                "name": iron_spec.name,
                "length_m": iron_spec.length_m,
                "total_mass_kg": iron_spec.total_mass_kg,
                "scene_xml": iron_path.name,
                "sha256": _sha256(iron_path),
            },
        },
        "left_hand_mirror": {
            "supported": True,
            "method": "sagittal_world_reflection_S_W_and_body_local_reflection_S_B_with_mesh_scale_1_1_-1",
            "world_reflection_diag": [-1.0, 1.0, 1.0],
            "body_reflection_diag": [1.0, 1.0, -1.0],
            "mesh_scale": [1.0, 1.0, -1.0],
        },
        "qualification": {
            "is_contact_fixture": True,
            "muscle_recruitment_in_scope": False,
            "actuators_active": False,
            "notes": (
                "Bilateral myoHand articulated grip fixture (HRG-GS-M8, #13330). "
                "Muscle recruitment is out of scope; actuators remain inactive."
            ),
        },
    }
    receipt_path = out_dir / "myohand_bilateral_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return GeneratedMyoHandBilateralScenePaths(
        driver=driver_path,
        iron=iron_path,
        receipt=receipt_path,
        right_chain=right_chain,
        left_chain=left_chain,
    )


def _compare_reflected_pose(
    model: Any,
    data: Any,
    mujoco_mod: Any,
) -> tuple[float, float, float, float, float, float]:
    """Return max errors (xpos, xipos, rot, mass, inertia, palmar_normal) across all 37 bodies."""
    max_xpos = 0.0
    max_xipos = 0.0
    max_rot = 0.0
    max_mass = 0.0
    max_inertia = 0.0
    max_normal = 0.0
    obj_body = mujoco_mod.mjtObj.mjOBJ_BODY
    palmar_local = np.array([1.0, 0.0, 0.0], dtype=float)

    for bname in MYOHAND_CANONICAL_BODIES:
        id_r = int(mujoco_mod.mj_name2id(model, obj_body, f"rh_{bname}"))
        id_l = int(mujoco_mod.mj_name2id(model, obj_body, f"lh_{bname}"))
        xpos_err = float(
            np.linalg.norm(
                SAGITTAL_WORLD_REFLECTION @ data.xpos[id_r] - data.xpos[id_l]
            )
        )
        xipos_err = float(
            np.linalg.norm(
                SAGITTAL_WORLD_REFLECTION @ data.xipos[id_r] - data.xipos[id_l]
            )
        )
        rot_r = np.asarray(data.xmat[id_r], dtype=float).reshape(3, 3)
        rot_l = np.asarray(data.xmat[id_l], dtype=float).reshape(3, 3)
        rot_err = float(
            np.linalg.norm(
                SAGITTAL_WORLD_REFLECTION @ rot_r @ BODY_LOCAL_REFLECTION - rot_l
            )
        )
        mass_err = abs(float(model.body_mass[id_r]) - float(model.body_mass[id_l]))

        ri_r = np.asarray(data.ximat[id_r], dtype=float).reshape(3, 3)
        ri_l = np.asarray(data.ximat[id_l], dtype=float).reshape(3, 3)
        iw_r = ri_r @ np.diag(model.body_inertia[id_r]) @ ri_r.T
        iw_l = ri_l @ np.diag(model.body_inertia[id_l]) @ ri_l.T
        in_err = float(
            np.linalg.norm(
                SAGITTAL_WORLD_REFLECTION @ iw_r @ SAGITTAL_WORLD_REFLECTION - iw_l
            )
        )
        n_r = rot_r @ palmar_local
        n_l = rot_l @ (BODY_LOCAL_REFLECTION @ palmar_local)
        norm_err = float(np.linalg.norm(SAGITTAL_WORLD_REFLECTION @ n_r - n_l))

        max_xpos = max(max_xpos, xpos_err)
        max_xipos = max(max_xipos, xipos_err)
        max_rot = max(max_rot, rot_err)
        max_mass = max(max_mass, mass_err)
        max_inertia = max(max_inertia, in_err)
        max_normal = max(max_normal, norm_err)
    return max_xpos, max_xipos, max_rot, max_mass, max_inertia, max_normal


def verify_left_hand_sagittal_reflection(
    repo_root: Path | None = None,
) -> LeftHandMirrorReport:
    """Verify that the mirrored left myoHand is an exact sagittal reflection of the right hand."""
    import mujoco

    root = _REPO_ROOT if repo_root is None else Path(repo_root)
    scene_path = root / HAND_SCENE_REL_DIR / "myohand_bilateral_driver.xml"
    if not scene_path.is_file():
        generate_myohand_bilateral_scene(repo_root=root)

    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    xpos0, xipos0, rot0, mass0, in0, norm0 = _compare_reflected_pose(
        model, data, mujoco
    )

    obj_jnt = mujoco.mjtObj.mjOBJ_JOINT
    for jname in MYOHAND_CANONICAL_JOINTS:
        jid_r = int(mujoco.mj_name2id(model, obj_jnt, f"rh_{jname}"))
        jid_l = int(mujoco.mj_name2id(model, obj_jnt, f"lh_{jname}"))
        q_lo, q_hi = model.jnt_range[jid_r]
        q_val = float(np.clip(0.35 * (q_lo + q_hi) + 0.15, q_lo, q_hi))
        data.qpos[int(model.jnt_qposadr[jid_r])] = q_val
        data.qpos[int(model.jnt_qposadr[jid_l])] = q_val
    mujoco.mj_forward(model, data)
    xpos_art, xipos_art, rot_art, _, in_art, norm_art = _compare_reflected_pose(
        model, data, mujoco
    )

    return LeftHandMirrorReport(
        supported=True,
        body_count=len(MYOHAND_CANONICAL_BODIES),
        joint_count=len(MYOHAND_CANONICAL_JOINTS),
        max_xpos_error_m=xpos0,
        max_xipos_error_m=xipos0,
        max_rotation_error=max(rot0, rot_art),
        max_mass_error_kg=mass0,
        max_world_inertia_error_kg_m2=max(in0, in_art),
        max_articulated_xpos_error_m=xpos_art,
        max_articulated_xipos_error_m=xipos_art,
        max_normal_reflection_error=max(norm0, norm_art),
    )
