"""Improved club with a two-hand grip for the Rajagopal OpenSim models (OSV-9).

The musculoskeletal golf models (``golf_humanoid.osim``, its scaled variant and
the muscle model built from them) carry the same club as the generated
full-body models, from the same shared sources:

* geometry: ``model_appearance.club_assembly`` (head, shaft and grip meshes
  with the address-square face roll), written by ``club_visuals``;
* mass, centre of mass and inertia: ``grip_contact.ClubDynamics`` (the club
  solids of the full-body spec, hands excluded);
* grip frames: ``grip_contact.GripInterface`` (lead and trail positions along
  the shaft, shaft-aligned grip frame, designed bushing parameters).

The ``Club`` body frame is the spec club-body frame (origin at the sole point,
shaft toward the grip along -y, shaft axis at z = ``axis_offset_m``), so the
meshes need no offset frame and the shared face roll applies unchanged.

Topology matches the generated models. ``grip_model="weld"``: the club is the
child of the lead hand (``WeldJoint hand_l_to_club``) and the trail hand is
closed by ``WeldConstraint hand_r_to_club``. ``grip_model="bushing"``: the club
is a free body and each hand holds it through its own ``BushingForce``
(frame1 = hand, frame2 = club, so the recorded wrench acts on the club).

The hand-side grip frames come from an address calibration (see
``msk_club_calibration``): the hand poses at the captured address pose, with
the club placed where the generated model holds it. The calibration is
committed next to the models; this module is pure XML and needs no OpenSim.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # construction only
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as SafeET
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python import club_visuals
from src.engines.physics_engines.opensim.python.full_body_osim import (
    _format_numbers,
    _make_physical_offset_frame,
    _rotation_matrix_to_xyz_euler,
)
from src.shared.python.grip_contact import (
    BushingState,
    ClubDynamics,
    ContactTopology,
    GripCapabilities,
    GripInterface,
    GripStepReport,
    GripStyle,
    HandClubBinding,
    ModelGripAdapter,
    TopologyContactModel,
    apply_report,
    build_topology_model,
    bushing_wrench,
    contact_topology,
    evaluate_contacts,
    topology_side_wrench,
    validate_binding,
)
from src.shared.python.grip_contact.binding_manifest import binding_from_manifest, validate_manifest
from src.shared.python.grip_contact.parity import QuantityError, compute_quantity_error
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    assembly_from_spec,
    clubface_centre,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

OPENSIM_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = OPENSIM_DIR / "models"
CALIBRATION_PATH = MODELS_DIR / "msk_club_grip_calibration.json"
REPO_ROOT = Path(__file__).resolve().parents[5]
SPEC_DIR = REPO_ROOT / "docs" / "development" / "full_body_models"
MANIFESTS_DIR = REPO_ROOT / "src" / "config" / "grip_bindings"
CALIBRATION_SCHEMA = "msk-club-grip-calibration-v1"

#: XML element type of the model trees (built here; parsed with defusedxml).
Element = ET.Element
CLUB_BODY = "Club"
#: Clubs with a committed anthropometric spec (``spec_path``).
CLUBS = ("driver", "iron7")
GRIP_MODELS = ("weld", "bushing")
DEFAULT_GRIP_MODEL = "weld"
#: Hand bodies (Rajagopal names) by grip side: lead = left, trail = right.
HAND_BODIES = {"L": "hand_l", "R": "hand_r"}
LEAD_JOINT = "hand_l_to_club"
TRAIL_CONSTRAINT = "hand_r_to_club"
FREE_JOINT = "ground_to_club"
#: ``grip_model="bushing"`` force names by grip side.
GRIP_BUSHINGS = {"L": "grip_bushing_left", "R": "grip_bushing_right"}
FREE_COORDINATES = (
    "club_rx",
    "club_ry",
    "club_rz",
    "club_tx",
    "club_ty",
    "club_tz",
)
#: Frame names kept from the one-hand model so FK consumers keep working.
CLUB_HEAD_FRAME = "club_head_offset"
CLUB_GRIP_FRAMES = {"L": "club_grip_offset", "R": "club_trail_grip_offset"}
HAND_GRIP_FRAMES = {"L": "hand_l_grip_offset", "R": "hand_r_grip_offset"}
#: Point in the Rajagopal hand frame the shaft axis passes through: palm side
#: (+x) of the metacarpal heads, mid-hand (y along the forearm, z radial).
#: From the hand bone meshes (metacarpal heads at y = -0.085, proximal
#: phalanges at x up to 0.022, scale 0.85); identical for both hands.
HAND_GRIP_POINT_M = (0.022, -0.085, 0.0)
_SOCKETS = {"L": "/bodyset/hand_l", "R": "/bodyset/hand_r"}
DEFAULT_HOLD_SQUEEZE_N = 1100.0
GRAVITY_M_S2 = (0.0, -9.80665, 0.0)


def _coerce_grip_style(grip_style: GripStyle | str) -> GripStyle:
    """Validate and coerce ``grip_style`` into a :class:`GripStyle`."""
    if isinstance(grip_style, GripStyle):
        return grip_style
    if isinstance(grip_style, str):
        try:
            return GripStyle(grip_style)
        except ValueError as exc:
            valid = [s.value for s in GripStyle]
            raise ValueError(
                f"grip_style must be one of {valid}, got {grip_style!r}"
            ) from exc
    raise TypeError(
        f"grip_style must be GripStyle or str, got {type(grip_style).__name__}"
    )


def _validate_trigger_finger(trigger_finger: bool) -> bool:
    """Validate that ``trigger_finger`` is a strict ``bool``."""
    if not isinstance(trigger_finger, bool):
        raise TypeError(
            f"trigger_finger must be bool, got {type(trigger_finger).__name__}"
        )
    return trigger_finger


@dataclass(frozen=True)
class MskClub:
    """The shared club as the musculoskeletal models carry it.

    ``grip_points`` are the lead/trail grip points on the shaft axis (club
    frame); ``grip_rotation`` is the shared shaft-aligned grip frame.
    """

    assembly: ClubAssembly
    dynamics: ClubDynamics
    interface: GripInterface
    grip_points: dict[str, np.ndarray]
    grip_rotation: np.ndarray
    face_centre: np.ndarray
    grip_style: GripStyle = GripStyle.OVERLAP
    trigger_finger: bool = False
    topology: ContactTopology = field(
        default_factory=lambda: contact_topology(GripSpec(style=GripStyle.OVERLAP))
    )


def spec_path(club: str) -> Path:
    """Generated full-body spec that defines ``club`` (``driver`` or ``iron7``)."""
    if club not in CLUBS:
        raise ValueError(f"club must be one of {CLUBS}, got {club!r}")
    return SPEC_DIR / f"full_body_spec_anthro_{club}.json"


def msk_club(
    spec: Mapping[str, Any],
    *,
    grip_style: GripStyle | str = GripStyle.OVERLAP,
    trigger_finger: bool = False,
) -> MskClub:
    """The club of a ``full-body-v1`` spec, for the musculoskeletal models.

    Grip points are the GripInterface positions projected onto the shaft axis
    (the generated lead frame is the wrist centre, 2.5 cm off the axis; the
    Rajagopal hands have their own wrists). Raises ``ValueError`` when the
    spec carries no club.
    """
    style = _coerce_grip_style(grip_style)
    trig = _validate_trigger_finger(trigger_finger)
    assembly = assembly_from_spec(spec)
    if assembly is None:
        raise ValueError("spec has no club body with a Grip solid")
    interface = GripInterface.from_spec(spec)
    points: dict[str, np.ndarray] = {}
    for side in HAND_BODIES:
        side_frame = interface.frame(side)
        y_pos = float(side_frame.position_m[1])
        points[side] = np.array([0.0, y_pos, assembly.axis_offset_m])
    right_frame = interface.right
    topo = contact_topology(GripSpec(style=style, trigger_finger=trig))
    return MskClub(
        assembly=assembly,
        dynamics=ClubDynamics.from_spec(spec),
        interface=interface,
        grip_points=points,
        grip_rotation=np.asarray(right_frame.rotation, dtype=float),
        face_centre=np.asarray(clubface_centre(assembly), dtype=float),
        grip_style=style,
        trigger_finger=trig,
        topology=topo,
    )


def load_msk_club(
    club: str = "driver",
    *,
    grip_style: GripStyle | str = GripStyle.OVERLAP,
    trigger_finger: bool = False,
) -> MskClub:
    """:func:`msk_club` of the committed anthropometric spec of ``club``."""
    spec_file = spec_path(club)
    spec_doc = json.loads(spec_file.read_text(encoding="utf-8"))
    return msk_club(spec_doc, grip_style=grip_style, trigger_finger=trigger_finger)


# ------------------------------------------------------------- calibration
@dataclass(frozen=True)
class GripCalibration:
    """Hand-side grip frames and the address pose of one model and club.

    ``hand_frames[side]`` is the 4x4 pose of the club grip frame in the hand
    body at the captured address pose; ``address_q`` the coordinate values
    (rad / m) of that pose; ``club_in_ground`` the club body pose there.
    """

    model: str
    club: str
    hand_frames: dict[str, np.ndarray]
    address_q: dict[str, float]
    club_in_ground: np.ndarray
    report: dict[str, Any]

    def __post_init__(self) -> None:
        if set(self.hand_frames) != set(HAND_BODIES):
            raise ValueError("hand_frames needs the L and R grip frames")
        for frame in (*self.hand_frames.values(), self.club_in_ground):
            mat = np.asarray(frame, dtype=float)
            if mat.shape != (4, 4) or not np.isfinite(mat).all():
                raise ValueError("calibration frames must be finite 4x4 matrices")
        if not all(math.isfinite(v) for v in self.address_q.values()):
            raise ValueError("address_q values must be finite")

    def to_json(self) -> dict[str, Any]:
        """JSON-ready dictionary (inverse of :func:`calibration_from_json`)."""
        return {
            "hand_frames": {
                k: np.asarray(v).tolist() for k, v in self.hand_frames.items()
            },
            "address_q": dict(sorted(self.address_q.items())),
            "club_in_ground": np.asarray(self.club_in_ground).tolist(),
            "report": self.report,
        }


def calibration_from_json(
    model: str, club: str, entry: Mapping[str, Any]
) -> GripCalibration:
    """Rebuild a :class:`GripCalibration` from its JSON entry."""
    return GripCalibration(
        model=model,
        club=club,
        hand_frames={k: np.asarray(v, float) for k, v in entry["hand_frames"].items()},
        address_q={k: float(v) for k, v in entry["address_q"].items()},
        club_in_ground=np.asarray(entry["club_in_ground"], float),
        report=dict(entry.get("report", {})),
    )


def load_calibration(
    model: str, club: str = "driver", path: Path = CALIBRATION_PATH
) -> GripCalibration:
    """The committed calibration of ``model`` holding ``club``.

    Raises ``KeyError`` when no calibration was recorded for the pair (run
    ``msk_club_calibration`` first).
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        entry = doc["models"][model][club]
    except KeyError as exc:
        raise KeyError(
            f"no grip calibration for model {model!r}, club {club!r}"
        ) from exc
    return calibration_from_json(model, club, entry)


def store_calibration(
    calibration: GripCalibration, path: Path = CALIBRATION_PATH
) -> None:
    """Merge ``calibration`` into the committed calibration file."""
    file = Path(path)
    doc: dict[str, Any] = (
        json.loads(file.read_text(encoding="utf-8"))
        if file.is_file()
        else {"schema": CALIBRATION_SCHEMA, "models": {}}
    )
    doc["generator"] = "src.engines.physics_engines.opensim.python.msk_club_calibration"
    doc["hand_grip_point_m"] = list(HAND_GRIP_POINT_M)
    models = doc.setdefault("models", {})
    models.setdefault(calibration.model, {})[calibration.club] = calibration.to_json()
    file.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ------------------------------------------------------------- XML builders
def _objects(model: ET.Element, set_tag: str) -> ET.Element:
    found = model.find(f"{set_tag}/objects")
    if found is not None:
        return found
    container = model.find(set_tag)
    if container is None:
        container = ET.SubElement(model, set_tag, attrib={"name": set_tag.lower()})
    return ET.SubElement(container, "objects")


def _references_club(element: ET.Element) -> bool:
    texts = (e.text or "" for e in element.iter())
    return element.get("name") == CLUB_BODY or any(
        t.strip() in (CLUB_BODY, f"/bodyset/{CLUB_BODY}")
        or t.strip().startswith(f"/bodyset/{CLUB_BODY}/")
        for t in texts
    )


def strip_club(model: ET.Element) -> int:
    """Remove the club body and everything attached to it; returns the count.

    Removes the ``Club`` body, every joint, constraint and force that
    references it, and the hand-side grip frames this module adds.
    """
    removed = 0
    for set_tag in ("BodySet", "JointSet", "ConstraintSet", "ForceSet"):
        objects = model.find(f"{set_tag}/objects")
        if objects is None:
            continue
        for item in list(objects):
            if _references_club(item):
                objects.remove(item)
                removed += 1
    for body in _objects(model, "BodySet"):
        components = body.find("components")
        if components is None:
            continue
        for frame in list(components):
            if frame.get("name") in HAND_GRIP_FRAMES.values():
                components.remove(frame)
                removed += 1
    return removed


def _transform_frame(name: str, parent: str, transform: np.ndarray) -> ET.Element:
    mat = np.asarray(transform, dtype=float)
    return _make_physical_offset_frame(
        name, parent, mat[:3, 3], _rotation_matrix_to_xyz_euler(mat[:3, :3])
    )


def _club_body(club: MskClub, finish: str) -> ET.Element:
    """``<Body name="Club">``: shared mass properties, meshes and grip frames."""
    body = ET.Element("Body", attrib={"name": CLUB_BODY})
    dyn = club.dynamics
    ET.SubElement(body, "mass").text = format(dyn.mass_kg, ".17g")
    ET.SubElement(body, "mass_center").text = _format_numbers(dyn.com_m)
    inertia = np.asarray(dyn.inertia_com_kg_m2)
    ET.SubElement(body, "inertia").text = _format_numbers(
        [
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ]
    )
    components = ET.SubElement(body, "components")
    for side, name in CLUB_GRIP_FRAMES.items():
        grip = np.eye(4)
        grip[:3, :3] = club.grip_rotation
        grip[:3, 3] = club.grip_points[side]
        components.append(_transform_frame(name, "..", grip))
    head = np.eye(4)
    head[:3, 3] = club.face_centre
    components.append(_transform_frame(CLUB_HEAD_FRAME, "..", head))
    club_visuals.attach_club_meshes(body, club.assembly.head_alias, "", finish)
    return body


def _add_hand_frames(model: ET.Element, calibration: GripCalibration) -> None:
    bodies = {b.get("name"): b for b in _objects(model, "BodySet")}
    for side, hand in HAND_BODIES.items():
        if hand not in bodies:
            raise ValueError(f"model has no {hand} body")
        components = bodies[hand].find("components")
        if components is None:
            components = ET.SubElement(bodies[hand], "components")
        components.append(
            _transform_frame(
                HAND_GRIP_FRAMES[side], "..", calibration.hand_frames[side]
            )
        )


def _frame_path(side: str, kind: str) -> str:
    if kind == "hand":
        return f"{_SOCKETS[side]}/{HAND_GRIP_FRAMES[side]}"
    return f"/bodyset/{CLUB_BODY}/{CLUB_GRIP_FRAMES[side]}"


def _weld(model: ET.Element) -> None:
    joint = ET.SubElement(
        _objects(model, "JointSet"), "WeldJoint", {"name": LEAD_JOINT}
    )
    ET.SubElement(joint, "socket_parent_frame").text = _frame_path("L", "hand")
    ET.SubElement(joint, "socket_child_frame").text = _frame_path("L", "club")
    weld = ET.SubElement(
        _objects(model, "ConstraintSet"), "WeldConstraint", {"name": TRAIL_CONSTRAINT}
    )
    ET.SubElement(weld, "isEnforced").text = "true"
    ET.SubElement(weld, "socket_frame1").text = _frame_path("R", "hand")
    ET.SubElement(weld, "socket_frame2").text = _frame_path("R", "club")


def _free_joint(model: ET.Element, club_in_ground: np.ndarray) -> None:
    joint = ET.SubElement(
        _objects(model, "JointSet"), "FreeJoint", {"name": FREE_JOINT}
    )
    ET.SubElement(joint, "socket_parent_frame").text = "/ground"
    ET.SubElement(joint, "socket_child_frame").text = f"/bodyset/{CLUB_BODY}"
    coords = ET.SubElement(joint, "coordinates")
    mat = np.asarray(club_in_ground, dtype=float)
    values = (*_rotation_matrix_to_xyz_euler(mat[:3, :3]), *mat[:3, 3])
    for name, value in zip(FREE_COORDINATES, values, strict=True):
        coord = ET.SubElement(coords, "Coordinate", {"name": name})
        ET.SubElement(coord, "default_value").text = format(float(value), ".17g")
        span = math.pi if name.startswith("club_r") else 5.0
        ET.SubElement(coord, "range").text = _format_numbers([-span, span])


def _bushings(model: ET.Element, club: MskClub) -> None:
    params = club.interface.bushing
    for side, name in GRIP_BUSHINGS.items():
        force = ET.SubElement(
            _objects(model, "ForceSet"), "BushingForce", {"name": name}
        )
        ET.SubElement(force, "socket_frame1").text = _frame_path(side, "hand")
        ET.SubElement(force, "socket_frame2").text = _frame_path(side, "club")
        for tag, values in (
            ("rotational_stiffness", params.rotational_stiffness_nm_rad),
            ("translational_stiffness", params.translational_stiffness_n_m),
            ("rotational_damping", params.rotational_damping_nms_rad),
            ("translational_damping", params.translational_damping_ns_m),
        ):
            ET.SubElement(force, tag).text = _format_numbers(values)


def set_default_pose(model: ET.Element, address_q: Mapping[str, float]) -> int:
    """Set each listed coordinate's ``default_value``; returns the count set."""
    count = 0
    for coord in model.iter("Coordinate"):
        name = coord.get("name")
        if name is None or name not in address_q:
            continue
        default = coord.find("default_value")
        if default is None:
            default = ET.SubElement(coord, "default_value")
        default.text = format(float(address_q[name]), ".17g")
        count += 1
    return count


def attach_club(
    model: ET.Element,
    club: MskClub,
    calibration: GripCalibration,
    *,
    grip_model: str = DEFAULT_GRIP_MODEL,
    finish: str = club_visuals.DEFAULT_FINISH,
    grip_style: GripStyle | str | None = None,
    trigger_finger: bool | None = None,
) -> None:
    """Replace any club on ``model`` by the shared club held in both hands.

    Postconditions: one ``Club`` body with the shared meshes, mass and inertia;
    lead and trail hand grip frames; ``weld`` adds the lead ``WeldJoint`` and
    trail ``WeldConstraint``, ``bushing`` a ``FreeJoint`` and two
    ``BushingForce``; coordinate defaults are the calibrated address pose so
    the default state is the closed address pose.
    """
    if grip_style is not None:
        _coerce_grip_style(grip_style)
    if trigger_finger is not None:
        _validate_trigger_finger(trigger_finger)
    if grip_model not in GRIP_MODELS:
        raise ValueError(f"grip_model must be one of {GRIP_MODELS}, got {grip_model!r}")
    assembly = club.assembly
    if calibration.club != assembly.head_alias:
        raise ValueError(
            f"calibration is for {calibration.club!r}, club is {assembly.head_alias!r}"
        )
    strip_club(model)
    _objects(model, "BodySet").append(_club_body(club, finish))
    _add_hand_frames(model, calibration)
    if grip_model == "weld":
        _weld(model)
    else:
        _free_joint(model, calibration.club_in_ground)
        _bushings(model, club)
    set_default_pose(model, calibration.address_q)


#: Rajagopal finger and thumb bone meshes (``index_proximal_rvs.vtp`` ...).
PHALANX_MESH_KEYS = ("_proximal_", "_medial_", "_distal_")


def _is_phalanx(mesh: ET.Element) -> bool:
    name = mesh.findtext("mesh_file") or ""
    return any(key in name for key in PHALANX_MESH_KEYS)


def hide_finger_bones(model: ET.Element) -> int:
    """Hide the phalanx meshes of both hand bodies; returns how many were hidden.

    The straight Rajagopal fingers never wrap a grip; :func:`attach_grip_hands`
    draws the curled-finger gloves in their place. Carpals and metacarpals stay
    visible: they bridge the wrist to the glove, whose palm is shorter than
    the Rajagopal hand. Visual only.
    """
    count = 0
    for body in _objects(model, "BodySet"):
        if body.get("name") not in HAND_BODIES.values():
            continue
        for mesh in filter(_is_phalanx, body.iter("Mesh")):
            appearance = mesh.find("Appearance")
            if appearance is None:
                appearance = ET.SubElement(mesh, "Appearance")
            visible = appearance.find("visible")
            if visible is None:
                visible = ET.SubElement(appearance, "visible")
            visible.text = "false"
            count += 1
    return count


def attach_grip_hands(model: ET.Element, club: MskClub) -> int:
    """Draw both wrapped-finger grip hands on the club and hide the finger bones.

    Precondition: ``model`` holds the ``Club`` body (:func:`attach_club`), else
    ``ValueError``. Postconditions: the club body carries exactly one lead and
    one trail glove mesh in its own frame (the shared OSV-2 placement, as in
    MuJoCo, Drake and Pinocchio, #12775); every phalanx mesh is hidden; no
    mass, inertia, joint or constraint changes. Idempotent; returns 2.
    """
    bodies = [b for b in _objects(model, "BodySet") if b.get("name") == CLUB_BODY]
    if len(bodies) != 1:
        raise ValueError("model has no Club body; call attach_club first")
    hide_finger_bones(model)
    assembly = club.assembly
    return club_visuals.attach_grip_hand_meshes(bodies[0], assembly.head_alias, "")


def _default_pose(model: ET.Element) -> dict[str, float]:
    pose = {}
    for coord in model.iter("Coordinate"):
        text = coord.findtext("default_value")
        if coord.get("name") and text is not None:
            pose[str(coord.get("name"))] = float(text)
    return pose


def grip_model_of(model: ET.Element) -> str:
    """``"bushing"`` when ``model``'s club hangs on a free joint, else ``"weld"``."""
    for joint in model.iter("FreeJoint"):
        if joint.get("name") == FREE_JOINT:
            return "bushing"
    return "weld"


def hand_grip_point(side: str) -> np.ndarray:
    """:data:`HAND_GRIP_POINT_M` for ``side`` (validated)."""
    if side not in HAND_BODIES:
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    return np.asarray(HAND_GRIP_POINT_M, dtype=float)


def coordinate_names(model: ET.Element) -> list[str]:
    """Coordinate names of ``model`` in document order."""
    return [str(c.get("name")) for c in model.iter("Coordinate") if c.get("name")]


def unlocked_coordinates(model: ET.Element, exclude: Sequence[str] = ()) -> list[str]:
    """Coordinates whose ``locked`` flag is not ``true`` (document order)."""
    names = []
    for coord in model.iter("Coordinate"):
        flag = coord.findtext("locked") or "false"
        locked = flag.strip().lower() == "true"
        name = coord.get("name")
        if name and not locked and name not in exclude:
            names.append(name)
    return names


# ------------------------------------------- HRG-GS-M4 OpenSim MSK Grip Adapter
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
        ("pro_sup_l", "wrist_flex_l", "wrist_dev_l", "pro_sup_r", "wrist_flex_r", "wrist_dev_r"),
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
        import opensim as osim

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
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".osim", dir=MODELS_DIR, delete=False) as tmp:
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
            pf = osim.PrescribedForce(f"shared_grip_force_{body_name}", body_set.get(body_name))
            pf.set_pointIsGlobal(False)
            pf.set_forceIsGlobal(True)
            pf.setForceFunctions(osim.Constant(0.0), osim.Constant(0.0), osim.Constant(0.0))
            pf.setPointFunctions(osim.Constant(0.0), osim.Constant(0.0), osim.Constant(0.0))
            pf.setTorqueFunctions(osim.Constant(0.0), osim.Constant(0.0), osim.Constant(0.0))
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
    rot_body = np.array([[float(rot_osim.get(i, j)) for j in range(3)] for i in range(3)])
    pos_body = np.array([float(pos_osim.get(i)) for i in range(3)])
    vel_spatial = body.getVelocityInGround(state)
    ang_osim, lin_osim = vel_spatial.get(0), vel_spatial.get(1)
    omega_w = np.array([float(ang_osim.get(i)) for i in range(3)])
    vel_w = np.array([float(lin_osim.get(i)) for i in range(3)])
    r_world = rot_body @ local_offset[:3, 3]
    return BushingState(
        rot_body @ local_offset[:3, :3], pos_body + r_world, vel_w + np.cross(omega_w, r_world), omega_w
    )


def _update_prescribed_force(
    pf: Any, force_w: np.ndarray, moment_w: np.ndarray, point_local: np.ndarray
) -> None:
    """Update the constant force, torque, and application point of a ``PrescribedForce``."""
    osim = _try_import_opensim()
    if osim is None:
        return
    cast = osim.Constant.safeDownCast
    f_set, t_set, p_set = pf.get_forceFunctions(), pf.get_torqueFunctions(), pf.get_pointFunctions()
    for idx in range(3):
        cast(f_set.get(idx)).setValue(float(force_w[idx]))
        cast(t_set.get(idx)).setValue(float(moment_w[idx]))
        cast(p_set.get(idx)).setValue(float(point_local[idx]))


class OpenSimMskGripAdapter(ModelGripAdapter):
    """Native OpenSim MSK ``ModelGripAdapter`` for both ``weld`` and ``bushing`` modes."""

    def __init__(
        self, model_id: str, club: MskClub, calibration: GripCalibration, manifest: Mapping[str, Any], *, grip_model: str = DEFAULT_GRIP_MODEL, osim_runtime: tuple[Any | None, Any | None, dict[str, Any], dict[str, Any]] | None = None
    ) -> None:
        if grip_model not in GRIP_MODELS:
            raise ValueError(f"grip_model must be one of {GRIP_MODELS}, got {grip_model!r}")
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
            return any(bool(bf.appliesForce(self.state)) for bf in self._bushings.values())
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
                    self.osim_model, self.state, binding.lead_hand_body, self._hand_iface_transforms["L"]
                ),
                binding.trail_hand_body: _osim_body_state(
                    self.osim_model, self.state, binding.trail_hand_body, self._hand_iface_transforms["R"]
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
            f_arr, m_arr = np.asarray(force, dtype=float), np.asarray(moment, dtype=float)
            if f_arr.shape != (3,) or m_arr.shape != (3,):
                raise ValueError(f"wrench for {body_name!r} must be a pair of 3-vectors")
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
    rot_c: np.ndarray, pos_c: np.ndarray, vel_c: np.ndarray, omega_c: np.ndarray, grip_local: np.ndarray
) -> BushingState:
    """World :class:`BushingState` of one club grip frame given club origin state."""
    r_w = rot_c @ grip_local[:3, 3]
    return BushingState(rot_c @ grip_local[:3, :3], pos_c + r_w, vel_c + np.cross(omega_c, r_w), omega_c)


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
            m_tot = m_tot + bw.moment_nm + np.cross(c_state.position_m - pos, bw.force_n)
        return f_tot, m_tot

    def _contact_fn(
        rot: np.ndarray, pos: np.ndarray, vel: np.ndarray, omg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        f_tot, m_tot = np.zeros(3), np.zeros(3)
        for side in ("L", "R"):
            c_state = _club_side_bushing_state(rot, pos, vel, omg, grip_local[side])
            sw = topology_side_wrench(topo_model, side, hand_fixed[side], c_state)
            f_tot = f_tot + sw.force_n
            m_tot = m_tot + sw.moment_nm + np.cross(c_state.position_m - pos, sw.force_n)
        return f_tot, m_tot

    return _bushing_fn, _contact_fn


def _make_hold_rhs(wrench_fn: Any, dyn: ClubDynamics, rot0: np.ndarray) -> Any:
    """Build ODE RHS ``dy/dt`` for 6-DOF free ``Club`` under gravity and ``wrench_fn``."""
    mass, com_local = float(dyn.mass_kg), np.asarray(dyn.com_m, dtype=float)
    inertia_local, grav = np.asarray(dyn.inertia_com_kg_m2, dtype=float), np.asarray(GRAVITY_M_S2, dtype=float)

    def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
        pos, vel, omg = y[:3], y[6:9], y[9:12]
        rot_rel = Rotation.from_rotvec(y[3:6])
        rot = rot_rel.as_matrix() @ rot0
        f_orig, m_orig = wrench_fn(rot, pos, vel, omg)
        r_com = rot @ com_local
        m_com = m_orig - np.cross(r_com, f_orig)
        i_world = rot @ inertia_local @ rot.T
        alpha = np.linalg.solve(i_world, m_com - np.cross(omg, i_world @ omg))
        acc_orig = (f_orig + mass * grav) / mass - np.cross(alpha, r_com) - np.cross(omg, np.cross(omg, r_com))
        return np.concatenate([vel, omg, acc_orig, alpha])

    return _rhs


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
        model_id, club=club, grip_model="bushing", grip_style=grip_style, trigger_finger=trigger_finger
    )
    msk = adapter.club
    topo, iface = msk.topology, msk.interface
    topo_model = build_topology_model(iface, topo, squeeze_per_hand_n=squeeze_per_hand_n)
    binding = adapter.resolve_binding(topo)
    init_states = adapter.sample_states(binding)
    lead_raw = init_states[binding.lead_hand_body]
    trail_raw = init_states[binding.trail_hand_body]
    club_raw = init_states[binding.club_body]
    assert isinstance(lead_raw, BushingState) and isinstance(trail_raw, BushingState)
    assert isinstance(club_raw, BushingState)
    hand_fixed = {
        "L": BushingState(lead_raw.rotation, lead_raw.position_m, np.zeros(3), np.zeros(3)),
        "R": BushingState(trail_raw.rotation, trail_raw.position_m, np.zeros(3), np.zeros(3)),
    }
    left_iface, right_iface = iface.left, iface.right
    grip_local = {"L": left_iface.matrix(), "R": right_iface.matrix()}
    bushing_fn, contact_fn = _make_hold_wrench_fns(topo_model, iface.bushing, hand_fixed, grip_local)
    rot0 = club_raw.rotation.copy()
    y0 = np.zeros(12)
    y0[:3] = club_raw.position_m
    t_eval = np.linspace(0.08, 0.12, 21)
    sol_b = solve_ivp(
        _make_hold_rhs(bushing_fn, msk.dynamics, rot0),
        (0.0, 0.12),
        y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        max_step=1e-3,
    )
    sol_c = solve_ivp(
        _make_hold_rhs(contact_fn, msk.dynamics, rot0),
        (0.0, 0.12),
        y0,
        method="Radau",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        max_step=1e-3,
    )
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
            club_seq[side].append(_club_side_bushing_state(rc, yc[:3], yc[6:9], yc[9:12], grip_local[side]))
    b_f_arr, b_m_arr = np.asarray(b_forces, float), np.asarray(b_moments, float)
    c_f_arr, c_m_arr = np.asarray(c_forces, float), np.asarray(c_moments, float)
    times = t_eval - t_eval[0]
    hand_seq = {"L": [hand_fixed["L"] for _ in times], "R": [hand_fixed["R"] for _ in times]}
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

