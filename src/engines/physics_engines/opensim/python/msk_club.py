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

import json
import math
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # construction only
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import club_visuals
from src.engines.physics_engines.opensim.python.full_body_osim import (
    _format_numbers,
    _make_physical_offset_frame,
    _rotation_matrix_to_xyz_euler,
)
from src.shared.python.grip_contact import ClubDynamics, GripInterface
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    assembly_from_spec,
    clubface_centre,
)

OPENSIM_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = OPENSIM_DIR / "models"
CALIBRATION_PATH = MODELS_DIR / "msk_club_grip_calibration.json"
REPO_ROOT = Path(__file__).resolve().parents[5]
SPEC_DIR = REPO_ROOT / "docs" / "development" / "full_body_models"
CALIBRATION_SCHEMA = "msk-club-grip-calibration-v1"

#: XML element type of the model trees (built here; parsed with defusedxml).
Element = ET.Element
CLUB_BODY = "Club"
GRIP_MODELS = ("weld", "bushing")
DEFAULT_GRIP_MODEL = "weld"
#: Hand bodies (Rajagopal names) by grip side: lead = left, trail = right.
HAND_BODIES = {"L": "hand_l", "R": "hand_r"}
LEAD_JOINT = "hand_l_to_club"
TRAIL_CONSTRAINT = "hand_r_to_club"
FREE_JOINT = "ground_to_club"
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


def spec_path(club: str) -> Path:
    """Generated full-body spec that defines ``club`` (``driver`` or ``iron7``)."""
    if club not in ("driver", "iron7"):
        raise ValueError(f"club must be 'driver' or 'iron7', got {club!r}")
    return SPEC_DIR / f"full_body_spec_anthro_{club}.json"


def msk_club(spec: Mapping[str, Any]) -> MskClub:
    """The club of a ``full-body-v1`` spec, for the musculoskeletal models.

    Grip points are the GripInterface positions projected onto the shaft axis
    (the generated lead frame is the wrist centre, 2.5 cm off the axis; the
    Rajagopal hands have their own wrists). Raises ``ValueError`` when the
    spec carries no club.
    """
    assembly = assembly_from_spec(spec)
    if assembly is None:
        raise ValueError("spec has no club body with a Grip solid")
    interface = GripInterface.from_spec(spec)
    points = {
        side: np.array(
            [0.0, float(interface.frame(side).position_m[1]), assembly.axis_offset_m]
        )
        for side in HAND_BODIES
    }
    return MskClub(
        assembly=assembly,
        dynamics=ClubDynamics.from_spec(spec),
        interface=interface,
        grip_points=points,
        grip_rotation=np.asarray(interface.right.rotation, dtype=float),
        face_centre=np.asarray(clubface_centre(assembly), dtype=float),
    )


def load_msk_club(club: str = "driver") -> MskClub:
    """:func:`msk_club` of the committed anthropometric spec of ``club``."""
    return msk_club(json.loads(spec_path(club).read_text(encoding="utf-8")))


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
    for side, label in (("L", "left"), ("R", "right")):
        force = ET.SubElement(
            _objects(model, "ForceSet"),
            "BushingForce",
            {"name": f"grip_bushing_{label}"},
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
) -> None:
    """Replace any club on ``model`` by the shared club held in both hands.

    Postconditions: one ``Club`` body with the shared meshes, mass and inertia;
    lead and trail hand grip frames; ``weld`` adds the lead ``WeldJoint`` and
    trail ``WeldConstraint``, ``bushing`` a ``FreeJoint`` and two
    ``BushingForce``; coordinate defaults are the calibrated address pose so
    the default state is the closed address pose.
    """
    if grip_model not in GRIP_MODELS:
        raise ValueError(f"grip_model must be one of {GRIP_MODELS}, got {grip_model!r}")
    if calibration.club != club.assembly.head_alias:
        raise ValueError(
            f"calibration is for {calibration.club!r}, club is {club.assembly.head_alias!r}"
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
        locked = (coord.findtext("locked") or "false").strip().lower() == "true"
        name = coord.get("name")
        if name and not locked and name not in exclude:
            names.append(name)
    return names
