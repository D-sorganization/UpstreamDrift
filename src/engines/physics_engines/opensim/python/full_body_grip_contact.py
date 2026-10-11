"""Elastic-foundation contact grip for the OpenSim full-body export (#11739, OSV-7).

Emits, for the ``contact`` grip model, the model elements that carry a
distributed hand-club interface:

* one ``ContactSphere`` per pad of the shared
  :class:`~src.shared.python.grip_contact.pad_layout.PadLayout`, fixed on the
  hand body at the pad's position in the hand grip frame;
* one closed ``ContactMesh`` per hand (a capped cylinder of the grip radius on
  the club body, written as an OBJ in ``mesh_dir``): the elastic foundation;
* one ``ElasticFoundationForce`` per pad (so the per-pad force is a record
  value), parameters from
  :func:`~src.shared.python.grip_contact.elastic_foundation.elastic_foundation_parameters`.

Force names start with :data:`CONTACT_FORCE_PREFIX`.  The recorded wrench of a
pad is the loading on each body of the pair; the club-side record is the force
exerted BY THE HAND ON THE CLUB.  ``ElasticFoundationForce`` requires closed
meshes and absolute mesh paths, and sphere-sphere or sphere-cylinder
Hunt-Crossley pairs give no force (see GRIP_PARITY_DECISIONS.md section 19).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.elastic_foundation import (
    ElasticFoundationParameters,
    elastic_foundation_parameters,
)
from src.shared.python.grip_contact.grip_mesh import (
    capped_cylinder_mesh,
    write_obj,
)
from src.shared.python.grip_contact.pad_contact import PadContactModel
from src.shared.python.grip_contact.pad_layout import PadLayout

CONTACT_FORCE_PREFIX = "grip_contact"
SIDES = ("L", "R")
#: Mesh margin beyond the outer pad rings of a hand [m] (allows axial slip).
MESH_MARGIN_M = 0.03


def _fmt(values: Sequence[float] | np.ndarray) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def pad_name(side: str, index: int) -> str:
    """Name of pad ``index`` of ``side`` (also the suffix of its force)."""
    return f"{side}{index:02d}"


@dataclass(frozen=True)
class ContactGripConfig:
    """Everything the OpenSim builder needs for the ``contact`` grip.

    ``mesh_dir`` is an existing directory for the grip OBJ files (OpenSim needs
    absolute mesh paths); ``foundation`` defaults to the parameters matched to
    ``pads``.
    """

    pads: PadContactModel
    mesh_dir: Path
    foundation: ElasticFoundationParameters | None = None
    mesh_segments: int = 96
    ring_pitch_m: float = 1.0e-3

    def __post_init__(self) -> None:
        mesh_dir = Path(self.mesh_dir)
        if not mesh_dir.is_absolute() or not mesh_dir.is_dir():
            raise ValueError(
                f"mesh_dir must be an existing absolute directory, got {mesh_dir}"
            )
        object.__setattr__(self, "mesh_dir", mesh_dir)
        if self.foundation is None:
            object.__setattr__(
                self, "foundation", elastic_foundation_parameters(self.pads)
            )

    @property
    def ef(self) -> ElasticFoundationParameters:
        """The (never ``None``) foundation parameters."""
        assert self.foundation is not None
        return self.foundation

    @property
    def layout(self) -> PadLayout:
        """The foundation's pad layout (delegates through ``ef``)."""
        return self.ef.layout

    def origin_axial_m(self, side: str) -> float:
        """Axial origin of ``side``'s pad cylinder (delegates through ``pads``)."""
        cylinder = self.pads.cylinder
        return cylinder.origin_axial_m(side)

    def mesh_path(self, side: str) -> Path:
        """OBJ path of ``side``'s grip mesh."""
        return self.mesh_dir / f"grip_mesh_{side}.obj"


def _grip_axis(interface: GripInterface, cfg: ContactGripConfig) -> tuple:
    rot = np.asarray(interface.right.rotation, dtype=float)
    point = np.asarray(interface.right.position_m) + rot @ (
        cfg.layout.axis_offset_grip_frame("R")
    )
    return point, rot[:, 0], rot[:, 1]


def write_grip_meshes(interface: GripInterface, cfg: ContactGripConfig) -> dict:
    """Write the two closed grip meshes (club-body frame); return their facts."""
    point, axis, radial = _grip_axis(interface, cfg)
    layout = cfg.layout
    half = max(layout.axial_offsets_m(), key=abs)
    half = abs(half) + layout.pad_radius_m + MESH_MARGIN_M
    facts: dict[str, Any] = {}
    for side in SIDES:
        centre = cfg.origin_axial_m(side)
        verts, faces = capped_cylinder_mesh(
            layout.grip_radius_m,
            point,
            axis,
            radial,
            (centre - half, centre + half),
            cfg.mesh_segments,
            cfg.ring_pitch_m,
        )
        write_obj(cfg.mesh_path(side), verts, faces)
        facts[side] = {
            "file": str(cfg.mesh_path(side)),
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            "axial_range_m": [centre - half, centre + half],
        }
    return facts


def _contact_sphere(
    parent: ET.Element, name: str, body: str, location: np.ndarray, radius: float
) -> None:
    cs = ET.SubElement(parent, "ContactSphere", attrib={"name": name})
    ET.SubElement(cs, "socket_frame").text = f"/bodyset/{body}"
    ET.SubElement(cs, "location").text = _fmt(location)
    ET.SubElement(cs, "orientation").text = "0 0 0"
    ET.SubElement(cs, "radius").text = format(radius, ".17g")


def _contact_mesh(parent: ET.Element, name: str, body: str, path: Path) -> None:
    cm = ET.SubElement(parent, "ContactMesh", attrib={"name": name})
    ET.SubElement(cm, "socket_frame").text = f"/bodyset/{body}"
    ET.SubElement(cm, "location").text = "0 0 0"
    ET.SubElement(cm, "orientation").text = "0 0 0"
    ET.SubElement(cm, "filename").text = str(path)


def _foundation_force(
    parent: ET.Element,
    name: str,
    geometry: Sequence[str],
    ef: ElasticFoundationParameters,
) -> None:
    force = ET.SubElement(parent, "ElasticFoundationForce", attrib={"name": name})
    pset = ET.SubElement(
        force,
        "ElasticFoundationForce::ContactParametersSet",
        attrib={"name": "contact_parameters"},
    )
    objects = ET.SubElement(pset, "objects")
    params = ET.SubElement(objects, "ElasticFoundationForce::ContactParameters")
    ET.SubElement(params, "geometry").text = " ".join(geometry)
    for key, value in (
        ("stiffness", ef.stiffness_n_m3),
        ("dissipation", ef.dissipation_s_m),
        ("static_friction", ef.static_friction),
        ("dynamic_friction", ef.dynamic_friction),
        ("viscous_friction", ef.viscous_friction),
    ):
        ET.SubElement(params, key).text = format(value, ".17g")
    ET.SubElement(pset, "groups")
    ET.SubElement(force, "transition_velocity").text = format(
        ef.transition_velocity_m_s, ".17g"
    )


def build_contact_grip(
    model: ET.Element,
    spec: Mapping[str, Any],
    interface: GripInterface,
    cfg: ContactGripConfig,
    body_names: Mapping[str, str],
) -> dict[str, Any]:
    """Append pads, meshes and per-pad forces to ``model``; return a receipt.

    ``spec`` is the bushing-topology spec (``grip_bushing`` frames) and
    ``body_names`` maps the spec body names of the hands and the club to the
    OpenSim body names.  Requires the model to already hold ``ForceSet`` and
    ``ContactGeometrySet`` elements.

    Raises:
        ValueError: if the spec has no ``grip_bushing`` frames or the model
            lacks the force or geometry sets.
    """
    frames = spec.get("grip_bushing")
    forces = model.find("ForceSet/objects")
    geometry = model.find("ContactGeometrySet/objects")
    if frames is None or forces is None or geometry is None:
        raise ValueError(
            "contact grip needs grip_bushing frames, ForceSet and geometry"
        )
    meshes = write_grip_meshes(interface, cfg)
    ef, layout = cfg.ef, cfg.ef.layout
    club = body_names[frames["L"]["club_body"]]
    for side in SIDES:
        _contact_mesh(geometry, f"grip_mesh_{side}", club, cfg.mesh_path(side))
        hand = body_names[frames[side]["hand_body"]]
        hand_frame = np.asarray(frames[side]["hand_frame"], dtype=float)
        pads = layout.positions_grip_frame(side)
        for k, local in enumerate(pads):
            pname = pad_name(side, k)
            location = hand_frame[:3, :3] @ local + hand_frame[:3, 3]
            _contact_sphere(
                geometry, f"grip_pad_{pname}", hand, location, layout.pad_radius_m
            )
            _foundation_force(
                forces,
                f"{CONTACT_FORCE_PREFIX}_{pname}",
                (f"grip_pad_{pname}", f"grip_mesh_{side}"),
                ef,
            )
    return {
        "pads_per_hand": layout.pad_count,
        "pad_radius_m": layout.pad_radius_m,
        "grip_radius_m": layout.grip_radius_m,
        "meshes": meshes,
        "foundation": {
            "stiffness_n_m3": ef.stiffness_n_m3,
            "dissipation_s_m": ef.dissipation_s_m,
            "static_friction": ef.static_friction,
            "dynamic_friction": ef.dynamic_friction,
            "viscous_friction": ef.viscous_friction,
            "transition_velocity_m_s": ef.transition_velocity_m_s,
            "preload_penetration_m": ef.preload_penetration_m,
            "force_per_pad_n": ef.force_per_pad_n,
        },
    }
