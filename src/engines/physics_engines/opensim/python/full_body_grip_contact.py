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

import math

from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.elastic_foundation import (
    ElasticFoundationParameters,
    elastic_foundation_for_law,
    elastic_foundation_parameters,
)
from src.shared.python.grip_contact.grip_mesh import (
    capped_cylinder_mesh,
    write_obj,
)
from src.shared.python.grip_contact.contact_topology import DEFAULT_TOPOLOGY_PRELOAD_M
from src.shared.python.grip_contact.pad_contact import PadContactModel
from src.shared.python.grip_contact.pad_layout import PAD_RADIUS_M, PadLayout
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    map_site_to_hand_frame,
)


def capsule_sphere_count(half_length_m: float, radius_m: float) -> int:
    """Number of discrete spheres approximating a capsule medial axis in OpenSim.

    Preconditions:
        ``half_length_m`` must be a nonnegative finite float.
        ``radius_m`` must be a positive finite float.

    Postconditions:
        Returns an integer >= 2 ensuring sphere center spacing <= radius_m / 2.
    """
    if not (math.isfinite(half_length_m) and half_length_m >= 0.0):
        raise ValueError(
            f"half_length_m must be nonnegative and finite, got {half_length_m}"
        )
    if not (math.isfinite(radius_m) and radius_m > 0.0):
        raise ValueError(f"radius_m must be positive and finite, got {radius_m}")
    core_len = 2.0 * half_length_m
    return max(2, int(round(core_len / (0.5 * radius_m))) + 1)


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

    pads: PadContactModel | None
    mesh_dir: Path
    foundation: ElasticFoundationParameters | None = None
    mesh_segments: int = 96
    ring_pitch_m: float = 1.0e-3
    topology_model: TopologyContactModel | None = None
    site_foundations: Mapping[tuple[float, int], ElasticFoundationParameters] | None = (
        None
    )

    def __post_init__(self) -> None:
        if self.pads is None and self.topology_model is None:
            raise ValueError("either pads or topology_model must be provided")
        mesh_dir = Path(self.mesh_dir)
        if not mesh_dir.is_absolute() or not mesh_dir.is_dir():
            raise ValueError(
                f"mesh_dir must be an existing absolute directory, got {mesh_dir}"
            )
        object.__setattr__(self, "mesh_dir", mesh_dir)
        if self.foundation is None:
            if self.pads is not None:
                object.__setattr__(
                    self, "foundation", elastic_foundation_parameters(self.pads)
                )
            elif self.topology_model is not None:
                top = self.topology_model
                ef_def = elastic_foundation_for_law(
                    top.law,
                    top.cylinder.radius_m,
                    PAD_RADIUS_M,
                    DEFAULT_TOPOLOGY_PRELOAD_M,
                )
                object.__setattr__(self, "foundation", ef_def)

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
        """Axial origin of ``side``'s pad cylinder."""
        if self.topology_model is not None:
            top = self.topology_model
            return top.cylinder.origin_axial_m(side)
        assert self.pads is not None
        pads = self.pads
        return pads.cylinder.origin_axial_m(side)

    def mesh_path(self, side: str) -> Path:
        """OBJ path of ``side``'s grip mesh."""
        return self.mesh_dir / f"grip_mesh_{side}.obj"

    def capsule_mesh_path(self, pair_name: str) -> Path:
        """OBJ path of a capsule pair's ring-aligned grip mesh."""
        if not isinstance(pair_name, str) or not pair_name.strip():
            raise ValueError("pair_name must be a non-empty string")
        return self.mesh_dir / f"grip_mesh_{pair_name}.obj"

    def site_foundation(
        self, radius_m: float, count: int = 1
    ) -> ElasticFoundationParameters:
        """Return the calibrated foundation for ``(radius_m, count)``, or ``self.ef``."""
        if self.site_foundations and (radius_m, count) in self.site_foundations:
            return self.site_foundations[(radius_m, count)]
        return self.ef


def _grip_axis(interface: GripInterface, cfg: ContactGripConfig) -> tuple:
    rot = np.asarray(interface.right.rotation, dtype=float)
    point = np.asarray(interface.right.position_m) + rot @ (
        cfg.layout.axis_offset_grip_frame("R")
    )
    return point, rot[:, 0], rot[:, 1]


def _write_capsule_meshes(
    point: np.ndarray,
    axis: np.ndarray,
    radial: np.ndarray,
    cfg: ContactGripConfig,
    facts: dict[str, Any],
) -> None:
    """Write ring-aligned grip cylinder meshes centered on each capsule site's axial midpoint."""
    assert cfg.topology_model is not None
    topo = cfg.topology_model.topology
    grip_r = cfg.layout.grip_radius_m
    for side in SIDES:
        centre = cfg.origin_axial_m(side)
        for pair in cfg.topology_model.physical_grip_pairs(side):
            site = topo.site(pair.site_a)
            mapped = map_site_to_hand_frame(topo, site, side)
            if mapped.primitive != "capsule":
                continue
            cap_centre = centre + mapped.centre_m[0]
            half_cap = (
                math.ceil(
                    (mapped.half_length_m + mapped.radius_m + MESH_MARGIN_M)
                    / cfg.ring_pitch_m
                    - 1e-12
                )
                * cfg.ring_pitch_m
            )
            cap_path = cfg.capsule_mesh_path(pair.name)
            verts, faces = capped_cylinder_mesh(
                grip_r,
                point,
                axis,
                radial,
                (cap_centre - half_cap, cap_centre + half_cap),
                cfg.mesh_segments,
                cfg.ring_pitch_m,
            )
            write_obj(cap_path, verts, faces)
            facts[pair.name] = {
                "file": str(cap_path),
                "vertices": int(len(verts)),
                "faces": int(len(faces)),
                "axial_range_m": [cap_centre - half_cap, cap_centre + half_cap],
            }


def write_grip_meshes(interface: GripInterface, cfg: ContactGripConfig) -> dict:
    """Write the closed grip meshes (club-body frame); return their facts."""
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
    if cfg.topology_model is not None:
        _write_capsule_meshes(point, axis, radial, cfg, facts)
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


def shift_topology_site_inward(
    pos_hand_m: Sequence[float] | np.ndarray,
    side: str,
    grip_radius_m: float,
    extra_penetration_m: float,
) -> np.ndarray:
    """Shift a hand-frame site centre radially toward the grip axis by ``extra_penetration_m``."""
    pos = np.asarray(pos_hand_m, dtype=float)
    if extra_penetration_m == 0.0:
        return pos
    z_axis = -grip_radius_m if side == "L" else grip_radius_m
    radial = np.array([0.0, pos[1], pos[2] - z_axis], dtype=float)
    norm_rad = float(np.linalg.norm(radial))
    if norm_rad <= 1e-12:
        return pos
    return pos - extra_penetration_m * (radial / norm_rad)


def _build_topology_contact(
    geometry: ET.Element,
    forces: ET.Element,
    cfg: ContactGripConfig,
    frames: Mapping[str, Any],
    body_names: Mapping[str, str],
    club: str,
) -> dict[str, Any]:
    assert cfg.topology_model is not None
    top = cfg.topology_model
    topo = top.topology
    grip_r = top.cylinder.radius_m
    ef = cfg.ef
    for side in SIDES:
        _contact_mesh(geometry, f"grip_mesh_{side}", club, cfg.mesh_path(side))
        hand = body_names[frames[side]["hand_body"]]
        hand_frame = np.asarray(frames[side]["hand_frame"], dtype=float)
        pairs = cfg.topology_model.physical_grip_pairs(side)
        for pair in pairs:
            site = topo.site(pair.site_a)
            mapped = map_site_to_hand_frame(topo, site, side)
            if mapped.primitive == "sphere":
                ef_site = (
                    cfg.site_foundations[(mapped.radius_m, 1)]
                    if cfg.site_foundations
                    else ef
                )
                extra = ef_site.preload_penetration_m - DEFAULT_TOPOLOGY_PRELOAD_M
                c_pos = shift_topology_site_inward(
                    np.asarray(mapped.centre_m, dtype=float), side, grip_r, extra
                )
                location = hand_frame[:3, :3] @ c_pos + hand_frame[:3, 3]
                _contact_sphere(
                    geometry,
                    f"grip_pad_{pair.name}",
                    hand,
                    location,
                    mapped.radius_m,
                )
                _foundation_force(
                    forces,
                    f"{CONTACT_FORCE_PREFIX}_{pair.name}",
                    (f"grip_pad_{pair.name}", f"grip_mesh_{side}"),
                    ef_site,
                )
            elif mapped.primitive == "capsule":
                cap_mesh_name = f"grip_mesh_{pair.name}"
                _contact_mesh(
                    geometry, cap_mesh_name, club, cfg.capsule_mesh_path(pair.name)
                )
                p0, p1 = mapped.endpoints_m
                count = capsule_sphere_count(mapped.half_length_m, mapped.radius_m)
                ef_site = (
                    cfg.site_foundations[(mapped.radius_m, count)]
                    if cfg.site_foundations
                    else ef
                )
                extra = ef_site.preload_penetration_m - DEFAULT_TOPOLOGY_PRELOAD_M
                for k in range(count):
                    t = k / (count - 1)
                    c_k = np.asarray(p0) + t * (np.asarray(p1) - np.asarray(p0))
                    c_pos = shift_topology_site_inward(c_k, side, grip_r, extra)
                    location = hand_frame[:3, :3] @ c_pos + hand_frame[:3, 3]
                    _contact_sphere(
                        geometry,
                        f"grip_pad_{pair.name}_{k}",
                        hand,
                        location,
                        mapped.radius_m,
                    )
                    _foundation_force(
                        forces,
                        f"{CONTACT_FORCE_PREFIX}_{pair.name}_{k}",
                        (f"grip_pad_{pair.name}_{k}", cap_mesh_name),
                        ef_site,
                    )
    return {
        "model_type": "topology",
        "pairs_per_hand": {
            s: len(cfg.topology_model.physical_grip_pairs(s)) for s in SIDES
        },
    }


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
    if cfg.topology_model is not None:
        receipt = _build_topology_contact(
            geometry, forces, cfg, frames, body_names, club
        )
        receipt["meshes"] = meshes
        return receipt

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
