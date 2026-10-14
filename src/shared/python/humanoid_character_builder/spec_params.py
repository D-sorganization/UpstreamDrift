"""Spec-native character parameters that compile to ``full_body_spec`` (CMB-1, #11652).

``BodyParameters`` drives the mesh/URDF generators; engines consume the
``full-body-v1`` document. This module is the bridge: a small, validated
parameter set (stature, mass, three proportion scales, club, grip roll) that
compiles through the one shared builder
(``motion_matching.execution.spec_builder.compose_anthropometric_document``,
De Leva tables in ``anthropometry``) so the character builder never keeps a
second anthropometric table.

Contract:
  * Preconditions: every value finite and inside the documented ranges;
    ``club`` is a known club id.
  * Postconditions: the compiled document passed ``validate_full_body_spec``;
    ``serialize_spec`` output is byte-identical for equal parameters;
    ``params_from_spec(compile_full_body_spec(p)) == p``.

Limits: the lower limbs and De Leva table are the male Rajagopal/De Leva
references; stature and mass scale them, proportion scales reshape trunk,
arms and shoulders. The legs keep the Rajagopal mass distribution and carry
whatever part of ``mass_kg`` the upper body does not, so the body mass
without the club equals ``mass_kg`` (#12199). Output is unqualified
geometry, like its builder.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)
from src.shared.python.motion_matching.club_models import CLUBS
from src.shared.python.motion_matching.execution import assets
from src.shared.python.motion_matching.execution.spec_builder import (
    compose_anthropometric_document,
)

__all__ = [
    "PARAMETER_RANGES",
    "SpecCharacterParameters",
    "compile_full_body_spec",
    "params_from_spec",
    "serialize_spec",
]

#: Inclusive bounds per numeric field (units in the field name).
PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "stature_m": (1.20, 2.30),
    "mass_kg": (30.0, 200.0),
    "trunk_scale": (0.7, 1.4),
    "arm_scale": (0.7, 1.4),
    "shoulder_scale": (0.7, 1.4),
    "grip_roll_deg": (-180.0, 180.0),
}


@dataclass(frozen=True)
class SpecCharacterParameters:
    """Validated subject parameters for the anthropometric full-body spec."""

    stature_m: float = 1.75
    mass_kg: float = 75.0
    trunk_scale: float = 1.0
    arm_scale: float = 1.0
    shoulder_scale: float = 1.0
    club: str = "driver"
    grip_roll_deg: float = 0.0

    def __post_init__(self) -> None:
        for name, (low, high) in PARAMETER_RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite in [{low}, {high}]: {value}")
            object.__setattr__(self, name, float(value))
        if self.club not in CLUBS:
            raise ValueError(f"club must be one of {sorted(CLUBS)}: {self.club!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpecCharacterParameters:
        """Build from a mapping; unknown keys are rejected, not ignored."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown character parameter(s): {unknown}")
        return cls(**dict(data))

    @classmethod
    def from_body_parameters(cls, body: BodyParameters) -> SpecCharacterParameters:
        """Map the mesh-side ``BodyParameters`` proportions onto spec scales."""
        return cls(
            stature_m=body.height_m,
            mass_kg=body.mass_kg,
            trunk_scale=body.torso_length_factor,
            arm_scale=body.arm_length_factor,
            shoulder_scale=body.shoulder_width_factor,
        )


def compile_full_body_spec(
    params: SpecCharacterParameters,
    *,
    native_path: Path | str | None = None,
    osim_path: Path | str | None = None,
    native_candidate_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compile parameters to a validated ``full-body-v1`` document.

    Reference inputs resolve through ``execution.assets`` (explicit path,
    environment variable, then repository checkout); a missing asset raises
    ``FileNotFoundError``.
    """
    if not isinstance(params, SpecCharacterParameters):
        raise TypeError("params must be SpecCharacterParameters")
    built = compose_anthropometric_document(
        native_path=assets.get_native_geometry_spec(native_path),
        osim_path=assets.get_opensim_model(osim_path),
        native_candidate_path=assets.get_candidate_geometry_spec(native_candidate_path),
        stature_m=params.stature_m,
        mass_kg=params.mass_kg,
        club=params.club,
        trunk_scale=params.trunk_scale,
        arm_scale=params.arm_scale,
        shoulder_scale=params.shoulder_scale,
        grip_roll_deg=params.grip_roll_deg,
        match_subject_mass=True,
    )
    return built.document


def serialize_spec(document: Mapping[str, Any]) -> str:
    """Canonical text form: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def params_from_spec(document: Mapping[str, Any]) -> SpecCharacterParameters:
    """Recover the parameters a spec was compiled from (``subject`` + club)."""
    subject = document.get("subject")
    if not isinstance(subject, Mapping):
        raise ValueError("document has no subject block; not a character spec")
    club = document.get("club")
    club_name = club.get("name") if isinstance(club, Mapping) else None
    try:
        return SpecCharacterParameters(
            stature_m=subject["stature_m"],
            mass_kg=subject["mass_kg"],
            trunk_scale=subject["trunk_scale"],
            arm_scale=subject["arm_scale"],
            shoulder_scale=subject["shoulder_scale"],
            grip_roll_deg=subject["grip_roll_deg"],
            club=club_name or "driver",
        )
    except KeyError as exc:
        raise ValueError(f"subject block missing {exc}") from exc


import defusedxml.ElementTree as DefusedET
import numpy as np
from scipy.spatial.transform import Rotation
from src.shared.python.motion_matching.full_body_spec import order_full_body_joints


def _body_mass(body: Any) -> float:
    return float(sum(s["mass_kg"] for s in body["solids"]))


def _compute_spec_poses(
    doc: dict[str, Any],
) -> tuple[float, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute spec mass, body poses and frame origins at q=0."""
    spec_mass_kg = float(sum(_body_mass(b) for b in doc["bodies"]))
    spec_body_poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    for joint in order_full_body_joints(doc):
        parent = joint["parent"]
        child = joint["child"]
        p_pose = spec_body_poses[parent]
        p_base = np.asarray(joint["parent_to_base"], dtype=float)
        c_follower = np.asarray(joint["child_to_follower"], dtype=float)
        spec_body_poses[child] = p_pose @ p_base @ np.linalg.inv(c_follower)

    spec_frame_origins: dict[str, np.ndarray] = {}
    for frame in doc.get("frames", []):
        b_name = frame["body"]
        placement = np.asarray(frame["placement"], dtype=float)
        f_world = spec_body_poses[b_name] @ placement
        spec_frame_origins[frame["name"]] = f_world[:3, 3]

    return spec_mass_kg, spec_body_poses, spec_frame_origins


def _parse_mjcf_pose(elem: Any) -> np.ndarray:
    T = np.eye(4)
    if "pos" in elem.attrib:
        T[:3, 3] = [float(x) for x in elem.attrib["pos"].split()]
    if "quat" in elem.attrib:
        w, x, y, z = [float(v) for v in elem.attrib["quat"].split()]
        T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return T


def _verify_mjcf(
    spec_bytes: bytes, spec_frame_origins: dict[str, np.ndarray]
) -> tuple[float, float]:
    """Verify MJCF mass conservation and site origins."""
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    mjcf_xml, mjcf_meta = export_full_body_mjcf(spec_bytes)
    mjcf_root = DefusedET.fromstring(mjcf_xml)
    mjcf_mass_kg = float(
        sum(float(i.attrib["mass"]) for i in mjcf_root.iter("inertial"))
    )

    mjcf_site_origins: dict[str, np.ndarray] = {}

    def _traverse(elem: Any, parent_pose: np.ndarray) -> None:
        world_T = parent_pose @ _parse_mjcf_pose(elem)
        for site in elem.findall("site"):
            s_name = site.attrib.get("name")
            if s_name:
                s_world = world_T @ _parse_mjcf_pose(site)
                mjcf_site_origins[s_name] = s_world[:3, 3]
        for child_body in elem.findall("body"):
            _traverse(child_body, world_T)

    worldbody = mjcf_root.find("worldbody")
    if worldbody is not None:
        for b in worldbody.findall("body"):
            _traverse(b, np.eye(4))

    frame_sites = mjcf_meta.get("frame_sites", {})
    mjcf_frame_errors = [
        float(np.linalg.norm(spec_frame_origins[fname] - mjcf_site_origins[sname]))
        for fname, sname in frame_sites.items()
        if fname in spec_frame_origins and sname in mjcf_site_origins
    ]
    mjcf_max_frame_err = max(mjcf_frame_errors) if mjcf_frame_errors else 0.0
    return mjcf_mass_kg, mjcf_max_frame_err


def _verify_urdf(
    spec_bytes: bytes, spec_frame_origins: dict[str, np.ndarray]
) -> tuple[float, float]:
    """Verify URDF mass conservation and frame links."""
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    urdf_xml, urdf_sidecar = export_full_body_urdf(spec_bytes)
    urdf_root = DefusedET.fromstring(urdf_xml)
    urdf_mass_kg = float(
        sum(
            float(m.attrib["value"])
            for link in urdf_root.findall("link")
            for m in link.findall("inertial/mass")
        )
    )

    body_links = urdf_sidecar.get("body_links", {})
    root_link = body_links.get("world", "body_0")
    urdf_link_origins: dict[str, np.ndarray] = {root_link: np.zeros(3)}
    urdf_link_poses: dict[str, np.ndarray] = {root_link: np.eye(4)}

    unresolved_urdf = list(urdf_root.findall("joint"))
    while unresolved_urdf:
        next_unresolved = []
        progress = False
        for j in unresolved_urdf:
            p_link = j.find("parent").attrib["link"]
            c_link = j.find("child").attrib["link"]
            if p_link in urdf_link_poses:
                origin_elem = j.find("origin")
                xyz = (
                    [float(x) for x in origin_elem.attrib.get("xyz", "0 0 0").split()]
                    if origin_elem is not None
                    else [0.0, 0.0, 0.0]
                )
                rpy = (
                    [float(x) for x in origin_elem.attrib.get("rpy", "0 0 0").split()]
                    if origin_elem is not None
                    else [0.0, 0.0, 0.0]
                )
                T = np.eye(4)
                T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                T[:3, 3] = xyz
                world_T = urdf_link_poses[p_link] @ T
                urdf_link_poses[c_link] = world_T
                urdf_link_origins[c_link] = world_T[:3, 3]
                progress = True
            else:
                next_unresolved.append(j)
        if not progress:
            break
        unresolved_urdf = next_unresolved

    urdf_frame_links = urdf_sidecar.get("frame_links", {})
    urdf_frame_errors = [
        float(np.linalg.norm(spec_frame_origins[fname] - urdf_link_origins[flink]))
        for fname, flink in urdf_frame_links.items()
        if fname in spec_frame_origins and flink in urdf_link_origins
    ]
    urdf_max_frame_err = max(urdf_frame_errors) if urdf_frame_errors else 0.0
    return urdf_mass_kg, urdf_max_frame_err


def _verify_osim_joint_offsets(
    unresolved_osim: list[Any],
) -> dict[str, np.ndarray]:
    """Solve OpenSim PhysicalOffsetFrame hierarchy to world transforms."""
    osim_poses: dict[str, np.ndarray] = {"/ground": np.eye(4)}
    while unresolved_osim:
        next_unresolved = []
        progress = False
        for j in unresolved_osim:
            frames = j.findall("frames/PhysicalOffsetFrame")
            if len(frames) >= 2:
                f_p, f_c = frames[0], frames[1]
                sp_p = (
                    f_p.find("socket_parent").text
                    if f_p.find("socket_parent") is not None
                    else ""
                )
                sp_c = (
                    f_c.find("socket_parent").text
                    if f_c.find("socket_parent") is not None
                    else ""
                )
                if sp_p in osim_poses:
                    t_p = np.array(
                        [
                            float(x)
                            for x in (f_p.find("translation").text or "0 0 0").split()
                        ]
                    )
                    o_p = np.array(
                        [
                            float(x)
                            for x in (f_p.find("orientation").text or "0 0 0").split()
                        ]
                    )
                    T_p = np.eye(4)
                    T_p[:3, :3] = Rotation.from_euler("XYZ", o_p).as_matrix()
                    T_p[:3, 3] = t_p

                    t_c = np.array(
                        [
                            float(x)
                            for x in (f_c.find("translation").text or "0 0 0").split()
                        ]
                    )
                    o_c = np.array(
                        [
                            float(x)
                            for x in (f_c.find("orientation").text or "0 0 0").split()
                        ]
                    )
                    T_c = np.eye(4)
                    T_c[:3, :3] = Rotation.from_euler("XYZ", o_c).as_matrix()
                    T_c[:3, 3] = t_c

                    osim_poses[sp_c] = osim_poses[sp_p] @ T_p @ np.linalg.inv(T_c)
                    progress = True
                else:
                    next_unresolved.append(j)
        if not progress:
            break
        unresolved_osim = next_unresolved
    return osim_poses
