"""Export full-body anthropometric model specification to OpenSim 4.0 XML (MS-40 #10339).

Produces complete .osim models from full_body_spec_anthro_driver.json / iron7.json
with 44 coordinates, CustomJoint trees, dual-grip weld closure constraints,
Hunt-Crossley foot contact spheres, coordinate actuators, and tour markers.
Pure-XML generation with zero OpenSim SDK dependency at build time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only; parse via DefusedET

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import precondition

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = {"full-body-v1"}
DEFAULT_CLOSURE_TYPE = "weld"


def _format_numbers(values: Sequence[float] | np.ndarray) -> str:
    """Format floating point numbers consistently."""
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def _validate_spec(spec: Mapping[str, Any]) -> None:
    """Validate top-level full-body specification structure."""
    if spec.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version: {spec.get('schema_version')!r}; "
            f"expected one of {SUPPORTED_SCHEMA_VERSIONS}"
        )
    for required_key in ("bodies", "joints", "coordinate_order", "contact", "closure"):
        if required_key not in spec:
            raise ValueError(f"Missing required key in specification: {required_key!r}")
    if len(spec["coordinate_order"]) != 44:
        raise ValueError(
            f"Expected 44 coordinates, found {len(spec['coordinate_order'])}"
        )


def _validate_rigid_transform(matrix_like: Any) -> np.ndarray:
    """Validate and return a 4x4 homogeneous rigid transformation matrix."""
    mat = np.asarray(matrix_like, dtype=float)
    if (
        mat.shape != (4, 4)
        or not np.isfinite(mat).all()
        or not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0)
        or not np.allclose(mat[:3, :3].T @ mat[:3, :3], np.eye(3), atol=1e-10, rtol=0)
        or not np.isclose(np.linalg.det(mat[:3, :3]), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("Invalid 4x4 rigid transformation matrix")
    return mat


def _rotation_matrix_to_xyz_euler(r_mat: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to body-fixed XYZ Euler angles in radians."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected.*", category=UserWarning
        )
        r = Rotation.from_matrix(r_mat)
        angles = r.as_euler("xyz")
    return float(angles[0]), float(angles[1]), float(angles[2])


def _aggregate_body_inertia(
    body: Mapping[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate composite total mass, center of mass, and inertia about COM."""
    solids = body.get("solids", [])
    if not solids:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    masses: list[float] = []
    coms_in_body: list[np.ndarray] = []
    inertias_in_body: list[np.ndarray] = []

    for solid in solids:
        m = float(solid["mass_kg"])
        if m < 0.0 or not np.isfinite(m):
            raise ValueError(f"Invalid solid mass {m} in body {body.get('name')}")
        if m == 0.0:
            continue
        t_solid = _validate_rigid_transform(solid["placement"])
        com_local = np.asarray(solid["com_m"], dtype=float)
        inertia_local = np.asarray(solid["inertia_com_kg_m2"], dtype=float)

        com_b = t_solid[:3, :3] @ com_local + t_solid[:3, 3]
        inertia_b = t_solid[:3, :3] @ inertia_local @ t_solid[:3, :3].T

        masses.append(m)
        coms_in_body.append(com_b)
        inertias_in_body.append(inertia_b)

    total_mass = sum(masses)
    if total_mass <= 0.0:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    total_com = np.zeros(3)
    for m, c in zip(masses, coms_in_body, strict=True):
        total_com += m * c
    total_com /= total_mass

    total_inertia = np.zeros((3, 3))
    for m, c, inert in zip(masses, coms_in_body, inertias_in_body, strict=True):
        d = c - total_com
        parallel_axis = m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        total_inertia += inert + parallel_axis

    return total_mass, total_com, total_inertia


def _make_physical_offset_frame(
    name: str,
    socket_parent: str,
    translation: tuple[float, float, float] | np.ndarray,
    orientation: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> ET.Element:
    """Create a PhysicalOffsetFrame XML component."""
    frame = ET.Element("PhysicalOffsetFrame", attrib={"name": name})
    fg = ET.SubElement(frame, "FrameGeometry", attrib={"name": "frame_geometry"})
    ET.SubElement(fg, "socket_frame").text = ".."
    ET.SubElement(fg, "scale_factors").text = "0.2 0.2 0.2"
    ET.SubElement(frame, "socket_parent").text = socket_parent
    ET.SubElement(frame, "translation").text = _format_numbers(translation)
    ET.SubElement(frame, "orientation").text = _format_numbers(orientation)
    return frame


def _build_ground(model: ET.Element) -> None:
    """Construct ground reference frame."""
    ground = ET.SubElement(model, "Ground", attrib={"name": "ground"})
    fg = ET.SubElement(ground, "FrameGeometry", attrib={"name": "frame_geometry"})
    ET.SubElement(fg, "socket_frame").text = ".."
    ET.SubElement(fg, "scale_factors").text = "0.2 0.2 0.2"
    ET.SubElement(ground, "attached_geometry")
    wrap = ET.SubElement(ground, "WrapObjectSet", attrib={"name": "wrapobjectset"})
    ET.SubElement(wrap, "objects")
    ET.SubElement(wrap, "groups")


OPENSIM_BODY_MAP: dict[str, str] = {
    "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso": "Hip",
    "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/UpperTorsoBase": "Torso",
    "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/COMRod": "Spine",
    "GolfSwing3D_Kinetic/Head": "Head",
    "solid_reference:GolfSwing3D_Kinetic/HubtoLS": "LScap",
    "solid_reference:GolfSwing3D_Kinetic/LUpperArm": "LS",
    "solid_reference:GolfSwing3D_Kinetic/Left Elbow Joint/Spherical Solid": "LE",
    "solid_reference:GolfSwing3D_Kinetic/Left Forearm/LLowerForearm": "LF",
    "solid_reference:GolfSwing3D_Kinetic/HubtoRS": "RScap",
    "solid_reference:GolfSwing3D_Kinetic/RUpperArm": "RS",
    "solid_reference:GolfSwing3D_Kinetic/Right Elbow Joint/Spherical Solid1": "RE",
    "solid_reference:GolfSwing3D_Kinetic/Right Forearm/RLowerForearm": "RF",
    "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector": "Clubhead",
    "solid_reference:GolfSwing3D_Kinetic/Grip/RHandStandoff": "Grip",
}


def clean_osim_body_name(name: str) -> str:
    """Map Simscape solid reference path to clean OpenSim body name."""
    return OPENSIM_BODY_MAP.get(name, name)


def clean_osim_joint_name(jname: str, child_name: str) -> str:
    """Construct clean OpenSim joint component name."""
    return f"joint_{clean_osim_body_name(child_name)}"


def _build_bodyset(model: ET.Element, spec: Mapping[str, Any]) -> dict[str, str]:
    """Construct BodySet from specification and return body name mappings."""
    bodyset = ET.SubElement(model, "BodySet", attrib={"name": "bodyset"})
    objects = ET.SubElement(bodyset, "objects")
    body_map: dict[str, str] = {}

    for body_spec in spec["bodies"]:
        raw_bname = body_spec["name"]
        if raw_bname == "world":
            continue

        bname = clean_osim_body_name(raw_bname)
        total_mass, total_com, total_inertia = _aggregate_body_inertia(body_spec)
        body_elem = ET.SubElement(objects, "Body", attrib={"name": bname})

        fg = ET.SubElement(
            body_elem, "FrameGeometry", attrib={"name": "frame_geometry"}
        )
        ET.SubElement(fg, "socket_frame").text = ".."
        ET.SubElement(fg, "scale_factors").text = "0.2 0.2 0.2"
        ET.SubElement(body_elem, "attached_geometry")

        wrap = ET.SubElement(
            body_elem, "WrapObjectSet", attrib={"name": "wrapobjectset"}
        )
        ET.SubElement(wrap, "objects")
        ET.SubElement(wrap, "groups")

        ET.SubElement(body_elem, "mass").text = format(total_mass, ".17g")
        ET.SubElement(body_elem, "mass_center").text = _format_numbers(total_com)

        # In OpenSim: Ixx Iyy Izz Ixy Ixz Iyz
        vec6 = (
            total_inertia[0, 0],
            total_inertia[1, 1],
            total_inertia[2, 2],
            total_inertia[0, 1],
            total_inertia[0, 2],
            total_inertia[1, 2],
        )
        ET.SubElement(body_elem, "inertia").text = _format_numbers(vec6)
        body_map[bname] = f"/bodyset/{bname}"

    return body_map


def _build_transform_axis(
    st: ET.Element,
    prefix: str,
    idx: int,
    axis_vec: str,
    coord_name: str | None,
) -> None:
    """Attach a single rotation or translation TransformAxis to SpatialTransform."""
    axis_elem = ET.SubElement(st, "TransformAxis", attrib={"name": f"{prefix}{idx}"})
    if coord_name is not None:
        ET.SubElement(axis_elem, "coordinates").text = coord_name
        ET.SubElement(axis_elem, "axis").text = axis_vec
        fn = ET.SubElement(axis_elem, "LinearFunction", attrib={"name": "function"})
        ET.SubElement(fn, "coefficients").text = " 1 0"
    else:
        ET.SubElement(axis_elem, "coordinates")
        ET.SubElement(axis_elem, "axis").text = axis_vec
        fn = ET.SubElement(axis_elem, "Constant", attrib={"name": "function"})
        ET.SubElement(fn, "value").text = "0"


def _attach_spatial_transform(
    joint_elem: ET.Element,
    primitives: Sequence[Mapping[str, Any]],
) -> None:
    """Construct 6-axis SpatialTransform matching primitive translations and rotations."""
    st = ET.SubElement(joint_elem, "SpatialTransform")

    rot_by_axis: dict[str, str] = {}
    trans_by_axis: dict[str, str] = {}
    for prim in primitives:
        kind = prim["primitive"]
        cname = prim["coordinate"]
        if kind.startswith("R"):
            rot_by_axis[kind[1].lower()] = cname
        elif kind.startswith("P"):
            trans_by_axis[kind[1].lower()] = cname

    axes = [(1, "x", "1 0 0"), (2, "y", "0 1 0"), (3, "z", "0 0 1")]
    for idx, ax_char, ax_vec in axes:
        _build_transform_axis(st, "rotation", idx, ax_vec, rot_by_axis.get(ax_char))

    for idx, ax_char, ax_vec in axes:
        _build_transform_axis(
            st, "translation", idx, ax_vec, trans_by_axis.get(ax_char)
        )


def _build_jointset(model: ET.Element, spec: Mapping[str, Any]) -> list[str]:
    """Construct JointSet with CustomJoints in document order."""
    jointset = ET.SubElement(model, "JointSet", attrib={"name": "jointset"})
    objects = ET.SubElement(jointset, "objects")
    order = {c: i for i, c in enumerate(spec["coordinate_order"])}
    ordered_joints = sorted(
        spec["joints"],
        key=lambda j: min(order[p["coordinate"]] for p in j["primitives"]),
    )
    all_coords: list[str] = []
    ranges_deg = spec.get("coordinate_ranges_deg", {})

    for joint in ordered_joints:
        raw_jname = joint["name"]
        parent = joint["parent"]
        child = joint["child"]
        jname = clean_osim_joint_name(raw_jname, child)

        joint_elem = ET.SubElement(objects, "CustomJoint", attrib={"name": jname})
        parent_frame_name = f"{jname}_parent_offset"
        child_frame_name = f"{jname}_child_offset"

        ET.SubElement(joint_elem, "socket_parent_frame").text = parent_frame_name
        ET.SubElement(joint_elem, "socket_child_frame").text = child_frame_name

        coords_elem = ET.SubElement(joint_elem, "coordinates")
        for prim in joint["primitives"]:
            cname = prim["coordinate"]
            all_coords.append(cname)
            coord_elem = ET.SubElement(
                coords_elem, "Coordinate", attrib={"name": cname}
            )
            ET.SubElement(coord_elem, "default_value").text = "0"
            ET.SubElement(coord_elem, "default_speed_value").text = "0"

            if cname in ranges_deg:
                min_deg, max_deg = ranges_deg[cname]
                r_min = np.deg2rad(float(min_deg))
                r_max = np.deg2rad(float(max_deg))
            elif prim["primitive"].startswith("P"):
                r_min, r_max = -10.0, 10.0
            else:
                r_min, r_max = -np.pi, np.pi

            ET.SubElement(coord_elem, "range").text = f"{r_min:.17g} {r_max:.17g}"
            ET.SubElement(coord_elem, "clamped").text = "false"
            ET.SubElement(coord_elem, "locked").text = "false"
            ET.SubElement(coord_elem, "prescribed_function")
            ET.SubElement(coord_elem, "prescribed").text = "false"

        frames_elem = ET.SubElement(joint_elem, "frames")
        t_parent = _validate_rigid_transform(joint["parent_to_base"])
        t_child = _validate_rigid_transform(joint["child_to_follower"])

        clean_parent = clean_osim_body_name(parent)
        clean_child = clean_osim_body_name(child)
        parent_socket = "/ground" if parent == "world" else f"/bodyset/{clean_parent}"
        child_socket = f"/bodyset/{clean_child}"

        euler_p = _rotation_matrix_to_xyz_euler(t_parent[:3, :3])
        euler_c = _rotation_matrix_to_xyz_euler(t_child[:3, :3])

        frames_elem.append(
            _make_physical_offset_frame(
                parent_frame_name, parent_socket, t_parent[:3, 3], euler_p
            )
        )
        frames_elem.append(
            _make_physical_offset_frame(
                child_frame_name, child_socket, t_child[:3, 3], euler_c
            )
        )

        _attach_spatial_transform(joint_elem, joint["primitives"])

    return all_coords


def _build_constraintset(
    model: ET.Element,
    spec: Mapping[str, Any],
    closure_type: str = DEFAULT_CLOSURE_TYPE,
) -> None:
    """Construct ConstraintSet with dual-grip closure constraints."""
    constraintset = ET.SubElement(
        model, "ConstraintSet", attrib={"name": "constraintset"}
    )
    objects = ET.SubElement(constraintset, "objects")
    ET.SubElement(constraintset, "groups")

    closure = spec["closure"]
    body_a = clean_osim_body_name(closure["body_a"])
    body_b = clean_osim_body_name(closure["body_b"])
    t_a = _validate_rigid_transform(closure["placement_a"])
    t_b = _validate_rigid_transform(closure["placement_b"])

    if closure_type == "point_pair":
        # Two point constraints representing proximal and distal grip contacts
        p1_a = t_a[:3, 3]
        p1_b = t_b[:3, 3]
        # Second point offset along local grip axis (+y or +z)
        offset = np.array([0.0, 0.05, 0.0])
        p2_a = t_a[:3, :3] @ offset + t_a[:3, 3]
        p2_b = t_b[:3, :3] @ offset + t_b[:3, 3]

        for suffix, pa, pb in (("proximal", p1_a, p1_b), ("distal", p2_a, p2_b)):
            pc = ET.SubElement(
                objects, "PointConstraint", attrib={"name": f"two_hand_grip_{suffix}"}
            )
            ET.SubElement(pc, "isEnforced").text = "true"
            ET.SubElement(pc, "socket_body1").text = f"/bodyset/{body_a}"
            ET.SubElement(pc, "socket_body2").text = f"/bodyset/{body_b}"
            ET.SubElement(pc, "point_on_body1").text = _format_numbers(pa)
            ET.SubElement(pc, "point_on_body2").text = _format_numbers(pb)
    else:
        # 6-DOF spatial WeldConstraint
        weld = ET.SubElement(
            objects, "WeldConstraint", attrib={"name": "two_hand_grip_closure"}
        )
        ET.SubElement(weld, "isEnforced").text = "true"
        ET.SubElement(weld, "socket_frame1").text = "closure_frame_a"
        ET.SubElement(weld, "socket_frame2").text = "closure_frame_b"

        frames = ET.SubElement(weld, "frames")
        euler_a = _rotation_matrix_to_xyz_euler(t_a[:3, :3])
        euler_b = _rotation_matrix_to_xyz_euler(t_b[:3, :3])
        frames.append(
            _make_physical_offset_frame(
                "closure_frame_a", f"/bodyset/{body_a}", t_a[:3, 3], euler_a
            )
        )
        frames.append(
            _make_physical_offset_frame(
                "closure_frame_b", f"/bodyset/{body_b}", t_b[:3, 3], euler_b
            )
        )


def _build_contact_geometries(model: ET.Element, spec: Mapping[str, Any]) -> None:
    """Construct ContactGeometrySet containing ground plane and foot contact spheres."""
    cset = ET.SubElement(
        model, "ContactGeometrySet", attrib={"name": "contactgeometryset"}
    )
    objects = ET.SubElement(cset, "objects")
    ET.SubElement(cset, "groups")

    # Rigid ground contact half space (normal along +y)
    hs = ET.SubElement(objects, "ContactHalfSpace", attrib={"name": "ground_plane"})
    ET.SubElement(hs, "socket_frame").text = "/ground"
    ET.SubElement(hs, "location").text = "0 0 0"
    ET.SubElement(hs, "orientation").text = "0 0 -1.57079633"

    for sphere in spec["contact"]["spheres"]:
        sname = sphere["name"]
        bname = clean_osim_body_name(sphere["body"])
        pos = sphere["position_m"]
        radius = float(sphere["radius_m"])

        cs = ET.SubElement(
            objects, "ContactSphere", attrib={"name": f"contact_geom_{sname}"}
        )
        ET.SubElement(cs, "socket_frame").text = f"/bodyset/{bname}"
        ET.SubElement(cs, "location").text = _format_numbers(pos)
        ET.SubElement(cs, "orientation").text = "0 0 0"
        ET.SubElement(cs, "radius").text = format(radius, ".17g")


def _build_forceset(
    model: ET.Element,
    spec: Mapping[str, Any],
    coordinates: Sequence[str],
    actuate_root: bool = False,
) -> None:
    """Construct ForceSet with HuntCrossleyForces and CoordinateActuators."""
    forceset = ET.SubElement(model, "ForceSet", attrib={"name": "forceset"})
    objects = ET.SubElement(forceset, "objects")
    ET.SubElement(forceset, "groups")

    # Hunt-Crossley contact forces
    c_params = spec["contact"]["parameters"]
    stiffness = float(c_params["stiffness_n_m"])
    dissipation = float(c_params["dissipation_s_m"])
    mu_s = float(c_params["static_friction"])
    mu_d = float(c_params["dynamic_friction"])
    mu_v = float(c_params["viscous_friction"])
    v_trans = float(c_params["transition_velocity_m_s"])

    for sphere in spec["contact"]["spheres"]:
        sname = sphere["name"]
        hc = ET.SubElement(
            objects, "HuntCrossleyForce", attrib={"name": f"contact_{sname}"}
        )
        ET.SubElement(hc, "appliesForce").text = "true"

        param_set = ET.SubElement(
            hc,
            "HuntCrossleyForce::ContactParametersSet",
            attrib={"name": "contact_parameters"},
        )
        param_objects = ET.SubElement(param_set, "objects")
        ET.SubElement(param_set, "groups")

        cp = ET.SubElement(param_objects, "HuntCrossleyForce::ContactParameters")
        ET.SubElement(cp, "geometry").text = f"ground_plane contact_geom_{sname}"
        ET.SubElement(cp, "stiffness").text = format(stiffness, ".17g")
        ET.SubElement(cp, "dissipation").text = format(dissipation, ".17g")
        ET.SubElement(cp, "static_friction").text = format(mu_s, ".17g")
        ET.SubElement(cp, "dynamic_friction").text = format(mu_d, ".17g")
        ET.SubElement(cp, "viscous_friction").text = format(mu_v, ".17g")

        ET.SubElement(hc, "transition_velocity").text = format(v_trans, ".17g")

    # Coordinate Actuators
    actuated = coordinates if actuate_root else coordinates[6:]
    for coord in actuated:
        act = ET.SubElement(
            objects, "CoordinateActuator", attrib={"name": f"tau_{coord}"}
        )
        ET.SubElement(act, "appliesForce").text = "true"
        ET.SubElement(act, "min_control").text = "-Inf"
        ET.SubElement(act, "max_control").text = "Inf"
        ET.SubElement(act, "coordinate").text = coord
        ET.SubElement(act, "optimal_force").text = "1"


def _build_markerset(model: ET.Element, spec: Mapping[str, Any]) -> None:
    """Construct MarkerSet containing all tour capture marker offsets."""
    markerset = ET.SubElement(model, "MarkerSet", attrib={"name": "markerset"})
    objects = ET.SubElement(markerset, "objects")
    ET.SubElement(markerset, "groups")

    marker_attachments = spec.get("marker_attachments", {})
    for mname, mdata in marker_attachments.items():
        raw_bname = mdata["body"]
        clean_b = clean_osim_body_name(raw_bname)
        raw_offset = mdata.get("offset_m")
        offset = [0.0, 0.0, 0.0] if raw_offset is None else raw_offset
        marker_elem = ET.SubElement(objects, "Marker", attrib={"name": mname})
        ET.SubElement(marker_elem, "socket_parent_frame").text = f"/bodyset/{clean_b}"
        ET.SubElement(marker_elem, "location").text = _format_numbers(offset)
        ET.SubElement(marker_elem, "fixed").text = "false"


@precondition(
    lambda spec, **kwargs: bool(spec), "spec mapping or bytes must not be empty"
)
def export_full_body_osim(
    spec_or_bytes: Mapping[str, Any] | bytes | str,
    *,
    model_name: str = "full_body_anthro_driver",
    closure_type: str = DEFAULT_CLOSURE_TYPE,
    actuate_root: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Export anthropometric full-body specification to OpenSim 4.0 XML string.

    Parameters
    ----------
    spec_or_bytes : Mapping[str, Any] | bytes | str
        Specification mapping or JSON bytes/str.
    model_name : str
        Target model name in <Model name="...">.
    closure_type : str
        Grip loop-closure formulation ("weld" or "point_pair").
    actuate_root : bool
        Whether to generate CoordinateActuators for root 6-DOF coordinates.

    Returns
    -------
    tuple[str, dict[str, Any]]
        (osim_xml_string, metadata_receipt)
    """
    if isinstance(spec_or_bytes, bytes):
        raw_bytes = spec_or_bytes
        spec = json.loads(spec_or_bytes.decode("utf-8"))
    elif isinstance(spec_or_bytes, str):
        raw_bytes = spec_or_bytes.encode("utf-8")
        spec = json.loads(spec_or_bytes)
    else:
        raw_bytes = json.dumps(spec_or_bytes, sort_keys=True).encode("utf-8")
        spec = dict(spec_or_bytes)

    _validate_spec(spec)

    root = ET.Element("OpenSimDocument", attrib={"Version": "40000"})
    model = ET.SubElement(root, "Model", attrib={"name": model_name})

    _build_ground(model)

    grav = spec.get("gravity_m_s2", [0.0, -9.80665, 0.0])
    ET.SubElement(model, "gravity").text = _format_numbers(grav)
    ET.SubElement(model, "length_units").text = "m"
    ET.SubElement(model, "force_units").text = "N"

    _build_bodyset(model, spec)
    coords = _build_jointset(model, spec)

    # ControllerSet (empty container for OpenSim compliance)
    cset = ET.SubElement(model, "ControllerSet", attrib={"name": "controllerset"})
    ET.SubElement(cset, "objects")
    ET.SubElement(cset, "groups")

    _build_constraintset(model, spec, closure_type=closure_type)
    _build_forceset(model, spec, coords, actuate_root=actuate_root)
    _build_markerset(model, spec)
    _build_contact_geometries(model, spec)

    xml_text = ET.tostring(root, encoding="utf-8").decode("utf-8")
    model_sha = hashlib.sha256(raw_bytes).hexdigest()
    osim_sha = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()

    metadata: dict[str, Any] = {
        "representation": "native-full-body-osim-v1",
        "model_sha256": model_sha,
        "osim_sha256": osim_sha,
        "coordinate_count": len(coords),
        "coordinate_order": coords,
        "body_count": len(spec["bodies"]) - 1,  # minus world
        "marker_count": len(spec.get("marker_attachments", {})),
        "contact_sphere_count": len(spec["contact"]["spheres"]),
        "closure_type": closure_type,
        "actuate_root": actuate_root,
    }

    return xml_text, metadata


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export full-body specification to OpenSim 4.0 XML document"
    )
    parser.add_argument("--spec", required=True, type=Path, help="Input spec JSON path")
    parser.add_argument(
        "--out", required=True, type=Path, help="Output .osim file path"
    )
    parser.add_argument(
        "--receipt", type=Path, default=None, help="Output receipt JSON path"
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="OpenSim model name"
    )
    parser.add_argument(
        "--closure-type",
        type=str,
        choices=["weld", "point_pair"],
        default=DEFAULT_CLOSURE_TYPE,
        help="Closure constraint formulation",
    )
    parser.add_argument(
        "--actuate-root",
        action="store_true",
        help="Generate CoordinateActuators for root floating-base coordinates",
    )

    args = parser.parse_args()
    spec_path = args.spec.resolve()
    if not spec_path.is_file():
        logger.error("Spec file not found: %s", spec_path)
        return 1

    spec_bytes = spec_path.read_bytes()
    name = args.model_name or spec_path.stem

    xml_str, receipt = export_full_body_osim(
        spec_bytes,
        model_name=name,
        closure_type=args.closure_type,
        actuate_root=args.actuate_root,
    )

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_str, encoding="utf-8")
    logger.info("Wrote OpenSim model to %s", out_path)

    if args.receipt:
        rec_path = args.receipt.resolve()
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        logger.info("Wrote export receipt to %s", rec_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
