"""Rust-backed URDF parsing facade.

This module is the thin Python adapter for the `upstream_urdf` Rust crate
introduced under tracking issue
[#5215](https://github.com/D-sorganization/UpstreamDrift/issues/5215).

Design contract
---------------
- **Optional / opt-in**: importing this module is *safe* whether or not the
  Rust extension is installed. ``HAVE_RUST`` advertises availability.
- **No API change**: the consumer (``URDFParser.parse``) keeps its existing
  public signature and dataclass return type. This facade only converts
  the Rust AST (delivered as JSON) into the same ``ParsedModel`` /
  ``Link`` / ``Joint`` instances that the pure-Python parser produces.
- **Routing**: callers gate via ``should_use_rust()``. Today that defaults
  to *off*; flip ``UPSTREAM_URDF_USE_RUST=1`` to opt in. Once parity has
  been monitored in CI we will reverse the default in a follow-up.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import-time guard
    import upstream_urdf as _rust  # type: ignore[import-not-found]

    HAVE_RUST = True
except ImportError:  # pragma: no cover - exercised only when the wheel is absent
    _rust = None  # type: ignore[assignment]
    HAVE_RUST = False


def should_use_rust() -> bool:
    """Return True when the caller has opted in to the Rust parser.

    Opt-in via the ``UPSTREAM_URDF_USE_RUST`` env var. Defaults to off so
    that the migration is reversible per release.
    """
    if not HAVE_RUST:
        return False
    return os.environ.get("UPSTREAM_URDF_USE_RUST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_urdf_to_dict(xml: str) -> dict[str, Any]:
    """Parse a URDF string via the Rust crate, return the AST as a dict.

    Raises:
        RuntimeError: if the Rust extension is not importable.
    """
    if not HAVE_RUST:
        raise RuntimeError("upstream_urdf Rust extension is not installed")
    return json.loads(_rust.parse_urdf(xml))


def parsed_model_from_rust_ast(
    ast: dict[str, Any],
    *,
    source_path: Path | None = None,
    original_xml: str | None = None,
    read_only: bool = False,
) -> Any:
    """Lift a Rust AST dict into the historical ``ParsedModel`` dataclass.

    The mapping is deliberately a 1:1 of fields the pure-Python parser
    populated. The Rust crate already enforces the URDF schema; this
    function never raises on well-formed inputs.
    """
    # Local import to keep the module side-effect free at import time and to
    # avoid a circular dependency with ``urdf_parser`` (which imports this
    # module).
    from model_generation.core.types import (
        Geometry,
        GeometryType,
        Inertia,
        Joint,
        JointDynamics,
        JointLimits,
        JointType,
        Link,
        Material,
        Origin,
    )
    from model_generation.converters.urdf_parser import ParsedModel

    materials: dict[str, Material] = {}
    for mat in ast.get("materials") or []:
        materials[mat["name"]] = _material_from_dict(mat, Material)

    links: list[Link] = []
    for link_ast in ast.get("links") or []:
        links.append(
            _link_from_dict(
                link_ast,
                Link=Link,
                Inertia=Inertia,
                Origin=Origin,
                Geometry=Geometry,
                GeometryType=GeometryType,
                Material=Material,
                materials=materials,
            )
        )

    joints: list[Joint] = []
    for joint_ast in ast.get("joints") or []:
        joints.append(
            _joint_from_dict(
                joint_ast,
                Joint=Joint,
                JointType=JointType,
                JointLimits=JointLimits,
                JointDynamics=JointDynamics,
                Origin=Origin,
            )
        )

    return ParsedModel(
        name=ast.get("name", "unnamed_robot"),
        links=links,
        joints=joints,
        materials=materials,
        original_xml=original_xml,
        source_path=source_path,
        warnings=[],
        read_only=read_only,
    )


def _origin_from_dict(d: dict[str, Any] | None, Origin: Any) -> Any:
    if not d:
        return Origin()
    return Origin(
        xyz=tuple(d.get("xyz", (0.0, 0.0, 0.0))),
        rpy=tuple(d.get("rpy", (0.0, 0.0, 0.0))),
    )


def _material_from_dict(d: dict[str, Any], Material: Any) -> Any:
    color = d.get("color") or (0.8, 0.8, 0.8, 1.0)
    return Material(
        name=d["name"],
        color=tuple(color),
        texture=d.get("texture"),
    )


def _geometry_from_dict(
    d: dict[str, Any] | None,
    Geometry: Any,
    GeometryType: Any,
) -> Any | None:
    if d is None:
        return None
    kind = d.get("kind")
    params = d.get("params") or {}
    if kind == "box":
        return Geometry(
            geometry_type=GeometryType.BOX,
            dimensions=tuple(params.get("size", (0.1, 0.1, 0.1))),
        )
    if kind == "cylinder":
        return Geometry(
            geometry_type=GeometryType.CYLINDER,
            dimensions=(params.get("radius", 0.05), params.get("length", 0.1)),
        )
    if kind == "sphere":
        return Geometry(
            geometry_type=GeometryType.SPHERE,
            dimensions=(params.get("radius", 0.05),),
        )
    if kind == "mesh":
        return Geometry(
            geometry_type=GeometryType.MESH,
            mesh_filename=params.get("filename", ""),
            mesh_scale=tuple(params.get("scale", (1.0, 1.0, 1.0))),
        )
    return None


def _link_from_dict(
    d: dict[str, Any],
    *,
    Link: Any,
    Inertia: Any,
    Origin: Any,
    Geometry: Any,
    GeometryType: Any,
    Material: Any,
    materials: dict[str, Any],
) -> Any:
    # Inertia
    inert_ast = d.get("inertial")
    if inert_ast is not None:
        com = tuple(inert_ast.get("origin", {}).get("xyz", (0.0, 0.0, 0.0)))
        inertia = Inertia(
            ixx=inert_ast.get("ixx", 0.1),
            iyy=inert_ast.get("iyy", 0.1),
            izz=inert_ast.get("izz", 0.1),
            ixy=inert_ast.get("ixy", 0.0),
            ixz=inert_ast.get("ixz", 0.0),
            iyz=inert_ast.get("iyz", 0.0),
            mass=inert_ast.get("mass", 1.0),
            center_of_mass=com,
        )
    else:
        inertia = Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)

    # The historical dataclass keeps only the first visual / collision.
    visuals = d.get("visuals") or []
    collisions = d.get("collisions") or []
    visual = visuals[0] if visuals else None
    collision = collisions[0] if collisions else None

    visual_geometry = (
        _geometry_from_dict(visual.get("geometry"), Geometry, GeometryType)
        if visual
        else None
    )
    visual_origin = _origin_from_dict(visual.get("origin") if visual else None, Origin)
    visual_material = None
    if visual and visual.get("material"):
        mat_dict = visual["material"]
        # Match the pure-Python behaviour: prefer the robot-level definition
        # when names collide.
        visual_material = materials.get(
            mat_dict.get("name", ""), _material_from_dict(mat_dict, Material)
        )

    collision_geometry = (
        _geometry_from_dict(collision.get("geometry"), Geometry, GeometryType)
        if collision
        else None
    )
    collision_origin = _origin_from_dict(
        collision.get("origin") if collision else None, Origin
    )

    return Link(
        name=d["name"],
        inertia=inertia,
        visual_geometry=visual_geometry,
        visual_origin=visual_origin,
        visual_material=visual_material,
        collision_geometry=collision_geometry,
        collision_origin=collision_origin,
    )


def _joint_from_dict(
    d: dict[str, Any],
    *,
    Joint: Any,
    JointType: Any,
    JointLimits: Any,
    JointDynamics: Any,
    Origin: Any,
) -> Any:
    try:
        kind = JointType(d.get("type", "fixed"))
    except ValueError:
        kind = JointType.FIXED

    limits_ast = d.get("limits")
    limits = (
        JointLimits(
            lower=limits_ast.get("lower"),
            upper=limits_ast.get("upper"),
            effort=limits_ast.get("effort"),
            velocity=limits_ast.get("velocity"),
        )
        if limits_ast is not None
        else None
    )

    dyn_ast = d.get("dynamics") or {}
    dynamics = JointDynamics(
        damping=dyn_ast.get("damping", 0.5),
        friction=dyn_ast.get("friction", 0.0),
    )

    return Joint(
        name=d["name"],
        joint_type=kind,
        parent=d["parent"],
        child=d["child"],
        origin=_origin_from_dict(d.get("origin"), Origin),
        axis=tuple(d.get("axis", (0.0, 0.0, 1.0))),
        limits=limits,
        dynamics=dynamics,
    )
