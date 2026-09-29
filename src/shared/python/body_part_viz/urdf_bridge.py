"""Bridge between :mod:`body_part_viz` shapes and URDF ``<visual>`` elements.

This module gives URDF generators the same shape vocabulary the C3D Viewer
and matcher already use. A user who imports a custom mesh into the viewer
can re-use it as a URDF visual link without re-modelling.

Mapping (forward):

* :class:`LineShape`        -> :class:`ValueError` (URDF cannot render lines).
* :class:`CylinderShape`    -> ``<cylinder length=L radius=R>``.
* :class:`EllipsoidShape`   -> ``<mesh>`` referencing a generated icosphere
  (logical filename ``__bpv_ellipsoid__a_b_c.obj``) with isotropic
  ``scale="1 1 1"``; the three semi-axes are encoded in the filename.
* :class:`CapsuleShape`     -> ``<mesh>`` (logical filename
  ``__bpv_capsule__length_radius.obj``); URDF has no native capsule.
* :class:`MeshShape`        -> ``<mesh filename="package://body_part_viz/<stem>.<ext>">``.
* :class:`CompositeShape`   -> ``list[Element]`` (caller wraps in a ``<link>``).

The inverse :func:`urdf_to_shape` recovers the original shape (within
``1e-9``) for the supported kinds.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405 — generating, not parsing untrusted input
from collections.abc import Callable
from pathlib import Path

from .contracts import BodyPartShape
from .shapes import (
    CapsuleShape,
    CompositeShape,
    CylinderShape,
    EllipsoidShape,
    LineShape,
    MeshShape,
)

__all__ = [
    "DEFAULT_PACKAGE",
    "shape_to_urdf_visual",
    "urdf_to_shape",
]

DEFAULT_PACKAGE = "body_part_viz"
"""Default URDF ``package://`` name for :class:`MeshShape` references."""

_ELLIPSOID_PREFIX = "__bpv_ellipsoid__"
_CAPSULE_PREFIX = "__bpv_capsule__"
_ELLIPSOID_EXT = ".obj"
_CAPSULE_EXT = ".obj"


def _add_origin(
    parent: ET.Element,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> None:
    if all(v == 0.0 for v in xyz) and all(v == 0.0 for v in rpy):
        return
    ET.SubElement(
        parent,
        "origin",
        xyz=f"{xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}",
        rpy=f"{rpy[0]:.17g} {rpy[1]:.17g} {rpy[2]:.17g}",
    )


def _build_visual(
    geometry_child: ET.Element,
    rest_origin_xyz: tuple[float, float, float],
    rest_origin_rpy: tuple[float, float, float],
) -> ET.Element:
    visual = ET.Element("visual")
    _add_origin(visual, rest_origin_xyz, rest_origin_rpy)
    geometry = ET.SubElement(visual, "geometry")
    geometry.append(geometry_child)
    return visual


def _validate_origin(
    name: str, value: tuple[float, float, float]
) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a length-3 tuple; got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def shape_to_urdf_visual(
    shape: BodyPartShape,
    *,
    rest_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rest_origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    package_name: str = DEFAULT_PACKAGE,
) -> ET.Element | list[ET.Element]:
    """Translate a :class:`BodyPartShape` to a URDF ``<visual>`` element.

    Parameters
    ----------
    shape:
        Shape to translate. Must satisfy the
        :class:`~body_part_viz.contracts.BodyPartShape` Protocol.
    rest_origin_xyz, rest_origin_rpy:
        Optional rest-pose origin (added as ``<origin>`` when non-zero).
    package_name:
        ROS package name for ``package://`` URIs (mesh shapes only).

    Returns
    -------
    ``ET.Element`` for atomic shapes; ``list[ET.Element]`` for
    :class:`CompositeShape`. The caller is expected to attach the
    ``<visual>`` element(s) to its ``<link>``.

    Raises
    ------
    ValueError
        If ``shape`` is a :class:`LineShape` (URDF cannot render lines).
    TypeError
        If ``shape`` is not one of the supported kinds.
    """
    if shape is None:
        raise TypeError("shape must not be None")
    if not isinstance(package_name, str) or not package_name:
        raise ValueError(
            f"package_name must be a non-empty string; got {package_name!r}"
        )
    rest_origin_xyz = _validate_origin("rest_origin_xyz", rest_origin_xyz)
    rest_origin_rpy = _validate_origin("rest_origin_rpy", rest_origin_rpy)

    if isinstance(shape, LineShape):
        raise ValueError("URDF cannot render line visuals; use cylinder")

    if isinstance(shape, CylinderShape):
        length, radius = shape.rest_dimensions
        cyl = ET.Element(
            "cylinder",
            length=f"{float(length):.17g}",
            radius=f"{float(radius):.17g}",
        )
        return _build_visual(cyl, rest_origin_xyz, rest_origin_rpy)

    if isinstance(shape, EllipsoidShape):
        a, b, c = shape.rest_dimensions
        filename = (
            f"package://{package_name}/{_ELLIPSOID_PREFIX}"
            f"{float(a):.17g}_{float(b):.17g}_{float(c):.17g}{_ELLIPSOID_EXT}"
        )
        mesh = ET.Element(
            "mesh",
            filename=filename,
            scale="1 1 1",
        )
        return _build_visual(mesh, rest_origin_xyz, rest_origin_rpy)

    if isinstance(shape, CapsuleShape):
        length, radius = shape.rest_dimensions
        filename = (
            f"package://{package_name}/{_CAPSULE_PREFIX}"
            f"{float(length):.17g}_{float(radius):.17g}{_CAPSULE_EXT}"
        )
        mesh = ET.Element(
            "mesh",
            filename=filename,
            scale="1 1 1",
        )
        return _build_visual(mesh, rest_origin_xyz, rest_origin_rpy)

    if isinstance(shape, MeshShape):
        if shape.source_path is None:
            raise ValueError(
                "MeshShape must have a source_path to produce a URDF "
                "package:// reference"
            )
        filename = f"package://{package_name}/{shape.source_path.name}"
        mesh = ET.Element(
            "mesh",
            filename=filename,
            scale="1 1 1",
        )
        return _build_visual(mesh, rest_origin_xyz, rest_origin_rpy)

    if isinstance(shape, CompositeShape):
        out: list[ET.Element] = []
        for child_shape, transform in shape.children:
            child_xyz = (
                float(transform[0, 3]),
                float(transform[1, 3]),
                float(transform[2, 3]),
            )
            child_rpy = _rotation_matrix_to_rpy(transform[:3, :3])
            child_visual = shape_to_urdf_visual(
                child_shape,
                rest_origin_xyz=child_xyz,
                rest_origin_rpy=child_rpy,
                package_name=package_name,
            )
            if isinstance(child_visual, list):
                out.extend(child_visual)
            else:
                out.append(child_visual)
        return out

    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")


def urdf_to_shape(
    visual_element: ET.Element,
    asset_resolver: Callable[[str], Path],
) -> BodyPartShape:
    """Inverse mapping: URDF ``<visual>`` element back to a shape.

    Parameters
    ----------
    visual_element:
        A ``<visual>`` element produced by :func:`shape_to_urdf_visual`
        (or compatible URDF input).
    asset_resolver:
        Callable that maps a ``package://`` URI to a filesystem
        :class:`pathlib.Path` for :class:`MeshShape` reconstruction.

    Returns
    -------
    The recovered :class:`BodyPartShape`. For round-trip pairs produced
    by :func:`shape_to_urdf_visual`, identity is preserved within
    ``1e-9`` numerical tolerance.

    Raises
    ------
    ValueError
        If the element cannot be interpreted.
    """
    if visual_element is None:
        raise ValueError("visual_element must not be None")
    if visual_element.tag != "visual":
        raise ValueError(
            f"visual_element.tag must be 'visual'; got {visual_element.tag!r}"
        )
    geometry = visual_element.find("geometry")
    if geometry is None:
        raise ValueError("visual_element must contain a <geometry> child")

    children = list(geometry)
    if len(children) != 1:
        raise ValueError(
            "URDF <geometry> must contain exactly one child element; "
            f"got {len(children)}"
        )
    geom = children[0]

    if geom.tag == "cylinder":
        length = float(geom.attrib["length"])
        radius = float(geom.attrib["radius"])
        return CylinderShape(length=length, radius=radius)

    if geom.tag == "mesh":
        filename = geom.attrib.get("filename", "")
        stem = _filename_stem(filename)
        if stem.startswith(_ELLIPSOID_PREFIX):
            payload = stem[len(_ELLIPSOID_PREFIX) :]
            parts = payload.split("_")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed ellipsoid filename: {filename!r} (expected "
                    f"three underscore-separated semi-axes)"
                )
            a, b, c = (float(p) for p in parts)
            return EllipsoidShape(a=a, b=b, c=c)
        if stem.startswith(_CAPSULE_PREFIX):
            payload = stem[len(_CAPSULE_PREFIX) :]
            parts = payload.split("_")
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed capsule filename: {filename!r} (expected "
                    f"two underscore-separated dimensions)"
                )
            length, radius = (float(p) for p in parts)
            return CapsuleShape(length=length, radius=radius)
        # Treat as a real on-disk mesh.
        path = asset_resolver(filename)
        return MeshShape.load(path)  # type: ignore[return-value]

    raise ValueError(f"Unsupported URDF geometry tag: {geom.tag!r}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _filename_stem(filename: str) -> str:
    """Return the filename stem from a ``package://`` URI or bare path."""
    if not filename:
        return ""
    tail = filename.rsplit("/", 1)[-1]
    return tail.rsplit(".", 1)[0]


def _rotation_matrix_to_rpy(rot: object) -> tuple[float, float, float]:
    """Decompose a 3x3 rotation matrix into URDF roll-pitch-yaw (XYZ)."""
    import math

    import numpy as np

    matrix = np.asarray(rot, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3); got {matrix.shape}")
    # URDF rpy is intrinsic XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    sy = -float(matrix[2, 0])
    if abs(sy) >= 1.0 - 1e-9:
        # Gimbal lock: choose roll = 0.
        pitch = math.copysign(math.pi / 2.0, sy)
        roll = 0.0
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
    else:
        pitch = math.asin(sy)
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return (roll, pitch, yaw)
