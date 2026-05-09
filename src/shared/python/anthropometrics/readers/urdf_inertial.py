"""Reader for URDF ``<inertial>`` blocks.

URDF (the Unified Robot Description Format) expresses a link's inertial
properties in a single ``<inertial>`` element with three children:

* ``<origin xyz="x y z" rpy="..."/>`` — pose of the link's centre of
  mass relative to the link frame. Only ``xyz`` is consumed here; ``rpy``
  is ignored because :class:`SegmentProperties` stores the inertia
  tensor already expressed at the centre of mass in the link frame.
* ``<mass value="m"/>`` — segment mass in kilograms.
* ``<inertia ixx="..." ixy="..." ixz="..." iyy="..." iyz="..." izz="..."/>``
  — the six independent components of the symmetric 3x3 inertia tensor
  about the centre of mass.

The reader is the inverse of
:func:`anthropometrics.writers.urdf_inertial.write_urdf_inertial`. The
two are guaranteed to round-trip exactly for any
:class:`SegmentProperties` instance whose inertia is expressed at its
centre of mass (``rtol=1e-9, atol=1e-12``).

Because URDF carries no canonical anthropometric metadata (no segment
name, source method, subject height, etc.), the caller must provide
those fields via keyword arguments. Defaults are deliberately neutral
sentinels so that :class:`SegmentProperties` invariants still hold.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ..segment_properties import SegmentProperties


def read_urdf_inertial(
    elem: ET.Element,
    *,
    name: str = "urdf_link",
    body_part_id: str = "urdf_link",
    length_m: float = 1.0,
    proximal_marker: str | None = None,
    distal_marker: str | None = None,
    source_method: str = "urdf_inertial",
    source_subject_height_m: float = 1.0,
    source_subject_mass_kg: float = 1.0,
) -> SegmentProperties:
    """Parse an ``<inertial>`` URDF element into a :class:`SegmentProperties`.

    Args:
        elem: The ``<inertial>`` element to parse. May be the element
            itself or a parent containing exactly one ``<inertial>``
            child.
        name: Canonical segment name to attach to the result. URDF does
            not store this — callers typically pass the parent
            ``<link>`` ``name`` attribute.
        body_part_id: Canonical body-part identifier.
        length_m: Segment length in metres. Used only to satisfy the
            :class:`SegmentProperties` invariant ``|com| <= 2 * length_m``.
        proximal_marker: Optional marker label.
        distal_marker: Optional marker label.
        source_method: Provenance string recorded on the result.
        source_subject_height_m: Subject stature in metres.
        source_subject_mass_kg: Subject body mass in kilograms.

    Returns:
        A fully-validated :class:`SegmentProperties` instance.

    Raises:
        ValueError: When the element is malformed (missing ``<mass>``
            or ``<inertia>``, missing required attributes, non-numeric
            values, or values that fail
            :class:`SegmentProperties` invariants — e.g. a non-positive
            mass).
        TypeError: When *elem* is not an ``xml.etree.ElementTree.Element``.
    """
    if not isinstance(elem, ET.Element):
        raise TypeError(
            f"elem must be an xml.etree.ElementTree.Element, got {type(elem).__name__}"
        )

    inertial = _resolve_inertial(elem)

    mass_elem = inertial.find("mass")
    if mass_elem is None:
        raise ValueError("URDF <inertial> is missing required <mass> child")
    mass_kg = _required_float_attr(mass_elem, "value", "<mass>")

    inertia_elem = inertial.find("inertia")
    if inertia_elem is None:
        raise ValueError("URDF <inertial> is missing required <inertia> child")
    tensor = _parse_inertia_tensor(inertia_elem)

    com_xyz = _parse_origin_xyz(inertial.find("origin"))

    if mass_kg <= 0.0:
        raise ValueError(f"URDF <mass value=...> must be positive, got {mass_kg!r}")

    return SegmentProperties(
        name=name,
        body_part_id=body_part_id,
        length_m=length_m,
        proximal_marker=proximal_marker,
        distal_marker=distal_marker,
        mass_kg=mass_kg,
        com_xyz_m=com_xyz,
        inertia_tensor=tensor,
        source_method=source_method,
        source_subject_height_m=source_subject_height_m,
        source_subject_mass_kg=source_subject_mass_kg,
    )


# --------------------------------------------------------------------------- #
# Private helpers.                                                            #
# --------------------------------------------------------------------------- #
def _resolve_inertial(elem: ET.Element) -> ET.Element:
    """Return the ``<inertial>`` element, accepting either it or its parent."""
    if elem.tag == "inertial":
        return elem
    inertial = elem.find("inertial")
    if inertial is None:
        raise ValueError(
            f"expected an <inertial> element (or a parent containing one); "
            f"got <{elem.tag}>"
        )
    return inertial


def _required_float_attr(elem: ET.Element, attr: str, label: str) -> float:
    """Return ``float(elem.attrib[attr])`` or raise a clear ``ValueError``."""
    raw = elem.get(attr)
    if raw is None:
        raise ValueError(f"URDF {label} is missing required attribute '{attr}'")
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"URDF {label} attribute '{attr}' is not a valid float: {raw!r}"
        ) from error


def _parse_inertia_tensor(inertia_elem: ET.Element) -> np.ndarray:
    """Return the symmetric 3x3 tensor encoded in an ``<inertia>`` element."""
    ixx = _required_float_attr(inertia_elem, "ixx", "<inertia>")
    ixy = _required_float_attr(inertia_elem, "ixy", "<inertia>")
    ixz = _required_float_attr(inertia_elem, "ixz", "<inertia>")
    iyy = _required_float_attr(inertia_elem, "iyy", "<inertia>")
    iyz = _required_float_attr(inertia_elem, "iyz", "<inertia>")
    izz = _required_float_attr(inertia_elem, "izz", "<inertia>")
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


def _parse_origin_xyz(origin_elem: ET.Element | None) -> np.ndarray:
    """Return the ``xyz`` vector from an ``<origin>`` element (default zeros).

    URDF treats the entire ``<origin>`` element as optional, defaulting
    to identity. Likewise, an ``<origin>`` without an ``xyz`` attribute
    defaults to ``"0 0 0"``.
    """
    if origin_elem is None:
        return np.zeros(3, dtype=float)
    raw = origin_elem.get("xyz", "0 0 0")
    parts = raw.split()
    if len(parts) != 3:
        raise ValueError(
            f"URDF <origin xyz=...> must have 3 space-separated components, got {raw!r}"
        )
    try:
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError as error:
        raise ValueError(
            f"URDF <origin xyz=...> contains a non-numeric component: {raw!r}"
        ) from error
