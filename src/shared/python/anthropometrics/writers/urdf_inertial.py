"""Writer for URDF ``<inertial>`` blocks.

Produces an ``xml.etree.ElementTree.Element`` of the form::

    <inertial>
        <origin xyz="cx cy cz"/>
        <mass value="m"/>
        <inertia ixx="..." ixy="..." ixz="..." iyy="..." iyz="..." izz="..."/>
    </inertial>

This is the inverse of
:func:`anthropometrics.readers.urdf_inertial.read_urdf_inertial`. The
two are guaranteed to round-trip exactly for any
:class:`SegmentProperties` instance whose inertia is expressed at its
centre of mass (``rtol=1e-9, atol=1e-12``).

URDF inertia is by convention expressed at the link's centre of mass,
which matches the :class:`SegmentProperties` convention — so no
parallel-axis transform is performed here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ..segment_properties import SegmentProperties

# repr(float(x)) gives Python's shortest round-trip representation,
# which is exactly what we need for a lossless write -> read cycle.
_FLOAT_FMT = repr


def write_urdf_inertial(props: SegmentProperties) -> ET.Element:
    """Return a URDF ``<inertial>`` element describing *props*.

    Args:
        props: Segment whose inertial properties to serialise. The
            inertia tensor must already be expressed at the centre of
            mass (the :class:`SegmentProperties` convention).

    Returns:
        A new ``xml.etree.ElementTree.Element`` rooted at ``<inertial>``,
        containing ``<origin>``, ``<mass>``, and ``<inertia>`` children.

    Raises:
        TypeError: When *props* is not a :class:`SegmentProperties`.
        ValueError: When *props.mass_kg* is not strictly positive. (In
            practice :class:`SegmentProperties` rejects this at
            construction; the explicit check here documents the
            invariant for callers reading the writer in isolation.)
    """
    if not isinstance(props, SegmentProperties):
        raise TypeError(
            f"props must be a SegmentProperties, got {type(props).__name__}"
        )
    if props.mass_kg <= 0.0:
        raise ValueError(
            f"mass_kg must be positive to write a URDF <mass>, got {props.mass_kg!r}"
        )

    inertial = ET.Element("inertial")

    cx, cy, cz = (float(v) for v in props.com_xyz_m.tolist())
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": f"{_FLOAT_FMT(cx)} {_FLOAT_FMT(cy)} {_FLOAT_FMT(cz)}"},
    )

    ET.SubElement(inertial, "mass", {"value": _FLOAT_FMT(float(props.mass_kg))})

    tensor = props.inertia_tensor
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _FLOAT_FMT(float(tensor[0, 0])),
            "ixy": _FLOAT_FMT(float(tensor[0, 1])),
            "ixz": _FLOAT_FMT(float(tensor[0, 2])),
            "iyy": _FLOAT_FMT(float(tensor[1, 1])),
            "iyz": _FLOAT_FMT(float(tensor[1, 2])),
            "izz": _FLOAT_FMT(float(tensor[2, 2])),
        },
    )
    return inertial
