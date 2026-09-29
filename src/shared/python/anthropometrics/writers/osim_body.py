"""Writer for OpenSim ``<Body>`` XML elements.

OpenSim's ``<Body>`` schema carries three inertial children:

* ``<mass>kg</mass>``
* ``<mass_center>x y z</mass_center>``
* ``<inertia>Ixx Iyy Izz Ixy Ixz Iyz</inertia>``

Plus a ``name`` attribute on the ``<Body>`` element itself. The
canonical :class:`anthropometrics.SegmentProperties` dataclass
carries additional metadata that OpenSim's schema does not natively
represent (segment length, marker labels, source method, source
subject anthropometry). To support a lossless round-trip those extra
fields are emitted into a single ``<UDMetadata>`` child element with
plain text children. OpenSim's XML reader ignores unknown tags, so
the resulting document remains a valid ``Body`` definition.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._types import FloatArray
    from ..segment_properties import SegmentProperties


_METADATA_TAG = "UDMetadata"


def write_osim_body(props: SegmentProperties) -> ET.Element:
    """Serialise *props* into an OpenSim ``<Body>`` XML element.

    Args:
        props: A canonical :class:`SegmentProperties` instance.

    Returns:
        An ``xml.etree.ElementTree.Element`` rooted at ``<Body>``
        with ``mass``, ``mass_center``, ``inertia``, and
        ``UDMetadata`` children, plus a ``name`` attribute.

    Raises:
        TypeError: When *props* is not a :class:`SegmentProperties`.
    """
    # Local import keeps this module light when only the type stub is needed.
    from ..segment_properties import SegmentProperties

    if not isinstance(props, SegmentProperties):
        raise TypeError(
            "write_osim_body requires a SegmentProperties instance, "
            f"got {type(props).__name__}"
        )

    body = ET.Element("Body", attrib={"name": props.name})

    mass = ET.SubElement(body, "mass")
    mass.text = _format_scalar(props.mass_kg)

    mass_center = ET.SubElement(body, "mass_center")
    mass_center.text = _format_vector(props.com_xyz_m)

    inertia = ET.SubElement(body, "inertia")
    inertia.text = _format_inertia(props.inertia_tensor)

    metadata = ET.SubElement(body, _METADATA_TAG)
    _append_text(metadata, "body_part_id", props.body_part_id)
    _append_text(metadata, "length_m", _format_scalar(props.length_m))
    _append_text(metadata, "source_method", props.source_method)
    _append_text(
        metadata,
        "source_subject_height_m",
        _format_scalar(props.source_subject_height_m),
    )
    _append_text(
        metadata,
        "source_subject_mass_kg",
        _format_scalar(props.source_subject_mass_kg),
    )
    if props.proximal_marker is not None:
        _append_text(metadata, "proximal_marker", props.proximal_marker)
    if props.distal_marker is not None:
        _append_text(metadata, "distal_marker", props.distal_marker)

    return body


# --------------------------------------------------------------------------- #
# Formatting helpers (kept private — matched by the reader's parsers).        #
# --------------------------------------------------------------------------- #
def _format_scalar(value: float) -> str:
    """Return a lossless ``repr``-style decimal for a finite float."""
    return repr(float(value))


def _format_vector(vec: FloatArray) -> str:
    """Return ``"x y z"`` with lossless float repr for a length-3 vector."""
    arr = np.asarray(vec, dtype=float).ravel()
    return " ".join(repr(float(component)) for component in arr)


def _format_inertia(tensor: FloatArray) -> str:
    """Return ``"Ixx Iyy Izz Ixy Ixz Iyz"`` from a symmetric 3x3 tensor."""
    arr = np.asarray(tensor, dtype=float)
    components = (
        arr[0, 0],
        arr[1, 1],
        arr[2, 2],
        arr[0, 1],
        arr[0, 2],
        arr[1, 2],
    )
    return " ".join(repr(float(component)) for component in components)


def _append_text(parent: ET.Element, tag: str, text: str) -> None:
    """Append ``<tag>text</tag>`` to *parent*."""
    child = ET.SubElement(parent, tag)
    child.text = text
