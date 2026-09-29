"""Writer for MuJoCo MJCF ``<body><inertial>`` elements.

The writer is the inverse of :func:`anthropometrics.readers.mjcf_body.read_mjcf_body`.
Given a canonical :class:`SegmentProperties` it emits a single
``<body>`` element with a single ``<inertial>`` child whose attributes
exactly satisfy the MJCF schema.

Design choices
--------------
* **Always emit ``fullinertia``.** MJCF's ``diaginertia`` cannot
  preserve off-diagonal terms; emitting ``fullinertia`` guarantees a
  loss-less round-trip even when the principal-axis frame happens to
  align with the body frame. The reader handles both forms, so input
  produced by other tools is still accepted.
* **Vendor-namespaced attributes carry the canonical anthropometric
  metadata MJCF cannot natively express** (segment ``length_m``,
  ``source_method``, ``source_subject_height_m``,
  ``source_subject_mass_kg``, optional marker labels, and the
  ``body_part_id`` when it differs from the body name). Stripping the
  ``ud:`` namespace yields a perfectly valid MJCF document for tools
  that ignore unknown attributes (MuJoCo itself does).
* **Float formatting uses ``repr`` precision**. The ``repr`` of a
  float is the shortest decimal string that round-trips to the
  identical IEEE-754 double — exactly what the round-trip contract
  (``rtol=1e-9``, ``atol=1e-12``) requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from ..segment_properties import SegmentProperties


# --------------------------------------------------------------------------- #
# Vendor-extension attribute names — kept synchronised with the reader.       #
# --------------------------------------------------------------------------- #
_UD_NAME = "ud:name"
_UD_BODY_PART_ID = "ud:body_part_id"
_UD_LENGTH_M = "ud:length_m"
_UD_SOURCE_METHOD = "ud:source_method"
_UD_SOURCE_HEIGHT_M = "ud:source_subject_height_m"
_UD_SOURCE_MASS_KG = "ud:source_subject_mass_kg"
_UD_PROXIMAL_MARKER = "ud:proximal_marker"
_UD_DISTAL_MARKER = "ud:distal_marker"


def write_mjcf_body(props: SegmentProperties) -> ET.Element:
    """Serialise *props* to an MJCF ``<body>`` element.

    Args:
        props: Validated :class:`SegmentProperties` to serialise. The
            caller is responsible for ensuring the instance was
            constructed via the dataclass (which enforces every
            physical invariant); the writer performs no additional
            validation.

    Returns:
        An ``xml.etree.ElementTree.Element`` rooted at ``<body>`` with
        the single required ``<inertial>`` child. The element is not
        attached to any tree; callers may insert it wherever needed.
    """
    body = ET.Element("body", attrib={"name": props.name})
    inertial = ET.SubElement(
        body,
        "inertial",
        attrib={
            "pos": _format_floats(props.com_xyz_m.tolist()),
            "mass": _format_float(props.mass_kg),
            "fullinertia": _format_floats(_full_inertia_six(props)),
        },
    )
    _attach_vendor_metadata(inertial, props)
    return body


# --------------------------------------------------------------------------- #
# Internals.                                                                  #
# --------------------------------------------------------------------------- #
def _full_inertia_six(props: SegmentProperties) -> list[float]:
    """Return the six unique entries of the inertia tensor in MJCF order.

    MJCF's ``fullinertia`` attribute orders the six unique entries as
    ``Ixx Iyy Izz Ixy Ixz Iyz``. We average the symmetric off-diagonal
    pairs (``[i, j]`` and ``[j, i]``) to absorb any floating-point
    asymmetry the dataclass tolerates (``atol=1e-9``).
    """
    tensor = props.inertia_tensor
    ixx = float(tensor[0, 0])
    iyy = float(tensor[1, 1])
    izz = float(tensor[2, 2])
    ixy = 0.5 * (float(tensor[0, 1]) + float(tensor[1, 0]))
    ixz = 0.5 * (float(tensor[0, 2]) + float(tensor[2, 0]))
    iyz = 0.5 * (float(tensor[1, 2]) + float(tensor[2, 1]))
    return [ixx, iyy, izz, ixy, ixz, iyz]


def _attach_vendor_metadata(elem: ET.Element, props: SegmentProperties) -> None:
    """Attach ``ud:*`` attributes that carry non-MJCF anthropometric fields."""
    elem.set(_UD_NAME, props.name)
    if props.body_part_id != props.name:
        elem.set(_UD_BODY_PART_ID, props.body_part_id)
    elem.set(_UD_LENGTH_M, _format_float(props.length_m))
    elem.set(_UD_SOURCE_METHOD, props.source_method)
    elem.set(_UD_SOURCE_HEIGHT_M, _format_float(props.source_subject_height_m))
    elem.set(_UD_SOURCE_MASS_KG, _format_float(props.source_subject_mass_kg))
    if props.proximal_marker is not None:
        elem.set(_UD_PROXIMAL_MARKER, props.proximal_marker)
    if props.distal_marker is not None:
        elem.set(_UD_DISTAL_MARKER, props.distal_marker)


def _format_float(value: float) -> str:
    """Return the shortest decimal string that round-trips to *value*."""
    return repr(float(value))


def _format_floats(values: list[float]) -> str:
    """Format an iterable of floats as a single space-separated MJCF token."""
    return " ".join(_format_float(v) for v in values)
